import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
from copy import deepcopy as copy
from abc import ABC, abstractmethod
from functools import partial

from .components import Block


class BaseModel(nn.Module, ABC):
    """Abstract base class for transformer models with activation caching."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.token_embeddings = nn.Embedding(cfg.d_vocab, cfg.d_m, dtype=cfg.dtype)
        self.pos_embeddings = nn.Embedding(cfg.n_ctx, cfg.d_m, dtype=cfg.dtype)
        self.transformer = nn.ModuleList([Block(cfg)
                                          for _ in range(cfg.n_l)])
        self.ln_final = nn.LayerNorm(cfg.d_m, dtype=cfg.dtype, bias=cfg.bias)

        # Model-specific layers and loss (to be implemented by subclasses)
        self.head = self._init_model_head()
        self.loss = self._init_model_loss()

        self.to(cfg.device, dtype=cfg.dtype)

        # Activation caching
        self.activation_cache = {}
        self._hook_handles = []

        # GPT-2 style initialization
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('W_O.weight') or pn.endswith('fc2.weight'):
                torch.nn.init.normal_(p, mean=0.0,
                                      std=0.02/math.sqrt(2 * self.cfg.n_l))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @abstractmethod
    def _init_model_head(self):
        pass

    @abstractmethod
    def _init_model_loss(self):
        pass

    @abstractmethod
    def _compute_loss(self, x, y):
        pass

    @abstractmethod
    def _pool(self, x):
        pass

    def _cache_hook(self, name, module, input, output):
        self.activation_cache[name] = output.detach()

    def _register_hooks(self):
        self._remove_hooks()

        def add_hook(block, name):
            self._hook_handles.append(block.register_forward_hook(
                partial(self._cache_hook, name)
            ))

        add_hook(self.token_embeddings, "token_embeddings")
        add_hook(self.pos_embeddings, "pos_embeddings")
        for i, block in enumerate(self.transformer):
            block_name = f"block_{i}"
            add_hook(block, block_name)
            add_hook(block.attn, f"{block_name}.attn")
            add_hook(block.mlp, f"{block_name}.mlp")
            add_hook(block.ln1, f"{block_name}.ln1")
            add_hook(block.ln2, f"{block_name}.ln2")
        add_hook(self.ln_final, "ln_final")
        add_hook(self.head, "head")

    def _remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def reset_cache(self):
        self.activation_cache.clear()
        self._remove_hooks()

    def forward(self, x, y=None, return_type=["logits"]):
        if self.cfg.act_cache:
            self.reset_cache()
            self._register_hooks()

        B, N = x.shape

        token_embeds = self.token_embeddings(x.to(torch.long))
        position_ids = torch.arange(0, N,
                                    dtype=torch.long, device=self.cfg.device)
        pos_embeds = self.pos_embeddings(position_ids).to(self.cfg.dtype)
        pos_embeds = einops.repeat(pos_embeds, "n d -> b n d", b=B)
        x = token_embeds + pos_embeds

        for i, block in enumerate(self.transformer):
            if self.cfg.act_cache:
                key = f"block_{i}.attn_map"
                self.activation_cache[key] = block.get_attention(x)
            x = block(x)
        x = self.ln_final(x)

        x = self._pool(x)
        logits = self.head(x)

        returnable = {}
        if "logits" in return_type:
            returnable["logits"] = logits
        if "loss" in return_type:
            returnable["loss"] = (self._compute_loss(logits, y)
                                  if y is not None
                                  else None)
        if "cache" in return_type:
            returnable["cache"] = (self.activation_cache
                                   if self.cfg.act_cache
                                   else {})

        if self.cfg.act_cache:
            self._remove_hooks()

        return returnable

    @property
    def module_weights(self) -> dict:
        weights = {}
        for name, module in self.named_modules():
            if name == "" or isinstance(module, nn.ModuleList):
                continue

            module_params = {}
            if hasattr(module, "weight") and module.weight is not None:
                module_params["weight"] = module.weight
            if hasattr(module, "bias") and module.bias is not None:
                module_params["bias"] = module.bias

            if module_params:
                weights[name] = copy(module_params)
        return weights


class RecognizerModel(BaseModel):
    """Binary classification model."""

    def _init_model_head(self):
        return nn.Linear(self.cfg.d_m, 1, dtype=self.cfg.dtype)

    def _init_model_loss(self):
        return F.binary_cross_entropy_with_logits

    def _pool(self, x):
        return x[:, -1, :]  # Last token pooling

    def _compute_loss(self, logits, y):
        loss = self.loss(logits.squeeze(), y.float().squeeze(),
                         reduction='none')
        return loss


class GeneratorModel(BaseModel):
    """Generative language model."""

    def _init_model_head(self):
        return nn.Linear(self.cfg.d_m, self.cfg.d_vocab, dtype=self.cfg.dtype)

    def _init_model_loss(self):
        return F.cross_entropy

    def _pool(self, x):
        return x

    def _compute_loss(self, logits, y):
        B, N = logits.shape[0], logits.shape[1]
        logits = einops.rearrange(logits, 'b n v -> (b n) v')
        y = einops.rearrange(y, 'b n -> (b n)')
        loss = self.loss(logits, y,
                         ignore_index=self.cfg.pad_index,
                         reduction='none')
        loss = einops.rearrange(loss, '(b n) -> b n', b=B, n=N).mean(dim=1)
        return loss

    def generate(self, x, num_samples=1, temperature=1.0):
        self.eval()
        generated = x.clone()

        for _ in range(num_samples):
            with torch.no_grad():
                logits = self.forward(generated, return_type=["logits"])
                logits = logits["logits"]
                logits = logits[:, -1, :] / temperature  # Use last position
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated = torch.cat((generated, next_token), dim=1)

        self.train()
        return generated
