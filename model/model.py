import torch
import torch.nn as nn
from copy import deepcopy as copy
from abc import ABC, abstractmethod
from functools import partial
from einops import rearrange, repeat

from .components import TransformerConfig, TransformerBlock


class BaseModel(nn.Module, ABC):
    """Abstract base class for transformer models with activation caching."""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.token_embeddings = nn.Embedding(cfg.d_vocab, cfg.d_m, dtype=cfg.dtype)
        self.pos_embeddings = nn.Embedding(cfg.n_ctx, cfg.d_m, dtype=cfg.dtype)
        self.transforemr = nn.ModuleList([TransformerBlock(cfg)
                                          for _ in range(cfg.n_l)])
        self.ln_final = nn.LayerNorm(cfg.d_m, dtype=cfg.dtype)
        
        # Model-specific layers and loss (to be implemented by subclasses)
        self._init_head_and_loss()
        
        self.to(cfg.device, dtype=cfg.dtype)
        
        # Activation caching
        self.activation_cache = {}
        self._hook_handles = []

    @abstractmethod
    def _init_head_and_loss(self):
        pass

    @abstractmethod
    def _compute_output_and_loss(self, x, y, return_type):
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
        for i, block in enumerate(self.transforemr):
            block_name = f"block_{i}"
            add_hook(block, block_name)
            add_hook(block.attn, f"{block_name}.attn")
            add_hook(block.mlp, f"{block_name}.mlp")
            add_hook(block.ln1, f"{block_name}.ln1")
            add_hook(block.ln2, f"{block_name}.ln2")
        add_hook(self.ln_final, "ln_final")
        self._add_head_hook()

    @abstractmethod
    def _add_head_hook(self):
        pass

    def _remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def reset_cache(self):
        self.activation_cache.clear()
        self._remove_hooks()

    def forward(self, x, y=None, mask=None, return_type=["logits"]):
        if self.cfg.act_cache:
            self.reset_cache()
            self._register_hooks()
        
        B, N = x.shape

        token_embeds = self.token_embeddings(x.to(torch.long))
        position_ids = torch.arange(0, N, dtype=torch.long, device=self.cfg.device)
        pos_embeds = self.pos_embeddings(position_ids).to(self.cfg.dtype)
        pos_embeds = repeat(pos_embeds, "n d -> b n d", b=B)

        x = token_embeds + pos_embeds
        x = x.to(self.cfg.dtype)

        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=self.cfg.device)

        for block in self.transforemr:
            x = block(x, attention_mask=mask)
        x = self.ln_final(x)

        returnable = self._compute_output_and_loss(x, y, return_type)
        if "cache" in return_type:
            returnable["cache"] = self.activation_cache if self.cfg.act_cache else {}

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
    
    def _init_head_and_loss(self):
        self.classifier = nn.Linear(self.cfg.d_m, 1, dtype=self.cfg.dtype)
        self.loss = nn.BCEWithLogitsLoss()

    def _add_head_hook(self):
        self._hook_handles.append(self.classifier.register_forward_hook(
            partial(self._cache_hook, "classifier")
        ))

    def _compute_output_and_loss(self, x, y, return_type):
        pooled_output = x[:, -1, :]  # Last token pooling
        logits = self.classifier(pooled_output).squeeze()

        returnable = {}
        if "logits" in return_type:
            returnable["logits"] = logits
        if "loss" in return_type:
            returnable["loss"] = self.loss(logits, y.float())
        
        return returnable


class GeneratorModel(BaseModel):
    """Generative language model."""
    
    def _init_head_and_loss(self):
        self.unembed = nn.Linear(self.cfg.d_m, self.cfg.d_vocab, dtype=self.cfg.dtype)
        self.loss = nn.CrossEntropyLoss()

    def _add_head_hook(self):
        self._hook_handles.append(self.unembed.register_forward_hook(
            partial(self._cache_hook, "unembedding")
        ))

    def _compute_output_and_loss(self, x, y, return_type):
        logits = self.unembed(x)

        returnable = {}
        if "logits" in return_type:
            returnable["logits"] = logits
        if "loss" in return_type:
            logits_flat = rearrange(logits, 'b n v -> (b n) v')
            y_flat = rearrange(y, 'b n -> (b n)')
            returnable["loss"] = self.loss(logits_flat, y_flat)
        
        return returnable

    def generate(self, x, num_samples=1, temperature=1.0):
        self.eval()
        generated = x.clone()

        for _ in range(num_samples):
            with torch.no_grad():
                logits = self.forward(generated, return_type=["logits"])["logits"]
                logits = logits[:, -1, :] / temperature  # Use last position
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated = torch.cat((generated, next_token), dim=1)

        return generated


def create_model(model_type, cfg):
    if model_type == "recognizer":
        return RecognizerModel(cfg)
    elif model_type == "generator":
        return GeneratorModel(cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types are 'recognizer' and 'generator'.")
