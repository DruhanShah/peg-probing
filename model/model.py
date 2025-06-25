import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from copy import deepcopy as copy
from functools import partial


class TransformerConfig:

    def __init__(self,
                 n_l = 6,
                 d_m = 512,
                 n_h = 8,
                 d_h = None,
                 d_mlp = None,
                 d_vocab = 10000,
                 n_ctx = 512,
                 attn_only = False,
                 act_fn="gelu",
                 act_cache = False,
                 dtype = torch.float32,
                 device = "cpu",
                 seed = 42,
                 checkpoint = None,
                 ):
        self.n_l = n_l
        self.d_m = d_m
        self.n_h = n_h
        self.d_h = d_h if d_h is not None else d_m // n_h
        self.d_mlp = d_mlp if d_mlp is not None else 4 * d_m
        self.d_vocab = d_vocab
        self.n_ctx = n_ctx
        self.attn_only = attn_only
        self.act_fn = act_fn
        self.act_cache = act_cache
        self.dtype = dtype
        self.device = device
        self.seed = seed


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Linear(cfg.d_m, cfg.n_h * cfg.d_h, bias=False, dtype=cfg.dtype)
        self.W_K = nn.Linear(cfg.d_m, cfg.n_h * cfg.d_h, bias=False, dtype=cfg.dtype)
        self.W_V = nn.Linear(cfg.d_m, cfg.n_h * cfg.d_h, bias=False, dtype=cfg.dtype)
        self.W_O = nn.Linear(cfg.n_h * cfg.d_h, cfg.d_m, bias=False, dtype=cfg.dtype)

        self.scale = 1.0 / (cfg.d_h ** 0.5)

    def forward(self, x, attention_mask=None):

        q = einops.rearrange(self.W_Q(x), "b s (h d) -> b h s d", h=self.cfg.n_h)
        k = einops.rearrange(self.W_K(x), "b s (h d) -> b h s d", h=self.cfg.n_h)
        v = einops.rearrange(self.W_V(x), "b s (h d) -> b h s d", h=self.cfg.n_h)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            mask_expanded = einops.repeat(attention_mask, "b s -> b 1 1 s")
            scores = scores.masked_fill(mask_expanded == 0,
                                        float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        output_heads = torch.matmul(attn_weights, v)

        output = einops.rearrange(output_heads, "b h s d -> b s (h d)")
        output = self.W_O(output)
        return output


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(cfg.d_m, cfg.d_mlp, dtype=cfg.dtype)
        self.fc2 = nn.Linear(cfg.d_mlp, cfg.d_m, dtype=cfg.dtype)
        if cfg.act_fn == "gelu":
            self.act = F.gelu
        elif cfg.act_fn == "relu":
            self.act = F.relu
        else:
            raise ValueError(f"Unsupported activation function: {cfg.act_fn}")

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class EncoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)

        self.ln1 = nn.LayerNorm(cfg.d_m, dtype=cfg.dtype)
        self.ln2 = nn.LayerNorm(cfg.d_m, dtype=cfg.dtype)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class RecognizerModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.token_embeddings = nn.Embedding(cfg.d_vocab, cfg.d_m, dtype=cfg.dtype)
        self.pos_embeddings = nn.Embedding(cfg.n_ctx, cfg.d_m, dtype=cfg.dtype)
        self.transforemr = nn.ModuleList([EncoderBlock(cfg)
                                          for _ in range(cfg.n_l)])
        self.ln_final = nn.LayerNorm(cfg.d_m, dtype=cfg.dtype)
        self.classifier = nn.Linear(cfg.d_m, 1, dtype=cfg.dtype)

        self.loss = nn.BCEWithLogitsLoss()

        self.to(cfg.device, dtype=cfg.dtype)

        self.activation_cache = {}
        self._hook_handles = []

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
        add_hook(self.classifier, "classifier")

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

        x = token_embeds + pos_embeds
        x = x.to(self.cfg.dtype)

        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=self.cfg.device)

        for block in self.transforemr:
            x = block(x, attention_mask=mask)
        x = self.ln_final(x)

        pooled_output = x[:, -1, :]
        logits = self.classifier(pooled_output).squeeze()

        if self.cfg.act_cache:
            self._remove_hooks()

        returnable = {}
        if "logits" in return_type:
            returnable["logits"] = logits
        if "loss" in return_type:
            returnable["loss"] = self.loss(logits, y.float())
        if "cache" in return_type:
            returnable["cache"] = self.activation_cache if self.cfg.act_cache else {}
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
