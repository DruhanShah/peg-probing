import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


# Configuration dataclasses

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
                 attn_dir = "bidirectional",
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
        self.attn_dir = attn_dir
        self.act_fn = act_fn
        self.act_cache = act_cache
        self.dtype = dtype
        self.device = device
        self.seed = seed


class ProbeConfig:

    def __init__(self,
                 d_m = 512,
                 d_mlp = 1024,
                 linear = True,
                 act_fn = "relu",
                 dtype = torch.float32,
                 device = "cpu",
                 seed = 42,
                 checkpoint = None
                 ):
        self.d_m = d_m
        self.d_mlp = d_mlp
        self.linear = linear
        self.act_fn = act_fn
        self.dtype = dtype
        self.device = device
        self.seed = seed


# Actual model components

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Linear(cfg.d_m, cfg.n_h * cfg.d_h, bias=False, dtype=cfg.dtype)
        self.W_K = nn.Linear(cfg.d_m, cfg.n_h * cfg.d_h, bias=False, dtype=cfg.dtype)
        self.W_V = nn.Linear(cfg.d_m, cfg.n_h * cfg.d_h, bias=False, dtype=cfg.dtype)
        self.W_O = nn.Linear(cfg.n_h * cfg.d_h, cfg.d_m, bias=False, dtype=cfg.dtype)

        self.scale = 1.0 / (cfg.d_h ** 0.5)

        if cfg.attn_dir == "causal":
            causal_mask = torch.tril(torch.ones(cfg.n_ctx, cfg.n_ctx, dtype=cfg.dtype))
            self.register_buffer("causal_mask", causal_mask)

    def forward(self, x, attention_mask=None):

        q = einops.rearrange(self.W_Q(x), "b s (h d) -> b h s d", h=self.cfg.n_h)
        k = einops.rearrange(self.W_K(x), "b s (h d) -> b h s d", h=self.cfg.n_h)
        v = einops.rearrange(self.W_V(x), "b s (h d) -> b h s d", h=self.cfg.n_h)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            mask_expanded = einops.repeat(attention_mask, "b s -> b 1 1 s")
            scores = scores.masked_fill(mask_expanded == 0, float("-inf"))

        if self.cfg.attn_dir == "causal":
            seq_len = scores.size(2)
            causal_mask = self.causal_mask[:seq_len, :seq_len]
            causal_mask = einops.repeat(causal_mask, "l m -> b h l m",
                                        b=scores.size(0), h=scores.size(1))
            scores = scores.masked_fill(causal_mask == 0, float("-inf"))

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


class TransformerBlock(nn.Module):
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
