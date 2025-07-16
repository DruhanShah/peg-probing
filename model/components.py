import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from dataclasses import dataclass


# Configuration dataclasses

@dataclass
class TransformerConfig:
    n_l: int = 6
    d_m: int = 512
    n_h: int = 8
    d_h: int = None
    d_mlp: int = None
    d_vocab: int = 10000
    n_ctx: int = 512
    dropout: float = 0.1
    bias: bool = True
    attn_only: bool = False
    attn_dir: str = "bidirectional"
    act_fn: str = "gelu"
    act_cache: bool = False
    pad_index: int = 0
    dtype: type = torch.float32
    device: str = "cpu"
    seed: int = 42
    checkpoint: int = None

    def __post_init__(self):
        self.d_h = self.d_h if self.d_h is not None else self.d_m // self.n_h
        self.d_mlp = self.d_mlp if self.d_mlp is not None else 4 * self.d_m

        if self.attn_dir not in ["bidirectional", "causal"]:
            raise ValueError(f"Unsupported attention direction: {self.attn_dir}")
        if self.d_h * self.n_h != self.d_m:
            raise ValueError("d_m must be divisible by n_h for multi-head attention.")


@dataclass
class ProbeConfig:
    d_m: int = 512,
    d_mlp: int = 1024,
    linear: bool = True,
    act_fn: str = "relu",
    dtype: type = torch.float32,
    device: str = "cpu",
    seed: int = 42,
    checkpoint: int = None


# Actual model components

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_QKV = nn.Linear(cfg.d_m, 3 * cfg.d_m, bias=cfg.bias, dtype=cfg.dtype)
        self.W_O = nn.Linear(cfg.d_m, cfg.d_m, bias=cfg.bias, dtype=cfg.dtype)

    def forward(self, x):

        q, k, v = self.W_QKV(x).split(self.cfg.d_m, dim=2)
        q = einops.rearrange(q, "b s (h d) -> b h s d", h=self.cfg.n_h)
        k = einops.rearrange(k, "b s (h d) -> b h s d", h=self.cfg.n_h)
        v = einops.rearrange(v, "b s (h d) -> b h s d", h=self.cfg.n_h)

        attention = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.cfg.dropout,
            is_causal=(self.cfg.attn_dir == "causal"),
        )

        output = einops.rearrange(attention, "b h s d -> b s (h d)")
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


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.ln1 = nn.LayerNorm(cfg.d_m, bias=cfg.bias, dtype=cfg.dtype)
        self.ln2 = nn.LayerNorm(cfg.d_m, bias=cfg.bias, dtype=cfg.dtype)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
