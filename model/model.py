import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformer_lens import HookPoint()
from einops import repeat


class LayerNorm(nn.Module):

    def __init__(self, cfg):
       super().__init__()
       self.cfg = cfg
       self.eps = self.cfg.eps
       self.len = self.cfg.d_model

       self.w = nn.Parameter(torch.ones(self.len, dtype=self.cfg.dtype))
       self.b = nn.Parameter(torch.zeros(self.len, dtype=self.cfg.dtype))

       self.hook_scale = HookPoint() # [b, pos, 1]
       self.hook_norm = HookPoint() # [b, pos, len]

    def forward(self, x):
        x -= x.mean(-1, keepdim=True)
        scale = self.hook_scale((x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt())
        x /= scale
        return self.hook_norm(x * self.w + self.b).to(self.cfg.dtype)


class Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mask = torch.tril(torch.ones((self.cfg.n_ctx, self.cfg.n_ctx)).bool())


    def attention(self, q_in, k_in, v_in, mask, drop=None):
        q = q_in * self.W_q + self.b_q
        k = k_in * self.W_k + self.b_k
        v = v_in * self.W_v + self.b_v

        return F.scaled_dot_product_attention(q, k, v, mask, dropout_p=drop)

    def forward(self, x):
        attention = 


class MLP(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg


class HookedEncoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.attn = Attention(self.cfg)
        self.ln1 = LayerNorm(self.cfg)
        self.mlp = MLP(self.cfg)
        self.ln2 = LayerNorm(self.cfg)

        self.hook_q_input = HookPoint()  # [b, pos, n_h, d_m]
        self.hook_k_input = HookPoint()  # [b, pos, n_h, d_m]
        self.hook_v_input = HookPoint()  # [b, pos, n_h, d_m]

        self.hook_attn_out = HookPoint()  # [b, pos, d_m]
        self.hook_mlp_out = HookPoint()  # [b, pos, d_m]
        self.hook_resid_pre = HookPoint()  # [b, pos, d_m]
        self.hook_resid_mid = HookPoint()  # [b, pos, d_m]
        self.hook_resid_mid_norm = HookPoint()  # [b, pos, d_m]
        self.hook_resid_post = HookPoint()  # [b, pos, d_m]
        self.hook_resid_post_norm = HookPoint()  # [b, pos, d_m]

    def forward(self, x):
        resid_pre = self.hook_resid_pre(x)

        q = self.hook_q_input(repeat(resid_pre, "bpd -> bpnd", n=self.cfg.n_h))
        k = self.hook_k_input(repeat(resid_pre, "bpd -> bpnd", n=self.cfg.n_h))
        v = self.hook_v_input(repeat(resid_pre, "bpd -> bpnd", n=self.cfg.n_h))

        attention = self.hook_attn_out(self.attn(
            q, k, v,
            attention_mask=attention_mask
        ))
        resid_mid = self.hook_resid_mid(resid_pre + attention)
        resid_mid_norm = self.hook_resid_mid_norm(self.ln1(resid_mid))

        mlp_out = self.hook_mlp_out(self.mlp(resid_mid_norm))
        resid_post = self.hook_resid_post(resid_mid_norm + mlp_out)
        resid_post_norm = self.hook_resid_post_norm(self.ln2(resid_post))

        return resid_post_norm


class HookedModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.blocks = nn.ModuleList([HookedEncoder(self.cfg)
                                     for _ in range(self.cfg.n_layers)])
        self.class_head = nn.Linear(self.cfg.d_m, 2)

    def forward(self, x):
        pass
