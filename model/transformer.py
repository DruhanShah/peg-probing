import torch
import torch.nn as nn
import copy
import math
from random import choices
from dataclasses import dataclass


def set_seed(seed=0):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PositionalEncoding(nn.Module):

    def __init__(self, d_m, N_max):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(N_max, d_m)
        position = torch.arange(0, N_max, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_m, 2).float() * -(math.log(10000) / d_m))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class FeedForward(nn.Module):

    def __init__(self, d_m, d_mlp):
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(d_m, d_mlp)
        self.fc2 = nn.Linear(d_mlp, d_m)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

    
class MultiHeadAttention(nn.Module):

    def __init__(self, d_m, heads):
        super(MultiHeadAttention, self).__init__()
        assert d_m % heads == 0, "d_model must be divisible by num_heads"
        
        self.d_m = d_m
        self.heads = heads
        self.d_k = d_m // heads
        
        self.W_q = nn.Linear(d_m, d_m)
        self.W_k = nn.Linear(d_m, d_m)
        self.W_v = nn.Linear(d_m, d_m)
        self.W_o = nn.Linear(d_m, d_m)

    def sdp_attention(self, Q, K, V, mask=None):
        attention = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)

        attention_probs = torch.softmax(attention, dim=1)
        A = torch.matmul(attention_probs, V)
        return A

    def split_heads(self, x):
        B, N, _ = x.size()
        return x.view(B, N, self.heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        B, _, N, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(B, N, self.d_m)
    
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_q(K))
        V = self.split_heads(self.W_q(V))

        A = self.sdp_attention(Q, K, V, mask)
        O = self.W_o(self.combine_heads(A))
        return O
    
class Encoder(nn.Module):

    def __init__(self, d_m, d_mlp, n_h, dropout):
        super(Encoder, self).__init__()
        self.self_attention = MultiHeadAttention(d_m, n_h)
        self.feed_forward = FeedForward(d_m, d_mlp)
        self.norm1 = nn.LayerNorm(d_m)
        self.norm2 = nn.LayerNorm(d_m)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention))
        feed_forward = self.feed_forward(x)
        x = self.norm2(x + self.dropout(feed_forward))
        return x
    

class Decoder(nn.Module):

    def __init__(self, d_m, d_mlp, n_h, dropout):
        super(Decoder, self).__init__()
        self.self_attention = MultiHeadAttention(d_m, n_h)
        self.cross_attention = MultiHeadAttention(d_m, n_h)
        self.feed_forward = FeedForward(d_m, d_mlp)
        self.norm1 = nn.LayerNorm(d_m)
        self.norm2 = nn.LayerNorm(d_m)
        self.norm3 = nn.LayerNorm(d_m)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc, mask_s, mask_t):
        attention = self.self_attention(x, x, x, mask_t)
        x = self.norm1(x + self.dropout(attention))
        attention = self.cross_attention(x, enc, enc, mask_s)
        x = self.norm2(x + self.dropout(attention))
        feed_forward = self.feed_forward(x)
        x = self.norm3(x + self.dropout(feed_forward))
        return x
    
    

class Transformer(nn.Module):

    def __init__(self, cfg):
        super(Transformer, self).__init__()
        self.device = cfg.device

        self.embed_enc = nn.Embedding(cfg.s_vocab, cfg.d_m).to(self.device)
        self.embed_dec = nn.Embedding(cfg.t_vocab, cfg.d_m).to(self.device)
        self.positional_encoding = PositionalEncoding(cfg.d_m, cfg.N_max).to(self.device)

        self.encoders = nn.ModuleList([
            Encoder(cfg.d_m, cfg.d_mlp, cfg.n_h, cfg.dropout).to(self.device)
            for _ in range(cfg.L_enc)
        ])
        self.decoders = nn.ModuleList([
            Decoder(cfg.d_m, cfg.d_mlp, cfg.n_h, cfg.dropout).to(self.device)
            for _ in range(cfg.L_dec)
        ])

        self.fc = nn.Linear(cfg.d_m, cfg.t_vocab).to(self.device)
        self.dropout = nn.Dropout(cfg.dropout).to(self.device)

    def mask_gen(self, src, tgt):
        mask_s = (src != 0).unsqueeze(1).unsqueeze(2).to(self.device)
        mask_t = (tgt != 0).unsqueeze(1).unsqueeze(3).to(self.device)
        n = tgt.size(1)
        no_peek = (1 - torch.triu(torch.ones(1, n, n), diagonal=1)).bool().to(self.device)
        mask_t = mask_t & no_peek
        return mask_s, mask_t
    
    def forward(self, x, y):
        mask_s, mask_t = self.mask_gen(x, y)
        s_embed = self.dropout(self.positional_encoding(self.embed_enc(x)))
        t_embed = self.dropout(self.positional_encoding(self.embed_dec(y)))

        enc_output = s_embed
        for encoder in self.encoders:
            enc_output = encoder(enc_output, mask_s)
        dec_output = t_embed
        for decoder in self.decoders:
            dec_output = decoder(dec_output, enc_output, mask_s, mask_t)
            
        return self.fc(dec_output)

    @torch.no_grad()
    def sample(self, prompt, max_len):
        for _ in range(max_len):
            print(prompt)
            pred = self.forward(prompt, prompt)
            prob = torch.softmax(pred[:, -1], dim=-1)
            print(prob)
            pred = choices(range(prob.size(1)), prob.squeeze().tolist())[0]
            prompt = torch.cat([prompt, torch.LongTensor([[pred]]).to(self.device)], dim=-1)
        return prompt


@dataclass
class TransformerConfig:

    d_m: int = -1
    d_mlp: int = -1
    n_h: int = -1
    L_enc: int = -1
    L_dec: int = -1
    N_max: int = -1
    dropout: float = 0.5
    s_vocab: int = -1
    t_vocab: int = -1
    device: str = "cpu"

    def __post_init__(self):
        for k, v in self.__dict__.items():
            if v == -1:
                raise RuntimeError(f"Missing value for {k}")

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, config):
        keylist = TransformerConfig.__dict__.keys()
        fil = dict(filter(lambda x: x[0] in keylist, config.items()))
        return cls(**fil)
