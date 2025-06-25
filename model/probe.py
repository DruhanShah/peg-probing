import torch
import torch.nn as nn


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


class Probe(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if cfg.linear:
            self.probe = nn.Linear(cfg.d_m, 1, bias=False, dtype=cfg.dtype)
        else:
            act = {
                "relu": nn.ReLU(),
                "gelu": nn.GELU(),
                "silu": nn.SiLU(),
                "tanh": nn.Tanh(),
                "sigmoid": nn.Sigmoid(),
            }.get(cfg.act_fn, nn.ReLU())
            self.probe = nn.Sequential(
                nn.Linear(cfg.d_m, cfg.d_mlp, bias=False, dtype=cfg.dtype),
                act,
                nn.Linear(cfg.d_mlp, 1, bias=False, dtype=cfg.dtype),
            )

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, y=None, return_type=["logits"]):
        y_pred = self.probe(x).squeeze()

        returnable = {}
        if "logits" in return_type:
            returnable["logits"] = y_pred
        if "loss" in return_type and y is not None:
            returnable["loss"] = self.loss(y_pred, y.float().squeeze())
        return returnable
