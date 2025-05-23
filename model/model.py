import torch
import torch.nn as nn
from transformer_lens import HookedEncoder, HookedTransformerConfig


class RecognizerModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.transformer = HookedEncoder(cfg)
        self.classifier = nn.Linear(cfg.d_model, 1)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, y=None, return_type="logits"):
        x = self.transformer(x, return_type="logits")
        x = x[:, -1, :] # Last tokens? TODO: Get clear on this
        y_pred = self.classifier(x)

        if return_type == "logits":
            return y_pred.squeeze(-1)
        elif return_type == "loss":
            return self.loss(y_pred, y)
