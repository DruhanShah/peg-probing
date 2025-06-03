import torch
import torch.nn as nn
from transformer_lens import HookedEncoder, HookedTransformerConfig


class RecognizerModel(nn.Module):

    def __init__(self, cfg, dev):
        super().__init__()

        self.transformer = HookedEncoder(cfg)
        self.classifier = nn.Linear(cfg.d_model, 1)
        self.loss = nn.BCEWithLogitsLoss()
        self.device = dev

    def forward(self, x, y=None, return_type="logits"):
        x, y = x.to(self.device), y.to(self.device)
        x = self.transformer.encoder_output(x)
        x = x[:, -1, :]
        y_pred = self.classifier(x).squeeze()

        if return_type == "logits":
            return y_pred
        elif return_type == "loss":
            loss = self.loss(y_pred, y)
            if torch.isnan(loss):
                print(y_pred)
            return self.loss(y_pred, y)
