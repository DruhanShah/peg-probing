import torch
import torch.nn as nn
from transformer_lens import HookedEncoder


class RecognizerModel(nn.Module):

    def __init__(self, cfg, dev):
        super().__init__()

        self.cfg = cfg
        self.transformer = HookedEncoder(cfg)
        self.classifier = nn.Linear(cfg.d_model, 1)
        self.loss = nn.BCEWithLogitsLoss()
        self.device = dev

        self.transformer.setup()

    def forward(self, x, y=None, mask=None, return_type=["logits"]):
        x, y = x.to(self.device), y.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        _, cache = self.transformer.run_with_cache(
            x, one_zero_attention_mask=mask, return_type="logits",
            return_cache_object=False,
        )
        last = self.cfg.n_layers - 1
        x = cache[f"blocks.{last}.hook_normalized_resid_post"]
        x = x.mean(dim=1)
        y_pred = self.classifier(x).squeeze()
        cache["classifier.hook_output"] = y_pred

        returnable = []
        
        if "logits" in return_type:
            returnable.append(y_pred)
        if "loss" in return_type:
            if y is None:
                raise ValueError("y must be provided to compute loss")
            loss = self.loss(y_pred, y)
            returnable.append(loss)
        if "cache" in return_type:
            returnable.append(cache) 
        return returnable[0] if len(returnable) == 1 else tuple(returnable)
