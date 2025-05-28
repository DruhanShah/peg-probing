import torch
import numpy as np

from data import get_dataloader
from utils import obj_to_dict

def grammar_evals(cfg, model, grammar, device, dt, seed):
    model.eval()
    dataloader = get_dataloader(**obj_to_dict(cfg), seed=seed)

    with torch.no_grad():
        for seqs, classes in tqdm(dataloader, desc=f"Epoch {e+1}"):
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device, dtype=dt):
                output = model(seqs, classes, return_type="logits")
                print(output.shape)
                # pred = torch.argmax(output, dim=)

        return None
