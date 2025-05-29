import torch
from omegaconf import OmegaConf
import numpy as np

from data import get_dataloader

def grammar_evals(cfg, model, grammar, work_dir, device, dt, seed):
    model.eval()
    dataloader = get_dataloader(cfg, work_dir, seed)

    with torch.no_grad():
        for seqs, classes in tqdm(dataloader, desc=f"Epoch {e+1}"):
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device, dtype=dt):
                output = model(seqs, classes, return_type="logits")
                print(output.shape)
                # pred = torch.argmax(output, dim=)

        return None
