import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from data import get_dataloader

def grammar_evals(cfg, model, device):
    dataloader = get_dataloader(cfg.eval, cfg.work_dir, cfg.seed)
    dt = torch.bfloat16 if cfg.train.bf16 else torch.float32

    results = {
        "loss": [],
        "accuracy": [],
    }
    
    with torch.no_grad():
        for i, _in in enumerate(dataloader):
            seqs = _in["input_ids"].to(device)
            masks = _in["masks"].to(device)
            classes = _in["labels"].to(device).squeeze()
            B = seqs.shape[0]
            with torch.amp.autocast(device_type=device, dtype=dt):
                output = model(seqs, classes, mask=masks, return_type="logits")
                loss = model.loss(output, classes).item()
                pred = (output > 0).to(device)
                success = (pred == classes).tolist()
                acc = sum(success)/B if isinstance(success, list) else int(success)

            results["loss"].append(loss)
            results["accuracy"].append(acc)

    return results
