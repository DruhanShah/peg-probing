import torch
from tqdm import tqdm

from data import get_dataloader


def validation(cfg, model, device):
    dataloader = get_dataloader(
        cfg.lang, cfg.model_type,
        cfg.eval,
        cfg.work_dir, cfg.seed,
        kind="PEG")
    dt = torch.bfloat16 if cfg.train.bf16 else torch.float32

    results = {
        "loss": [],
        "accuracy": [],
    }
    
    with torch.no_grad():
        for i, _in in enumerate(dataloader):
            inputs = _in["inputs"].to(device)
            masks = _in["masks"].to(device)
            outputs = _in["outputs"].to(device)
            B = inputs.shape[0]

            with torch.amp.autocast(device_type=device, dtype=dt):
                # Forward pass through the model
                _out = model(inputs, outputs, mask=masks,
                             return_type=["logits", "loss"])
                logits = _out["logits"].to(device)
                loss = _out["loss"].item()

                # Get vector of prediction successes
                pred = (logits.argmax(dim=-1)
                        if cfg.model_type == "generator"
                        else logits > 0)
                success = (pred == outputs).tolist()

                # Calculate accuracy
                acc = 0
                for i in range(B):
                    acc += (sum(success[i])/len(success[i])
                            if isinstance(success[i], list)
                            else success[i])
                acc /= B

            results["loss"].append(loss)
            results["accuracy"].append(acc)

    return results


def ps_intervention(*args):
    pass

