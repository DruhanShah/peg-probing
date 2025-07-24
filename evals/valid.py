import torch
from tqdm import tqdm

from data import PEG, get_dataloader


def _calculate_binary_accuracy(predictions, targets):
    success = (predictions == targets).float()

    if success.dim() > 1:
        success = success.mean(dim=1)
    return success.mean().item()


def _get_autocast_context(device, bf16):
    dt = torch.bfloat16 if bf16 else torch.float32
    return torch.amp.autocast(device_type=device.split(":")[0], dtype=dt)


def binary_validation(cfg, model, device):
    dataloader = get_dataloader(
        cfg.lang, cfg.model_type,
        cfg.eval,
        cfg.work_dir, cfg.seed,
        kind="PEG", quiet=True
    )
    dt = torch.bfloat16 if cfg.train.bf16 else torch.float32

    results = {"loss": [], "accuracy": []}
    debug_results = {"outputs": []}
    
    with torch.no_grad():
        for i, _in in enumerate(dataloader):
            inputs = _in["inputs"].to(device)
            outputs = _in["outputs"].to(device)
            B = inputs.shape[0]

            with _get_autocast_context(device, cfg.train.bf16):
                # Forward pass through the model
                _out = model(inputs, outputs,
                             return_type=["logits", "loss"])
                logits = _out["logits"].to(device)
                loss = _out["loss"].mean().item()

                # Get predictions and accuracy
                pred = logits > 0
                acc = _calculate_binary_accuracy(pred, outputs)

            results["loss"].append(loss)
            results["accuracy"].append(acc)

    return results, debug_results


def grammaticality_validation(cfg, model, device):
    grammar = PEG(cfg.lang, max_length=cfg.eval.max_len)
    dt = torch.bfloat16 if cfg.train.bf16 else torch.float32

    results = {"grammaticality": []}
    debug_results = {"outputs": []}
    
    with torch.no_grad():
        template = ""
        inputs = [
            torch.tensor(grammar.tokenize_string(template),
                         dtype=torch.long)[:-1].to(device).unsqueeze(0)
            for _ in range(cfg.eval.num_samples)
        ]

        for _in in inputs:
            with _get_autocast_context(device, cfg.train.bf16):
                _out = model.generate(_in, num_samples=cfg.eval.max_len,
                                      temperature=cfg.eval.temperature).tolist()
                accuracy = 0
                for pred in _out:
                    output = grammar.detokenize_string(pred, clean=True)
                    grammaticality = grammar.grammar_check(output)
                    accuracy += 1 if grammaticality else 0
                    debug_results["outputs"].append(output)
                accuracy /= len(_out)
                results["grammaticality"].append(accuracy)

    return results, debug_results


def probe_validation(cfg, model, device, probe):
    dataloader = get_dataloader(
        cfg.lang, cfg.model_type,
        cfg.eval,
        cfg.work_dir, cfg.seed,
        kind="PS", quiet=True
    )
    dt = torch.bfloat16 if cfg.train.bf16 else torch.float32

    results = {"loss": [], "accuracy": []}
    debug_results = {"outputs": []}

    with torch.no_grad():
        for _in in dataloader:
            inputs = _in["inputs"].to(device)
            outputs = _in["outputs"].to(device)
            B = inputs.shape[0]

            with _get_autocast_context(device, cfg.train.bf16):
                # Forward pass through the model
                _out = model(inputs, return_type=["cache"])
                block0_out = _out["cache"]["block_0.mlp"]

                # Forward pass through the probe
                probe_out = probe(block0_out, outputs,
                                  return_type=["logits", "loss"])
                logits = probe_out["logits"].to(device)
                loss = probe_out["loss"].mean().item()

                # Get vector of prediction successes
                pred = logits > 0
                acc = _calculate_binary_accuracy(pred, outputs)

            results["loss"].append(loss)
            results["accuracy"].append(acc)

    return results, debug_results

def validation(cfg, model, device, probe=None):
    if probe is not None:
        return probe_validation(cfg, model, device, probe)
    elif cfg.model_type == "recognizer":
        return binary_validation(cfg, model, device)
    elif cfg.model_type == "generator":
        return grammaticality_validation(cfg, model, device)
    else:
        raise ValueError(f"Unsupported model type: {cfg.model_type}")
