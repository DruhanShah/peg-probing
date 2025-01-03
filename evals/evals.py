import torch
import numpy as np

def grammar_evals(cfg, model, template, grammar, device):
    """
    Evaluate the model on grammaticality.

    Args:
        cfg (dict): Configuration dictionary.
        model (torch.nn.Module): Model to evaluate.
        template (torch.Tensor): Template to generate samples from.
        grammar (Grammar): Grammar object.
        device (torch.device): Device to run on.

    Returns:
        results_dict (dict): Results of the grammaticality evaluation.
    """
    model.eval()
    eval_bsize = 1024

    with torch.no_grad():

        # Generate samples
        inputs = template.repeat(eval_bsize, 1).to(device)
        samples, per_token_logprobs = model.sample(
            inputs=inputs, 
            max_new_tokens=cfg.data.max_len - 10, 
            retrieve_llhoods="tokens",
            )

        # Transfer to CPU and detokenize
        samples = samples.cpu()
        samples = [grammar.detokenize_string(s).split("<eos>")[0][5:] for s in samples]

        # Eval grammatical correctness
        results_grammaticality = {
            "validity": {"num": 0, "satisfied": 0},
            "prefix": [0 for i in range(cfg.data.max_len)],
            "failures": 0,
        }

        failures = []
        for sid, s in enumerate(samples):
            results_grammaticality["validity"]["num"] += 1
            grammaticality, pref_len = grammar.check_grammaticality(s)

            if grammaticality:
                results_grammaticality["validity"]["satisfied"] += 1
                results_grammaticality["prefix"][pref_len-1] += 1
            else:
                failures.append(s)
                results_grammaticality["failures"] += 1

        return results_grammaticality, failures
