import hydra
from omegaconf import OmegaConf
import torch
from torch import nn
import matplotlib.pyplot as plt

from data import get_dataloader
from model import TransformerConfig, RecognizerModel
from model import ProbeConfig, Probe
from evals import grammar_evals

from utils import set_seed, open_log, cleanup
from utils import save_model, move_to_device

FP = None


@hydra.main(config_path=".", config_name="config.yaml", version_base="1.3")
def main(cfg):
    set_seed(cfg.seed)
    FP = open_log(cfg)

    dataloader = get_dataloader(
        cfg.lang, cfg.shenanigans,
        cfg.work_dir, cfg.seed,
        kind="PS",
    )

    model_config = TransformerConfig(
        **OmegaConf.to_object(cfg.model),
        dtype=torch.bfloat16 if cfg.train.bf16 else torch.float32,
        d_vocab=dataloader.dataset.PEG.vocab_size,
        seed=cfg.seed,
    )
    model = RecognizerModel(model_config)
    model_path = f"{cfg.work_dir}/models/{cfg.lang}/ckpt_{cfg.model.checkpoint}.pt"
    model_state = torch.load(model_path, weights_only=False)
    model.load_state_dict(model_state["net"])
    model.eval()

    probe_config = ProbeConfig(
        **OmegaConf.to_object(cfg.probe),
        dtype=torch.bfloat16 if cfg.train.bf16 else torch.float32,
        seed=cfg.seed,
    )
    probe = Probe(probe_config)
    probe_path = f"{cfg.work_dir}/probes/{cfg.lang}/ps_{cfg.probe.checkpoint}.pt"
    probe_state = torch.load(probe_path, weights_only=False)
    probe.load_state_dict(probe_state["net"])
    probe.train()

    for _in in dataloader:
        with torch.no_grad():
            seqs = _in["input_ids"]
            classes = _in["labels"]
            masks = _in["masks"]

            output = model(seqs, mask=masks, return_type=["logits", "cache"])
            cache = output["cache"]
            for k, v in cache.items():
                cache[k] = v.squeeze()

            string = dataloader.dataset.PEG.detokenize_string(seqs[0])

            probe_output = probe(cache["block_0.mlp"], return_type=["logits"])
            probe_pred = probe_output["logits"] > 0

            plt.matshow(cache["block_0.mlp"].cpu().numpy(), cmap='viridis')
            plt.colorbar()
            plt.title(f"Output of Layer 1 for {string}")
            plt.show()

            print()
            print(f"Classification output for {string}:")
            print(f"True class: {classes.squeeze()}, {classes.squeeze().shape}")
            print(f"Predicted class: {probe_pred.to(dtype=torch.long)}")

    FP = cleanup(cfg, FP)


if __name__ == "__main__":
    main()
