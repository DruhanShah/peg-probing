import hydra
from omegaconf import OmegaConf
import torch
from torch import nn
import matplotlib.pyplot as plt

from model import TransformerConfig, create_model
from evals import validation

from utils import set_seed, open_log, cleanup
from utils import save_model, move_to_device

FP = None


@hydra.main(config_path=".", config_name="config.yaml", version_base="1.3")
def main(cfg):
    set_seed(cfg.seed)
    global FP
    FP = open_log(cfg)

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

    for _ in range(10):
        validation(cfg, model, dataloader)

    FP = cleanup(cfg, FP)


if __name__ == "__main__":
    main()
