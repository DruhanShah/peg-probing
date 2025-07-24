import hydra
from omegaconf import OmegaConf
import torch
from torch import nn
import matplotlib.pyplot as plt

from model import TransformerConfig, create_model
from evals import validation

from utils import set_seed, init_wandb, open_log, cleanup
from utils import log_eval, log_debug
from utils import save_model, move_to_device

FP = None


@hydra.main(config_path=".", config_name="config.yaml", version_base="1.3")
def main(cfg):
    set_seed(cfg.seed)
    init_wandb(["parse", "eval"])
    global FP
    FP = open_log(cfg)

    model_path = f"{cfg.work_dir}/models/{cfg.model_type}/{cfg.lang}/ckpt_{cfg.model.checkpoint}.pt"
    model_state = torch.load(model_path, weights_only=False)
    model = create_model(cfg.model_type, model_state["config"])
    model.load_state_dict(model_state["net"])
    model.eval()


    FP = cleanup(cfg, FP)


if __name__ == "__main__":
    main()
