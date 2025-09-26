import hydra
from omegaconf import OmegaConf
import torch

from data.dataloader import get_dataloader
from model import (
    ModelConfig, GeneratorModel,
    ProbeConfig, Probe
)
from trainer.trainer import Trainer
from evals.validator import Validator

from utils import init_wandb, set_seed, open_log, cleanup


@hydra.main(config_path=".", config_name="config.yaml", version_base="1.3")
def main(cfg):
    init_wandb(cfg)
    set_seed(cfg.seed)
    fp = open_log(cfg)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    if cfg.task == "train_model":
        train_model(cfg, device)
    elif cfg.task == "train_probe":
        train_probe(cfg, device)
    elif cfg.task == "run_intervention":
        run_intervention(cfg, device)
    else:
        raise ValueError(f"Unknown task: {cfg.task}")

    cleanup(cfg, fp)


def train_model(cfg, device):
    dataloader = get_dataloader(cfg.lang,
                                "generator", cfg.data,
                                cfg.work_dir, cfg.seed,
                                kind="model")
    model_config = ModelConfig(
        **OmegaConf.to_object(cfg.model),
        d_vocab=dataloader.dataset.PEG.vocab_size,
        pad_index=dataloader.dataset.pad_token_id,
        seed=cfg.seed,
    )
    model = GeneratorModel(model_config)
    evaluator = Validator(cfg, device)
    trainer = Trainer(model, dataloader, evaluator, cfg, device)
    trainer.train()


def train_probe(cfg, device):
    model_path = (f"{cfg.work_dir}/models/generator/"
                  f"{cfg.lang}/ckpt_{cfg.model.checkpoint}.pt")
    model_state = torch.load(model_path, map_location=device)
    model_config = model_state["config"]
    model_config.act_cache = True
    model = GeneratorModel(model_config)
    model.load_state_dict(model_state["net"])
    model.eval()

    probe_config = ProbeConfig(
        d_m=model_config.d_m,
        **OmegaConf.to_object(cfg.probe)
    )
    probe = Probe(probe_config)

    dataloader = get_dataloader(cfg.lang,
                                cfg.probe.type, cfg.data,
                                cfg.work_dir, cfg.seed, kind="probe")
    evaluator = Validator(cfg, device, probe=probe)
    trainer = Trainer(probe, dataloader, evaluator, cfg, device, model=model)
    trainer.train_probe()


def run_intervention(cfg, device):
    # Placeholder for running interventions
    print("Running interventions...")
    pass


if __name__ == "__main__":
    main()
