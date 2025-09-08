import hydra
from omegaconf import OmegaConf
import torch

from data.dataloader import get_dataloader
from model import TransformerConfig, create_model, ProbeConfig, Probe
from trainer.trainer import Trainer
from evals.evaluator import Evaluator

from utils import init_wandb, set_seed, open_log, cleanup


@hydra.main(config_path=".", config_name="config.yaml", version_base="1.3")
def main(cfg):
    init_wandb(cfg)
    set_seed(cfg.seed)
    fp = open_log(cfg)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    if cfg.experiment.task == "train_model":
        train_model(cfg, device)
    elif cfg.experiment.task == "train_probe":
        train_probe(cfg, device)
    elif cfg.experiment.task == "run_intervention":
        run_intervention(cfg, device)
    else:
        raise ValueError(f"Unknown task: {cfg.experiment.task}")

    cleanup(cfg, fp)


def train_model(cfg, device):
    dataloader = get_dataloader(cfg.lang,
                                cfg.model.type, cfg.data,
                                cfg.work_dir, cfg.seed,
                                kind="PEG")
    model_config = TransformerConfig(
        **OmegaConf.to_object(cfg.model),
        d_vocab=dataloader.dataset.PEG.vocab_size,
        pad_index=dataloader.dataset.pad_token_id,
        seed=cfg.seed,
    )
    model = create_model(cfg.model.type, model_config)
    evaluator = Evaluator(cfg, device)
    trainer = Trainer(model, dataloader, evaluator, cfg, device)
    trainer.train()


def train_probe(cfg, device):
    model_path = (f"{cfg.work_dir}/models/{cfg.model.type}/"
                  f"{cfg.lang}/ckpt_{cfg.model.checkpoint}.pt")
    model_state = torch.load(model_path, map_location=device)
    model_config = model_state["config"]
    model_config.act_cache = True
    model = create_model(cfg.model.type, model_config)
    model.load_state_dict(model_state["net"])
    model.eval()

    probe_config = ProbeConfig(
        d_m=model_config.d_m,
        **OmegaConf.to_object(cfg.probe)
    )
    probe = Probe(probe_config)

    dataloader = get_dataloader(cfg.lang,
                                cfg.probe.type, cfg.data,
                                cfg.work_dir, cfg.seed, kind="PS")
    evaluator = Evaluator(cfg, device, probe=probe)
    trainer = Trainer(probe, dataloader, evaluator, cfg, device, model=model)
    trainer.train_probe()


def run_intervention(cfg, device):
    # Placeholder for running interventions
    print("Running interventions...")
    pass


if __name__ == "__main__":
    main()
