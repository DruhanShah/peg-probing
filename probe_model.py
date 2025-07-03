import hydra
from omegaconf import OmegaConf
import torch
from tqdm import tqdm

from data import get_dataloader
from model import TransformerConfig, RecognizerModel
from model import ProbeConfig, Probe
from evals import intervention_evals

from utils import init_wandb, set_seed, open_log, cleanup
from utils import sanity_checks, configure_optimizers
from utils import save_probe, move_to_device
from utils import log_train, log_eval, log_debug

FP = None

def it_compare(it, interval):
    return (it % interval == 0 and it > 0) if interval > 0 else False


@hydra.main(config_path=".", config_name="config.yaml", version_base="1.3")
def main(cfg):
    init_wandb(cfg, ["probe"])
    set_seed(cfg.seed)
    FP = open_log(cfg)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    dataloader = get_dataloader(
        cfg.lang, cfg.parse,
        cfg.work_dir, cfg.seed,
        kind="PS",
    )

    # Load the main model
    model_path = f"{cfg.work_dir}/models/{cfg.lang}/ckpt_{cfg.model.checkpoint}.pt"
    model_state = torch.load(model_path, weights_only=False)
    model_config = TransformerConfig(
        **OmegaConf.to_object(model_state["config"]),
        dtype=torch.bfloat16 if cfg.train.bf16 else torch.float32,
        d_vocab=dataloader.dataset.PEG.vocab_size,
        seed=cfg.seed,
    )
    model = RecognizerModel(model_config)
    model_path = f"{cfg.work_dir}/models/{cfg.lang}/ckpt_{cfg.model.checkpoint}.pt"
    model_state = torch.load(model_path, weights_only=False)
    model.load_state_dict(model_state["net"])

    # Load the probe model
    probe_config = ProbeConfig(
        **OmegaConf.to_object(cfg.probe),
        dtype=torch.bfloat16 if cfg.train.bf16 else torch.float32,
        seed=cfg.seed,
    )
    probe = Probe(probe_config)

    train_probe(cfg, model, probe, dataloader)
    FP = cleanup(cfg, FP)


def train_probe(cfg, model, probe, dataloader):
    model.eval()
    probe.train()

    optimizer, scheduler = configure_optimizers(probe, cfg.optim)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if cfg.train.bf16 else torch.float32
    train_loss = {"loss": []}
    it = 0

    for e in range(cfg.train.epochs):
        for _in in tqdm(dataloader, desc=f"Epoch {e+1}"):
            it += 1
            seqs = _in["input_ids"].to(device, dtype=dt)
            classes = _in["labels"].to(device, dtype=dt)
            masks = _in["masks"].to(device, dtype=dt)

            output = model(seqs, mask=masks, return_type=["cache"])

            block0_out = output["cache"]["block_0.mlp"]

            optimizer.zero_grad()
            output = probe(block0_out, classes, return_type=["loss"])
            loss = output["loss"]

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss["loss"].append(loss.item())

            if it_compare(it, cfg.log.train_interval):
                lr = optimizer.param_groups[0]["lr"]
                train_loss = log_train(it, cfg.deploy, lr, train_loss)
            if it_compare(it, cfg.log.eval_interval):
                probe.eval()
                val_results = intervention_evals()
                val_results = log_eval(it, cfg.deploy, val_results)
                probe.train()
            if it_compare(it, cfg.log.save_interval):
                save_probe(cfg, probe, optimizer, it, task="ps")


if __name__ == "__main__":
    main()
