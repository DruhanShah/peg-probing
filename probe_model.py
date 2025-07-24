import hydra
from omegaconf import OmegaConf
import torch
from tqdm import tqdm

from data import get_dataloader
from model import create_model
from model import ProbeConfig, Probe
from evals import validation

from utils import init_wandb, set_seed, open_log, cleanup
from utils import sanity_checks, configure_optimizers
from utils import save_probe, move_to_device
from utils import log_train, log_eval, log_debug

FP = None

def it_compare(it, interval):
    return (it % interval == 0 and it > 0) if interval > 0 else False


@hydra.main(config_path=".", config_name="config.yaml", version_base="1.3")
def main(cfg):
    init_wandb(cfg, ["probe", "parse"])
    set_seed(cfg.seed)
    FP = open_log(cfg)

    dataloader = get_dataloader(
        cfg.lang, cfg.model_type,
        cfg.parse,
        cfg.work_dir, cfg.seed,
        kind="PS", quiet=(not cfg.deploy),
    )

    # Load the main model
    model_path = f"{cfg.work_dir}/models/{cfg.model_type}/{cfg.lang}/ckpt_{cfg.model.checkpoint}.pt"
    model_state = torch.load(model_path, weights_only=False)
    model_state["config"].act_cache = cfg.model.act_cache
    model = create_model(cfg.model_type, model_state["config"])
    model.load_state_dict(model_state["net"])

    # Initialize the probe
    probe_config = ProbeConfig(
        d_m = model_state["config"].d_m,
        d_mlp = cfg.probe.d_mlp,
        linear = cfg.probe.linear,
        dtype=torch.bfloat16 if cfg.train.bf16 else torch.float32,
        seed=cfg.seed,
    )
    probe = Probe(probe_config)

    train_probe(cfg, model, probe, dataloader)
    FP = cleanup(cfg, FP)


def train_probe(cfg, model, probe, dataloader):
    model.eval()
    probe.train()
    global FP

    optimizer, scheduler = configure_optimizers(probe, cfg.optim)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if cfg.train.bf16 else torch.float32
    train_loss = {"loss": []}
    it = 0

    for e in range(cfg.train.epochs):
        for _in in tqdm(dataloader, desc=f"Epoch {e+1}"):
            it += 1
            seqs = _in["inputs"].to(device, dtype=dt)
            classes = _in["outputs"].to(device, dtype=dt)

            output = model(seqs, return_type=["cache"])
            block0_out = output["cache"]["block_0.mlp"]

            optimizer.zero_grad()
            output = probe(block0_out, classes, return_type=["logits", "loss"])
            loss = output["loss"]

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss["loss"].append(loss.item())

            if it_compare(it, cfg.log.train_interval):
                lr = optimizer.param_groups[0]["lr"]
                train_loss = log_train(cfg.deploy, lr, train_loss)
            if it_compare(it, cfg.log.eval_interval):
                probe.eval()
                val_results, debug_results = validation(cfg, model, device, probe=probe)
                val_results = log_eval(cfg.deploy, val_results)
                probe.train()
            if it_compare(it, cfg.log.save_interval):
                save_probe(cfg, probe, optimizer, it, task="ps")


if __name__ == "__main__":
    main()
