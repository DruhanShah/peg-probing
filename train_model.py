import hydra
from omegaconf import OmegaConf
import torch
from torch import nn
from tqdm import tqdm

from data import get_dataloader
from model import TransformerConfig, create_model
from evals import validation

from utils import init_wandb, set_seed, open_log, cleanup
from utils import sanity_checks, configure_optimizers
from utils import save_model, move_to_device
from utils import log_train, log_eval, log_debug

FP = None


def it_compare(it, interval):
    return (it % interval == 0 and it > 0) if interval > 0 else False


@hydra.main(config_path=".", config_name="config.yaml", version_base="1.3")
def main(cfg):
    global FP

    init_wandb(cfg, ["train", "eval"])
    set_seed(cfg.seed)
    FP = open_log(cfg)

    dataloader = get_dataloader(
        cfg.lang, cfg.model_type,
        cfg.data,
        cfg.work_dir, cfg.seed,
        kind="PEG",
    )

    model_config = TransformerConfig(
        **OmegaConf.to_object(cfg.model),
        dtype=torch.bfloat16 if cfg.train.bf16 else torch.float32,
        d_vocab=dataloader.dataset.PEG.vocab_size,
        pad_index=dataloader.dataset.pad_token_id,
        seed=cfg.seed,
    )
    model = create_model(cfg.model_type, model_config)

    params = pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"No. of parameters: {params/1e6:.2f}M")

    train_model(cfg, model, dataloader)
    FP = cleanup(cfg, FP)


def train_model(cfg, model, dataloader):
    global FP
    model.train()

    optimizer, scheduler = configure_optimizers(model, cfg.optim)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if cfg.train.bf16 else torch.float32
    train_loss = {"loss": []}
    it = 0

    print(f"Total training steps: {cfg.train.epochs * len(dataloader)}")
    print(f"Learning rate warmup steps: {cfg.optim.warmup_steps}")

    for e in range(cfg.train.epochs):
        for _in in tqdm(dataloader, desc=f"Epoch {e+1}"):
            inputs, outputs = _in["inputs"], _in["outputs"]
            it += 1

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device, dtype=dt):
                output = model(
                    inputs, outputs,
                    return_type=["loss"]
                )
                loss = output["loss"].mean()
                train_loss["loss"].append(loss.item())

            loss.backward()
            if cfg.optim.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            optimizer.step()
            scheduler.step()

            if it_compare(it, cfg.log.train_interval):
                lr = optimizer.param_groups[0]["lr"]
                train_loss = log_train(it, cfg.deploy, lr, train_loss)
            if it_compare(it, cfg.log.eval_interval):
                model.eval()
                val_results, debug_info = validation(cfg, model, device)
                if cfg.log.debug:
                    log_debug(it, FP, debug_info)
                val_results = log_eval(it, cfg.deploy, val_results)
                model.train()
            if it_compare(it, cfg.log.save_interval):
                save_model(cfg, model, optimizer, it)


if __name__ == "__main__":
    main()
