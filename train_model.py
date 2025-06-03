import hydra
from omegaconf import OmegaConf
import torch
from torch import nn
from tqdm import tqdm

from data import get_dataloader
from model import RecognizerModel
from transformer_lens import HookedTransformerConfig
from evals import grammar_evals

from utils import init_wandb, set_seed, open_log, cleanup
from utils import sanity_checks, configure_optimizers
from utils import save_model, move_to_device, log_train, log_eval


def it_compare(it, interval):
    return (it % interval == 0 and it > 0) if interval > 0 else False


@hydra.main(config_path=".", config_name="config.yaml", version_base="1.3")
def main(cfg):
    init_wandb(cfg)
    set_seed(cfg.seed)
    fp = open_log(cfg)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    dataloader = get_dataloader(cfg.data, cfg.work_dir, cfg.seed)
    sanity_checks(cfg, dataloader.dataset.max_len)

    model_config = HookedTransformerConfig(
        **OmegaConf.to_object(cfg.model),
        dtype=torch.bfloat16 if cfg.train.bf16 else torch.float32,
        d_vocab=dataloader.dataset.PEG.vocab_size,
    )
    model = RecognizerModel(model_config, device)
    model.to(device)

    params = pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"No. of parameters: {params/1e6:.2f}M")

    optimizer, scheduler = configure_optimizers(model, cfg.optim)

    train_model(cfg, model, dataloader, optimizer, scheduler, device)
    cleanup(cfg, fp)


def train_model(cfg, model, dataloader, optimizer, scheduler, device):
    """
    Function to train the base model
    """
    model.train()

    dt = torch.bfloat16 if cfg.train.bf16 else torch.float32
    total_steps = len(dataloader) * cfg.train.epochs
    train_loss = {"loss": []}
    it = 0

    print(f"Total training steps: {total_steps}")
    print(f"Learning rate warmup steps: {cfg.optim.warmup_steps}")

    for e in range(cfg.train.epochs):
        for _in in tqdm(dataloader, desc=f"Epoch {e+1}"):
            seqs, classes = _in["input_ids"], _in["labels"]
            it += 1

            optimizer.zero_grad(set_to_none=True)
            device_type = "cuda" if "cuda" in device else "cpu"

            with torch.amp.autocast(device_type=device_type, dtype=dt):
                loss = model(seqs, classes, return_type="loss")
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
                val_results = grammar_evals(cfg, model, device_type)
                val_results = log_eval(it, cfg.deploy, val_results)
                model.train()
            if it_compare(it, cfg.log.save_interval):
                save_model(cfg, model, optimizer, it)

        save_model(cfg, model, optimizer, it)



if __name__ == "__main__":
    main()
