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
from utils import save_model, move_to_device
from utils import log_train, log_eval, log_debug


def it_compare(it, interval):
    return (it % interval == 0 and it > 0) if interval > 0 else False


@hydra.main(config_path=".", config_name="config.yaml", version_base="1.3")
def main(cfg):
    init_wandb(cfg, ["train", "eval"])
    set_seed(cfg.seed)
    fp = open_log(cfg)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    dataloader = get_dataloader(cfg.data, cfg.work_dir, cfg.seed)
    sanity_checks(cfg, dataloader.dataset.max_len)

    model_config = HookedTransformerConfig(
        **OmegaConf.to_object(cfg.model),
        dtype=torch.bfloat16 if cfg.train.bf16 else torch.float32,
        d_vocab=dataloader.dataset.PEG.vocab_size,
        seed=cfg.seed,
    )
    model = RecognizerModel(model_config, device)
    model.to(device)

    params = pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"No. of parameters: {params/1e6:.2f}M")

    train_model(cfg, model, dataloader, device, fp)
    cleanup(cfg, fp)


def train_model(cfg, model, dataloader, device, fp):
    """
    Function to train the base model
    """
    model.train()

    optimizer, scheduler = configure_optimizers(model, cfg.optim)

    dt = torch.bfloat16 if cfg.train.bf16 else torch.float32
    total_steps = len(dataloader) * cfg.train.epochs
    train_loss = {"loss": []}
    it = 0

    print(f"Total training steps: {total_steps}")
    print(f"Learning rate warmup steps: {cfg.optim.warmup_steps}")
    print("\n--- Initial Embedding Weight Statistics ---")
    if hasattr(model.transformer, 'W_E'):
        print(f"W_E (Token Embeddings) Stats:\n"
              f"  Mean: {model.transformer.W_E.mean().item():.4e}\n"
              f"  Std: {model.transformer.W_E.std().item():.4e}\n"
              f"  Min: {model.transformer.W_E.min().item():.4e}\n"
              f"  Max: {model.transformer.W_E.max().item():.4e}")
    if hasattr(model.transformer, 'W_pos'):
        print(f"W_pos (Positional Embeddings) Stats:\n"
              f"  Mean: {model.transformer.W_pos.mean().item():.4e}\n"
              f"  Std: {model.transformer.W_pos.std().item():.4e}\n"
              f"  Min: {model.transformer.W_pos.min().item():.4e}\n"
              f"  Max: {model.transformer.W_pos.max().item():.4e}")

    for e in range(cfg.train.epochs):
        for _in in tqdm(dataloader, desc=f"Epoch {e+1}"):
            seqs, masks, classes = _in["input_ids"], _in["masks"], _in["labels"]
            it += 1

            optimizer.zero_grad(set_to_none=True)
            device_type = "cuda" if "cuda" in device else "cpu"

            with torch.amp.autocast(device_type=device_type, dtype=dt):
                loss, cache = model(
                    seqs, classes, mask=masks,
                    return_type=["loss", "cache"]
                )
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


if __name__ == "__main__":
    main()
