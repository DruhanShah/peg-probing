import hydra
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from transformer_lens import HookedTransformer, HookedTransformerConfig

from data import get_dataloader
from evals import grammar_evals

from utils import init_wandb, set_seed, open_log, cleanup
from utils import sanity_checks, configure_optimizers, update_cosine_warmup_lr
from utils import save_model, move_to_device, log_train, log_eval
from utils import obj_to_dict


def it_compare(it, interval):
    return (it % interval == 0 and it > 0) if interval > 0 else False


@hydra.main(config_path=".", config_name="config.yaml", version_base="1.3")
def main(cfg):
    init_wandb(cfg)
    set_seed(cfg.seed)
    fp = open_log(cfg)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    num_devices = torch.cuda.device_count()

    data_dir = cfg.work_dir + "/" + cfg.data.save_dir
    dataloader = get_dataloader(
        **obj_to_dict(cfg.data),
        results_dir=data_dir,
        seed=cfg.seed,
    )
    sanity_checks(cfg, dataloader.dataset.max_len)

    model_config = HookedTransformerConfig(
        **obj_to_dict(cfg.model),
        dtype=torch.bfloat16 if cfg.model.bf16 else torch.float32,
        d_vocab=dataloader.dataset.PEG.vocab_size,
        n_devices=num_devices,
    )
    model = HookedTransformer(model_config)
    model.to(device)
    print(f"No. of parameters: {model.get_num_params()/1e6:.2f}M")

    optimizer = configure_optimizers(model, cfg.optim)

    train_model(cfg, model, dataloader, optimizer, device)
    cleanup(cfg, fp)


def train_model(cfg, model, dataloader, optimizer, device):
    """
    Function to train the base model
    """
    model.train()

    dt = torch.bfloat16 if cfg.bf16 else torch.float32
    total_steps = len(dataloader) * cfg.epochs
    train_loss = {"loss": []}
    lr, it = 0.0, 0

    print(f"Total training steps: {total_steps}")
    print(f"Learning rate warmup steps: {cfg.optim.warmup_steps}")

    for e in range(cfg.epochs):
        for seqs, seq_lengths in tqdm(dataloader, desc=f"Epoch {e+1}"):
            B = seqs.size(0)
            inputs, labels = move_to_device([seqs[:, :-1], seqs[:, 1:]], device)

            it, lr = update_cosine_warmup_lr(it, cfg.optim, optimizer, total_steps)
            optimizer.zero_grad(set_to_none=True)
            device_type = "cuda" if "cuda" in device else "cpu"

            with torch.amp.autocast(device_type=device_type, dtype=dt):
                logits = model(inputs, return_type="loss")
                train_loss["loss"].append(loss.item())

            loss.backward()
            if cfg.optim.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            optimizer.step()

            if it_compare(it, cfg.log.train_interval):
                train_loss = log_train(it, cfg.deploy, lr, train_loss)
            if it_compare(it, cfg.log.eval_interval):
                # Evals to be done here
            if it_compare(it, cfg.log.save_interval):
                save_model(cfg, model, optimizer, it)

        save_model(cfg, model, optimizer, it)



if __name__ == "__main__":
    main()
