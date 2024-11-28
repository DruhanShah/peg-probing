import hydra
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from model import GPT
from data import get_dataloader
from evals import grammar_evals

from utils import init_wandb, set_seed, save_config, open_log, cleanup
from utils import sanity_checks, configure_optimizers, update_cosine_warmup_lr
from utils import save_model, move_to_device, log_train, log_eval


@hydra.main(config_path=".", config_name="config.yaml", version_base="1.3")
def main(cfg):
    init_wandb(cfg)
    set_seed(cfg.seed)
    save_config(cfg)
    fp = open_log(cfg)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    dataloader = get_dataloader(
        language=cfg.data.language,
        config=cfg.data.config,
        num_iters=cfg.data.num_iters,
        max_sample_length=cfg.data.max_sample_length,
        seed=cfg.seed,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    sanity_checks(cfg, dataloader.dataset.max_sample_length)

    model = GPT(cfg.model, dataloader.dataset.PEG.vocab_size)
    model.to(device)
    if cfg.model.compile:
        model = torch.compile(model)
    print(f"No. of parameters: {model.get_num_params()/1e6:.2f}M")

    optimizer = configure_optimizers(model, cfg.optimizer)

    train(cfg, model, dataloader, optimizer, device)
    cleanup(cfg, fp)


def train(cfg, model, dataloader, optimizer, device):
    """
    Function to train the base model
    """
    model.train()

    save_grammar = True
    dt = torch.bfloat16 if cfg.bf16 else torch.float32

    total_steps = len(dataloader) * cfg.epochs

    train_loss = {"total": []}
    for k in dataloader.dataset.tasks_dict.keys():
        train_loss[k] = []
    lr, it, save_tables = 0, 0, 0
    print(f"Total training steps: {total_steps}")
    print(f"Learning rate warmup steps: {cfg.optim.warmup_steps}")

    results_dir = save_model(cfg, model, optimizer, it)

    if cfg.model.use_pretrained:
        model.load_state_dict(torch.load(cfg.model.pretrain_dir)["net"])
        optimizer.load_state_dict(torch.load(cfg.model.pretrain_dir)["optim"])

    # Training loop
    for e in range(cfg.epochs):
        for sequences, seq_lengths in tqdm(dataloader, desc=f"Epoch {e+1}"):
            if it > 7e4:
                save_model(cfg, model, optimizer, it)
                break

            B = sequences.size(0)
            inputs, labels = move_to_device([sequences[:, :-1], sequences[:, 1:]], device)

            samples_per_task = {
                k: inputs[:, 1] == dataloader.dataset.task_token_idx[k]
                for k in dataloader.dataset.tasks_token_idx
            }

            train_lengths = {
                "max": seq_lengths.max().item(),
                "min": seq_lengths.min().item(),
                "mean": seq_lengths.mean().item(),
            }

            if it % cfg.log.log_interval == 0:
                train_loss = log_train(it, cfg.deploy, lr, train_loss, train_lengths)

                
            if it % cfg.log.eval_interval == 0:
                model.eval()
                grammar_results_dict = grammar_evals(
                    cfg=cfg, model=model, templates=dataloader.dataset.templates,
                    grammar=dataloader.dataset.PEG, device=device,
                ) if cfg.eval.grammar else (None, None)
                save_tables = log_eval(
                    deploy=cfg.deploy, it=it, save_tables=save_tables,
                    grammaticality_results=grammar_results_dict,
                )
                model.train()

            it, lr = update_cosine_warmup_lr(it, cfg.optimizer, optimizer, total_steps)

            optimizer.zero_grad(set_to_none=True)
            device_type = "cuda" if "cuda" in device else "cpu"
            with torch.amp.autocast(device_type=device_type, dtype=dt):
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=dataloader.dataset.pad_token_idx,
                    reduction="none",
                ).reshape(B, -1).mean(dim=1)

                for k in dataloader.dataset.task_tokens:
                    train_loss[k].append(loss[samples_per_task[k]].mean().item())

                loss = loss.mean()
                train_loss["total"].append(loss.item())

            loss.backward()
            if cfg.optimizer.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)
            optimizer.step()

            if it % cfg.log.save_interval == 0:
                save_model(cfg, model, optimizer, it)
            if save_grammar:
                dataloader.dataset.save_grammar(results_dir)
                save_gramar = False

        save_model(cfg, model, optimizer, it)



if __name__ == "__main__":
    main()
