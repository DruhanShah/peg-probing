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
    # save_config(cfg)
    fp = open_log(cfg)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    data_dir = cfg.work_dir + "/" + cfg.data.save_dir
    dataloader = get_dataloader(
        language=cfg.data.language,
        config=cfg.data.config,
        precomp=cfg.data.precomp,
        results_dir=data_dir,
        num_iters=cfg.data.num_iters,
        max_len=cfg.data.max_len,
        seed=cfg.seed,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    sanity_checks(cfg, dataloader.dataset.max_len)

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

    train_loss = {"loss": []}
    lr, it, save_tables = 0.0, 0, 0
    print(f"Total training steps: {total_steps}")
    print(f"Learning rate warmup steps: {cfg.optimizer.warmup_steps}")

    results_dir = save_model(cfg, model, optimizer, it)

    if cfg.model.use_pretrained:
        pretrain_dir = cfg.work_dir + "/" + cfg.model.pretrain_dir
        model.load_state_dict(torch.load(pretrain_dir)["net"])
        optimizer.load_state_dict(torch.load(pretrain_dir)["optim"])

    # Training loop
    for e in range(cfg.epochs):
        for sequences, seq_lengths in tqdm(dataloader, desc=f"Epoch {e+1}"):
            if it > 7e4:
                save_model(cfg, model, optimizer, it)
                break

            B = sequences.size(0)
            inputs, labels = move_to_device([sequences[:, :-1], sequences[:, 1:]], device)

            train_lengths = {
                "max": seq_lengths.max().item(),
                "min": seq_lengths.min().item(),
                "mean": seq_lengths.mean().item(),
            }

            if it % cfg.log.log_interval == 0:
                train_loss = log_train(it, cfg.deploy, lr, train_loss, train_lengths)

                
            if it % cfg.log.eval_interval == 0 and it > 0:
                model.eval()
                (grammar_results_dict, failures), _ = grammar_evals(
                    cfg=cfg, model=model, template=dataloader.dataset.template,
                    grammar=dataloader.dataset.PEG, device=device,
                ) if cfg.eval.grammar else None, None
                save_tables = log_eval(
                    cfg=cfg, it=it, save_tables=save_tables,
                    grammaticality_results=grammar_results_dict,
                    failures=failures
                ) if cfg.eval.save_tables else save_tables
                model.train()

            it, lr = update_cosine_warmup_lr(it, cfg.optimizer, optimizer, total_steps)

            optimizer.zero_grad(set_to_none=True)
            device_type = "cuda" if "cuda" in device else "cpu"
            with torch.amp.autocast(device_type=device_type, dtype=dt):
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=dataloader.dataset.pad_token_id,
                    reduction="none",
                ).reshape(B, -1).mean(dim=1)

                loss = loss.mean()
                train_loss["loss"].append(loss.item())

            loss.backward()
            if cfg.optimizer.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)
            optimizer.step()

            if it % cfg.log.save_interval == 0 and it > 0:
                save_model(cfg, model, optimizer, it)
            if save_grammar:
                dataloader.dataset.save_grammar(results_dir)
                save_gramar = False

        save_model(cfg, model, optimizer, it)



if __name__ == "__main__":
    main()
