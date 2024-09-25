import argparse, sys
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    train,
    utils
)
from data import LangDataset
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_data(lang, work_dir):
    with open(f"{work_dir}/data/{lang}.txt", "r") as file:
        data = file.readlines()
    for i, line in enumerate(data):
        if line[-1] == "\n":
            data[i] = line[:-1]
    dataset = LangDataset(lang, data)
    return dataset


def peggpt_train(lang, work_dir):
    print("Loading dataset")
    dataset = get_data(lang, work_dir)

    model_config = {
        "n_layers": 4,
        "d_head": 32,
        "d_model": 128,
        "d_vocab": len(dataset.alpha),
        "n_ctx": 64,
        "n_heads": 4,
        "d_mlp": 512,
        "act_fn": "gelu",
    }
    training_config = {
        "num_epochs": 1,
        "batch_size": 16,
        "lr": 0.0001,
        "optimizer_name": "AdamW",
        "save_dir": "{save_dir}/models/",
        "print_every": 10000,
        "device": DEVICE,
    }

    model = HookedTransformer(HookedTransformerConfig(**model_config))
    config = train.HookedTransformerTrainConfig(**training_config)

    print("Training")
    torch.manual_seed(config.seed)
    model.train()

    if config.device is None:
        config.device = utils.get_device()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
    )

    scheduler = None
    if config.warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, step / config.warmup_steps),
        )

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model.to(config.device)

    for epoch in tqdm(range(1, config.num_epochs + 1)):
        samples = 0
        for step, tokens in tqdm(enumerate(dataloader)):
            loss = model(tokens, return_type="loss")
            loss.backward()
            if config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            if config.warmup_steps > 0:
                assert scheduler is not None
                scheduler.step()
            optimizer.zero_grad()

            samples += tokens.shape[0]

            if config.wandb:
                wandb.log({"train_loss": loss.item(), "samples": samples, "epoch": epoch})

            if config.print_every is not None and step % config.print_every == 0 and step > 0:
                print(f"Epoch {epoch} Samples {samples} Step {step} Loss {loss.item()}")

            if (
                config.save_every is not None
                and step % config.save_every == 0
                and config.save_dir is not None
            ):
                torch.save(model.state_dict(), f"{config.save_dir}/{lang}_model_{step}.pt")

            if config.max_steps is not None and step >= config.max_steps:
                break

    return model
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", help="Directory containing models and data")
    parser.add_argument("--lang", help="Language to train")
    args = parser.parse_args()
    
    peggpt_train(args.lang, args.work_dir)
