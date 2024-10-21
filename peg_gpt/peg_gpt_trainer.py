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
        "n_layers": 2,
        "d_head": 32,
        "d_model": 128,
        "d_vocab": len(dataset.alpha),
        "n_ctx": 64,
        "n_heads": 4,
        "d_mlp": 512,
        "act_fn": "relu",
        "device": DEVICE,
    }
    training_config = {
        "num_epochs": 5,
        "batch_size": 32,
        "lr": 0.0005,
        "optimizer_name": "Adam",
        "print_every": 10000,
        "save_dir": f"{work_dir}/models/",
        "device": DEVICE,
    }

    model = HookedTransformer(HookedTransformerConfig(**model_config))
    config = train.HookedTransformerTrainConfig(**training_config)

    print("Training")
    torch.manual_seed(config.seed)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model.to(config.device)

    losses = []
    for epoch in range(config.num_epochs):
        epoch_loss = 0
        samples = 0
        
        for step, tokens in tqdm(enumerate(dataloader)):
            loss = model(tokens, return_type="loss")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            samples += tokens.shape[0]

            if config.print_every is not None and step % config.print_every == 0 and step > 0:
                print()
                print(f"Epoch {epoch+1} Samples {samples} Step {step} Loss {loss.item()}")
            if config.max_steps is not None and step >= config.max_steps:
                break
            
        epoch_loss /= len(dataloader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1} Loss {epoch_loss}")
        torch.save(model.state_dict(), f"{config.save_dir}/{lang}_model_{epoch}.pt")

    return model
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", help="Directory containing models and data")
    parser.add_argument("--lang", help="Language to train")
    args = parser.parse_args()
    
    peggpt_train(args.lang, args.work_dir)
