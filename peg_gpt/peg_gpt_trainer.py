import argparse, sys
import json
from data import LangDataset
from transformer import Transformer, TransformerConfig
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_data(lang, work_dir):
    with open(f"{work_dir}/data/{lang}.txt", "r") as file:
        data = file.readlines()
    for i, line in enumerate(data):
        if line[-1] == "\n":
            data[i] = line[:-1]
    dataset = LangDataset(lang, data)
    return dataset


class Trainer:

    def __init__(self, model, config, loss_fn, dataset, lang):
        self.lang = lang
        self.model = model
        self.config = config
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        self.dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
        if self.config["wandb"]:
            wandb.init(project=self.config["wandb_project"])

    def train(self):
        torch.manual_seed(self.config["seed"])
        self.model.train()
        self.model.to(self.config["device"])

        for epoch in range(self.config["epochs"]):
            epoch_loss = 0
            for step, tokens in tqdm(enumerate(self.dataloader)):
                x, y = tokens[:, :-1], tokens[:, 1:]
                y_pred = self.model(x, y)
                loss = self.loss_fn(y_pred.view(-1, self.config["t_vocab"]), y.reshape(-1))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                epoch_loss += (loss.item() - epoch_loss)/(step+1)

                if step % self.config["pe"] == 0 and step > 0:
                    if self.config["wandb"]:
                        wandb.log({"loss": epoch_loss})
                    else:
                        print()
                        print(f"Epoch {epoch+1} Step {step} Loss {loss.item()}")

            print(f"Epoch {epoch+1} Loss {epoch_loss}")
            torch.save(
                self.model.state_dict(),
                f"{self.config['save_dir']}/{self.lang}_model_{epoch}.pt"
            )

        return model
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", help="Directory containing models and data")
    parser.add_argument("--lang", help="Language to train")
    args = parser.parse_args()


    with open("config.json", "r") as f:
        config = json.load(f)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["t_vocab"] = len(dataset.alpha)
    config["s_vocab"] = len(dataset.alpha)
    config["save_dir"] = f"{args.work_dir}/models"

    dataset = get_data(args.lang, args.work_dir)
    model = Transformer(TransformerConfig.from_dict(config))
    loss_fn = torch.nn.CrossEntropyLoss()

    trainer = Trainer(model, config, loss_fn, dataset, args.lang)
    model = trainer.train()
