import argparse, sys
import json
from data.generate import LangDataset, ALPHA
from model.transformer import Transformer, TransformerConfig
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb


def get_data(lang, work_dir, device):
    with open(f"{work_dir}/data/corpus/{lang}.txt", "r") as file:
        data = file.readlines()
    for i, line in enumerate(data):
        if line[-1] == "\n":
            data[i] = line[:-1]
    dataset = LangDataset(lang, data, device)
    return dataset


class Trainer:

    def __init__(self, model, config, loss_fn, dataset, lang):
        self.lang = lang
        self.model = model
        self.config = config
        self.dataset = dataset

        self.loss_fn = loss_fn
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config["batch_size"],
            shuffle=True
        )
        if self.config["wandb"]:
            wandb.init(project=self.config["wandb_project"])

    def val_loop(self):
        for _ in range(self.config["val_steps"]):
            prompt = [self.dataset.stoi["<bos>"]]
            prompt = torch.tensor(prompt, dtype=torch.long).unsqueeze(0)
            prompt = prompt.to(self.config["device"])

            output = self.model.sample(prompt, max_len=self.config["N_max"])
            st = "".join([self.dataset.alpha[i] for i in output[0].tolist()])
            print(st)

    def train_loop(self):
        torch.manual_seed(self.config["seed"])

        self.model.train()
        self.model.to(self.config["device"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])

        for epoch in range(self.config["epochs"]):
            epoch_loss = 0
            for step, tokens in tqdm(enumerate(self.dataloader)):
                x, y = tokens[:, :-1], tokens[:, 1:]
                y_pred = self.model(x, y)
                y_pred = y_pred.contiguous().view(-1, config["t_vocab"])
                y = y.contiguous().view(-1)
                loss = self.loss_fn(y_pred, y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                epoch_loss += (loss.item() - epoch_loss)/(step+1)

                if step % self.config["val_every"] == 0 and step > 0:
                    self.val_loop()

                if step % self.config["log_every"] == 0 and step > 0:
                    if self.config["wandb"]:
                        wandb.log({"loss": epoch_loss})
                    else:
                        print()
                        print(f"Epoch {epoch+1} Step {step} Loss {loss.item()}")

            print(f"Epoch {epoch+1} Loss {epoch_loss}")
            state_dict = self.model.state_dict()
            torch.save(
                state_dict,
                f"{self.config['save_dir']}/{self.lang}_model_{epoch+1}.pt"
            )

        return model
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", help="Directory containing models and data")
    parser.add_argument("--lang", help="Language to train")
    args = parser.parse_args()

    with open("config.json", "r") as f:
        config = json.load(f)
    config["t_vocab"] = len(ALPHA[args.lang]) + 2
    config["s_vocab"] = len(ALPHA[args.lang]) + 2
    config["save_dir"] = f"{args.work_dir}/model/models"


    dataset = get_data(args.lang, args.work_dir, config["device"])
    model = Transformer(TransformerConfig.from_dict(config))
    loss_fn = torch.nn.CrossEntropyLoss()

    trainer = Trainer(model, config, loss_fn, dataset, args.lang)
    model = trainer.train_loop()
