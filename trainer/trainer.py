import torch
from torch import nn
from tqdm import tqdm

from utils.optimizer import configure_optimizers
from utils.saving import save_model, save_probe
from utils.logging import log_train, log_eval


def it_compare(it, interval):
    return (it % interval == 0 and it > 0) if interval > 0 else False


class Trainer:

    def __init__(self, model, dataloader, evaluator, cfg, device, probe=None):
        self.model = model
        self.dataloader = dataloader
        self.evaluator = evaluator
        self.cfg = cfg
        self.device = device
        self.probe = probe
        self.optimizer, self.scheduler = configure_optimizers(model, cfg.optim)

    def train(self):
        self.model.train()
        train_loss = {"loss": []}
        it = 0

        for e in range(self.cfg.experiment.epochs):
            for _in in tqdm(self.dataloader, desc=f"Epoch {e+1}"):
                it += 1
                inputs, outputs = (_in["inputs"].to(self.device),
                                   _in["outputs"].to(self.device))

                self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=self.device.split(':')[0],
                                        dtype=(torch.bfloat16
                                               if self.cfg.train.bf16
                                               else torch.float32)):
                    output = self.model(inputs, outputs, return_type=["loss"])
                    loss = output["loss"].mean()

                loss.backward()
                if self.cfg.optim.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(),
                                             self.cfg.optim.grad_clip)
                self.optimizer.step()
                self.scheduler.step()

                train_loss["loss"].append(loss.item())

                if it_compare(it, self.cfg.log.train_interval):
                    lr = self.optimizer.param_groups[0]["lr"]
                    log_train(self.cfg.deploy, lr, train_loss)
                if it_compare(it, self.cfg.log.eval_interval):
                    self.model.eval()
                    val_results, _ = self.evaluator.validate_model(self.model)
                    log_eval(self.cfg.deploy, val_results)
                    self.model.train()
                if it_compare(it, self.cfg.log.save_interval):
                    save_model(self.cfg, self.model, self.optimizer, it)

    def train_probe(self):
        self.model.eval()
        self.probe.train()

        optimizer, scheduler = configure_optimizers(self.probe, self.cfg.optim)
        train_loss = {"loss": []}
        it = 0

        for e in range(self.cfg.experiment.epochs):
            for _in in tqdm(self.dataloader, desc=f"Epoch {e+1}"):
                it += 1
                seqs = _in["inputs"].to(self.device)
                classes = _in["outputs"].to(self.device)

                with torch.no_grad():
                    output = self.model(seqs, return_type=["cache"])
                    block0_out = output["cache"]["block_0.mlp"]

                optimizer.zero_grad()
                output = self.probe(block0_out, classes,
                                    return_type=["logits", "loss"])
                loss = output["loss"]

                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss["loss"].append(loss.item())

                if it_compare(it, self.cfg.log.train_interval):
                    lr = optimizer.param_groups[0]["lr"]
                    log_train(self.cfg.deploy, lr, train_loss)
                if it_compare(it, self.cfg.log.eval_interval):
                    self.probe.eval()
                    val_results, _ = self.evaluator.validate_probe(self.model,
                                                                   self.probe)
                    log_eval(self.cfg.deploy, val_results)
                    self.probe.train()
                if it_compare(it, self.cfg.log.save_interval):
                    save_probe(self.cfg, self.probe, optimizer, it, task="ps")
