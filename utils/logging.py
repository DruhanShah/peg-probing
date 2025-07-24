import torch
import wandb
import numpy as np
import random
import os
import warnings
import pickle as pkl
import matplotlib.pyplot as plt
from omegaconf import OmegaConf


def log_debug(it, fp, debug_info):
        """
        Log debug information
        """

        print(f"Debug info at iteration {it}:\n{debug_info}", file=fp)
        print("", file=fp)

        debug_info = {}
        return debug_info


def log_gen(deploy, stats):
    """
    Log generated data information
    """

    fig, ax = plt.subplots()

    if isinstance(stats, dict):
        ax.plot(range(len(stats["pos"])), stats["pos"], label="Positive samples")
        ax.plot(range(len(stats["neg"])), stats["neg"], label="Negative samples")
    else:
        ax.plot(range(len(stats)), stats, label="Sample lengths")
    if deploy:
        wandb.log({"data": {"lengths": fig}})
    else:
        plt.show()

    stats = {}
    return stats


def log_train(deploy, lr, train_loss):
    """
    Log training loss information
    """

    logs = {"train": {k: np.mean(v) for k, v in train_loss.items()}, "lr": lr}
    if deploy:
        wandb.log(logs)
    else:
        print(f"train_loss: {logs}")

    train_loss = {"loss": []}
    return train_loss


def log_eval(deploy, eval_results):
    """
    Log eval information
    """
    
    logs = {"eval": {k: np.mean(v) for k, v in eval_results.items()}}
    if deploy:
        wandb.log(logs)
    else:
        print(f"eval_results: {logs}")

    eval_results = {"accuracy": [], "loss": []}
    return eval_results
