import os
import warnings
import torch
import wandb
import numpy as np
import random
from omegaconf import OmegaConf
from matplotlib import pyplot as plt


def open_log(cfg):
    """
    Open log file but don't redirect stdout and stderr to it
    """

    os.makedirs(cfg.work_dir + "/logs/", exist_ok=True)
    return open(f"{cfg.work_dir}/logs/{cfg.lang}.log", "w")


def cleanup(cfg, fp):
    """
    Close log file and wandb
    """

    if fp is not None:
        fp.close()
    if cfg.deploy:
        wandb.finish()
    return None


def sanity_checks(cfg, max_len):
    """
    Basic sanity checks for model configuration and data compatibility
    """

    assert cfg.model.n_ctx >= max_len
    assert cfg.model.d_m % cfg.model.d_h == 0

    if not torch.cuda.is_available():
        warnings.warn("WARNING: running on CPU", UserWarning)
    elif not torch.cuda.is_bf16_supported():
        warnings.warn("WARNING: running without BF16", UserWarning)


def set_seed(seed=0):
    """
    Don't set true seed to be nearby values. Doesn't give best randomness
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_wandb(cfg, tags=None):
    """
    Initialize wandb
    """

    if cfg.deploy:
        wandb.init(
            project=cfg.project_name,
            tags=tags,
        )
        wandb.run.name = wandb.run.id
        wandb.run.save()
        wandb.config.update(OmegaConf.to_container(cfg))


def visualise(tensor, title="Heatmap", string=""):
    """
    Visualise an attention map's different heads
    """
    H, N = tensor.shape[0], tensor.shape[1]
    string = "<" + string + ">"

    fig, axes = plt.subplots(1, H, figsize=(H * 4, 4))
    fig.suptitle(title + " with input " + string, fontsize=16)
    if H == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        if string:
            ticks = np.arange(N)
            ax.set_xticks(ticks)
            ax.set_xticklabels(list(string))
            ax.set_yticks(ticks)
            ax.set_yticklabels(list(string))
        else:
            ax.axis('off')
        ax.imshow(tensor[i], cmap='hot')
        ax.set_title(f"Submatrix {i}")

    plt.show()
