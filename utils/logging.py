import torch
import wandb
import numpy as np
import random
import os
import warnings
import matplotlib.pyplot as plt
from omegaconf import OmegaConf


def sanity_checks(cfg, max_len):
    """
    Basic sanity checks for model configuration and data compatibility
    """

    assert(cfg.model.context_size >= max_len)
    assert(cfg.model.n_embd % cfg.model.n_head == 0)

    if not torch.cuda.is_available():
        warnings.warn("WARNING: running on CPU", UserWarning)
    else:
        if not torch.cuda.is_bf16_supported():
            warnings.warn("WARNING: running without BF16", UserWarning)
        if not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            raise NotImplementedError("Flash Attention requires PyTorch >= 2.0")


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


def open_log(cfg):
    """
    Open log file but don't redirect stdout and stderr to it
    """

    os.makedirs(cfg.work_dir + '/logs/' + cfg.tag, exist_ok=True)


def init_wandb(cfg):
    """
    Initialize wandb
    """

    if cfg.deploy:
        wandb.init(project=cfg.project_name)
        wandb.run.name = wandb.run.id
        wandb.run.save()
        wandb.config.update(OmegaConf.to_container(cfg))


def cleanup(cfg, fp):
    """
    Close log file and wandb
    """

    if fp is not None:
        fp.close()
    if cfg.deploy:
        wandb.finish()


def log_gen(deploy, prefs):
    """
    Log generated data information
    """

    if deploy:
        fig, ax = plt.subplots()


def log_train(it, deploy, lr, train_loss):
    """
    Log training loss information
    """

    if deploy and len(train_loss) > 0:
        log_dict = {"train_loss": {k: np.mean(v) for k, v in train_loss.items()}}
        wandb.log(log_dict)

    train_loss = {k: [] for k in train_loss.keys()}
    return train_loss


def log_eval(cfg, it, grammaticality_results):
    """
    Log eval information
    """
    
    if cfg.deploy and grammaticality_results is not None:
        prefs = [grammaticality_results["failures"]]
        for i in range(cfg.data.max_len):
            prefs.append(grammaticality_results["prefix"][i])

        fig, ax = plt.subplots()
        plot_pref = {i: v for i, v in enumerate(prefs) if v > 0}
        plot_pref[0] = prefs[0]
        ax.plot(plot_pref.keys(), plot_pref.values())
        wandb.log({"eval": {"failures": prefs[0], "prefixes": fig}})


def save_model(cfg, net, optimizer, it):
    """
    Save model checkpoint
    """

    if cfg.deploy:
        checkpoint = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": it,
            "config": cfg,
        }
        fdir = cfg.work_dir + "/models/" + cfg.data.language
        os.makedirs(fdir, exist_ok=True)
        if cfg.log.save_multiple:
            fname = os.path.join(fdir, "ckpt_" + str(it+1) + ".pt")
        else:
            fname = os.path.join(fdir, "latest_ckpt.pt")
        torch.save(checkpoint, fname)
