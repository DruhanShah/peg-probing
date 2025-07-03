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

    assert(cfg.model.n_ctx >= max_len)
    assert(cfg.model.d_m% cfg.model.d_h == 0)

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


def open_log(cfg):
    """
    Open log file but don't redirect stdout and stderr to it
    """

    os.makedirs(cfg.work_dir + "/logs/", exist_ok=True)
    return open(f"{cfg.work_dir}/logs/{cfg.lang}.log", "w")


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


def cleanup(cfg, fp):
    """
    Close log file and wandb
    """

    if fp is not None:
        fp.close()
    if cfg.deploy:
        wandb.finish()
    return None


def log_debug(it, fp, deploy, debug_info):
        """
        Log debug information
        """

        log_file = fp
        
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
        fig.show()

    stats = {}
    return stats


def log_train(it, deploy, lr, train_loss):
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


def log_eval(it, deploy, eval_results):
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
        fdir = cfg.work_dir + "/models/" + cfg.lang
        os.makedirs(fdir, exist_ok=True)
        if cfg.log.save_multiple:
            fname = os.path.join(fdir, f"ckpt_{it}.pt")
        else:
            fname = os.path.join(fdir, "latest_ckpt.pt")
        torch.save(checkpoint, fname)


def save_probe(cfg, probe, optimizer, it, task="probe"):
    """
    Save probe checkpoint
    """

    if cfg.deploy:
        checkpoint = {
            "net": probe.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": it,
            "config": cfg,
        }
        fdir = cfg.work_dir + "/probes/" + cfg.lang
        os.makedirs(fdir, exist_ok=True)
        if cfg.log.save_multiple:
            fname = os.path.join(fdir, f"{task}_{it}.pt")
        else:
            fname = os.path.join(fdir, "latest_ps.pt")
        torch.save(checkpoint, fname)
