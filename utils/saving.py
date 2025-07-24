import torch
import os
import pickle as pkl

def save_data(cfg, dataset, model=True):
    """
    Save dataset to file
    """
    type = cfg.model_type if model else cfg.probe_type
    fdir = f"{cfg.work_dir}/data/{type}/{cfg.lang}"
    os.makedirs(fdir, exist_ok=True)
    fname = os.path.join(fdir, "dataset.pkl")
    save_dict = {
        "data": dataset.data,
        "labels": dataset.labels,
        "language": dataset.language,
        "max_len": dataset.max_len,
        "seed": dataset.seed
    }
        
    with open(fname, "wb") as f:
        pkl.dump(save_dict, f)
        


def save_model(cfg, net, optimizer, it):
    """
    Save model checkpoint
    """
    checkpoint = {
        "net": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter": it,
        "config": net.cfg,
    }
    fdir = f"{cfg.work_dir}/models/{cfg.model_type}/{cfg.lang}"
    os.makedirs(fdir, exist_ok=True)
    if cfg.log.save_multiple:
        fname = os.path.join(fdir, f"ckpt_{it}.pt")
    else:
        fname = os.path.join(fdir, "ckpt_latest.pt")
    torch.save(checkpoint, fname)


def save_probe(cfg, probe, optimizer, it, task="ps"):
    """
    Save probe checkpoint
    """

    if cfg.deploy:
        checkpoint = {
            "net": probe.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": it,
            "config": probe.cfg,
        }
        fdir = f"{cfg.work_dir}/probes/{cfg.model_type}/{cfg.lang}"
        os.makedirs(fdir, exist_ok=True)
        if cfg.log.save_multiple:
            fname = os.path.join(fdir, f"{task}_{it}.pt")
        else:
            fname = os.path.join(fdir, f"{task}_latest.pt")
        torch.save(checkpoint, fname)
