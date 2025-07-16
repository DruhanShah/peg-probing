import math
import inspect
import torch
from torch import optim
from typing import List


def move_to_device(vars_to_move: List[torch.Tensor], device):
    """
    Move data to device
    """    
    if len(vars_to_move) == 1:
        return vars_to_move[0].to(device)
    else:
        return [v.to(device) for v in vars_to_move]


def _choose_optimizer(groups, cfg):
    optims = {
        "adamw": lambda: optim.AdamW(
            groups,
            lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps, weight_decay=cfg.weight_decay,
        ),
        "sgd": lambda: optim.SGD(
            groups,
            lr=cfg.learning_rate, momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        ),
    } 

    return optims[cfg.optimizer]()


def _choose_scheduler(optimizer, cfg):
    schedulers = {
        "constant": lambda: optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=0,
        ),
        "warmup": lambda: optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, total_iters=cfg.warmup_steps,
        ),
        "cosine": lambda: optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.restart_steps, eta_min=cfg.min_lr,
        ),
        "exponential": lambda: optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=1-cfg.decay_factor,
        ),
    }

    scheduler_list = ["warmup", cfg.scheduler] if cfg.decay_lr else ["constant"]
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[schedulers[name]() for name in scheduler_list],
        milestones=[cfg.warmup_steps],
    )

    return scheduler
        

def configure_optimizers(net, optim_cfg):
    """
    Configure the optimizer and the learning rate scheduler
    """
    param_dict = {pn: p for pn, p in net.named_parameters() if p.requires_grad}

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': optim_cfg.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    
    optimizer = _choose_optimizer(optim_groups, optim_cfg)
    scheduler = _choose_scheduler(optimizer, optim_cfg)

    return optimizer, scheduler
