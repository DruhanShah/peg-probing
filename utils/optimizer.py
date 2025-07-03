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
        
        
def configure_optimizers(net, optim_cfg):
    """
    Configure the optimizer and the learning rate scheduler
    """
    param_dict = {pn: p for pn, p in net.named_parameters() if p.requires_grad}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': optim_cfg.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    
    optimizer = optim.SGD(
        optim_groups,
        lr=optim_cfg.learning_rate, momentum=optim_cfg.momentum,
        weight_decay=optim_cfg.weight_decay,
    )

    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1, total_iters=optim_cfg.warmup_steps,
    )
    decay = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=optim_cfg.restart_steps, eta_min=optim_cfg.min_lr,
    ) if optim_cfg.scheduler == "cosine" else optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=1-optim_cfg.decay_factor,
    )

    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, decay],
        milestones=[optim_cfg.warmup_steps],
    ) if optim_cfg.decay_lr else optim.lr_scheduler.ConstantLR(
        optimizer, factor=1.0, total_iters=0,
    )

    return optimizer, scheduler
