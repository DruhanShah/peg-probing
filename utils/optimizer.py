import math
import inspect
import torch
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
    param_dict = {pn: p for pn, p in net.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': optim_cfg.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and torch.cuda.is_available()
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(
        optim_groups, lr=optim_cfg.learning_rate,
        betas=(optim_cfg.beta1, optim_cfg.beta2), **extra_args)
    print(f"Using fused AdamW" if use_fused else "Not using fused AdamW")

    return optimizer


def update_cosine_warmup_lr(it, cfg, optimizer, total_steps):
    """
    Update learning rate with cosine warmup
    """
    it += 1
    lr = cfg.learning_rate

    if cfg.decay_lr:
        if it < cfg.warmup_steps:
            lr = lr * (it) / cfg.warmup_steps
        else:
            num = (it - cfg.warmup_steps)
            decay_ratio = num / (total_steps - cfg.warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = cfg.min_lr + coeff * (lr - cfg.min_lr)
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return it, lr
