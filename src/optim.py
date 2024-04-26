import torch
from torch import nn
from torch import optim

import numpy as np


def create_optimizer(args, model: nn.Module) -> torch.optim.Optimizer:
    optimizer_lower = args.optimizer.lower()
    lr = args.lr
    weight_decay = args.weight_decay
    parameters = model.parameters()

    if optimizer_lower == "sgd":
        return optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_lower == "adam":
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_lower == "adamw":
        return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise NameError(f"Optimizer {args.optimizer} is invalid")


def cosine_scheduler(
    epochs: int,
    base_lr: float,
    min_lr: float,
    niter_per_epoch: int,
    warmup_epochs: int = 0,
    start_warmup_lr: float = 0,
):
    warmup_schedule = np.array([])
    warmup_iters = 0
    if warmup_epochs > 0:
        warmup_iters = warmup_epochs * niter_per_epoch
        warmup_schedule = np.linspace(start_warmup_lr, base_lr, warmup_iters)

    max_iters = epochs * niter_per_epoch
    iters = np.arange(max_iters - warmup_iters)

    schedule = np.array(
        [
            min_lr
            + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * i / len(iters)))
            for i in iters
        ]
    )
    schedule = np.concatenate([warmup_schedule, schedule])

    return schedule
