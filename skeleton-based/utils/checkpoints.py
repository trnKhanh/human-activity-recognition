import numpy as np
import torch
from torch import nn, optim


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    lr_scheduler,
    epoch: int,
    min_loss: float,
):
    print(f"Saving model to {path}")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "min_loss": min_loss,
        },
        path,
    )


def load_checkpoint(
    path: str, model: nn.Module, optimizer: optim.Optimizer, lr_scheduler
):
    print(f"Loading model from {path}")
    state_dict = torch.load(path)
    if "model" in state_dict:
        model.load_state_dict(state_dict["model"])
    if "optimizer" in state_dict:
        optimizer.load_state_dict(state_dict["optimizer"])
    min_loss = np.Inf
    if "min_loss" in state_dict:
        min_loss = state_dict["min_loss"]
    if "epoch" in state_dict:
        return state_dict["epoch"] + 1, min_loss
    else:
        return 1, min_loss
