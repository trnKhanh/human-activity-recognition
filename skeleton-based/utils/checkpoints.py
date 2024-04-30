import torch
from torch import nn, optim


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    lr_scheduler,
    epoch: int,
):
    print(f"Saving model to {path}")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
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
    if "scheduler" in state_dict:
        lr_scheduler.load_state_dict(state_dict["scheduler"])
    if "epoch" in state_dict:
        return state_dict["epoch"] + 1
    else:
        return 1
