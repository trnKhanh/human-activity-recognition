import torch
from torch import nn, optim
import os


def save_model(
    ckpt_path: str,
    args,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
):
    to_save = {
        "model": model,
        "epoch": epoch,
        "optimizer": optimizer,
        "args": args,
    }
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    torch.save(to_save, ckpt_path)


def load_model(
    ckpt_path: str, args, model: nn.Module, optimizer: optim.Optimizer
):
    if not os.path.isfile(ckpt_path):
        return False
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        if args.auto_resume:
            if "optimizer" in checkpoint and "epoch" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
                args.start_epoch = checkpoint["epoch"]
                print(f"Resume training at epoch {args.start_epoch}")
        return True
    return False

