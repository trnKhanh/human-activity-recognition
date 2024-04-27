import torch
from torch import nn, optim
import os
from model import stvit_base_patch16_224


def create_model(
    name: str,
    **kwargs,
):
    if name == "stvit_base_patch16_224":
        return stvit_base_patch16_224(**kwargs)
    else:
        raise NameError(f"Model {name} is not found")


def save_model(
    ckpt_path: str,
    args,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
):
    print(f"Save model to {ckpt_path}")
    to_save = {
        "model": model.state_dict(),
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
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

