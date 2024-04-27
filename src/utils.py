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


class MetricValue(object):
    def __init__(self):
        self.n = 0
        self.sum = 0
        self.values = []

    def update(self, value):
        self.n += 1
        self.sum += value
        self.values.append(value)

    def avg(self):
        if self.n != 0:
            return self.sum / self.n
        else:
            return 0

    def __repr__(self):
        return (
            "{"
            + f"n: {self.n}, sum: {self.sum}, avg: {self.avg()}, values: {str(self.values)} "
            + "}"
        )


class MetricLogger(object):
    def __init__(self, log_dir: str = ""):
        self.log_dir = log_dir
        self.data = dict()
        self.cur_epoch = 0

    def update(self, epoch: int, train: bool, **kwargs):
        if epoch not in self.data:
            self.data[epoch] = dict()
            self.data[epoch]["train"] = dict()
            self.data[epoch]["validation"] = dict()

        if train:
            set_name = "train"
        else:
            set_name = "validation"
        for k in kwargs.keys():
            if k not in self.data[epoch][set_name]:
                self.data[epoch][set_name][k] = MetricValue()
            self.data[epoch][set_name][k].update(kwargs[k])

    def save(self):
        if len(self.log_dir) > 0:
            with open(self.log_dir, "a") as f:
                for e in self.data.keys():
                    if e < self.cur_epoch:
                        continue


    def __repr__(self):
        return str(self.data)
