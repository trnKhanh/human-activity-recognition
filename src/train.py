import os
from typing import Iterable

import torch
from torch import nn

from tqdm import tqdm
import numpy as np


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    lr_schedule_values = None,
    start_step: int = 0,
):
    print("-" * os.get_terminal_size().columns)
    print(f"Epoch {epoch}:")
    model.train()
    loss_func = nn.CrossEntropyLoss()
    loss_values = []

    it = start_step
    for id, (videos, labels) in enumerate(dataloader):
        it = it + 1
        if lr_schedule_values is not None:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule_values[it]

        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        preds = model(videos)

        loss = loss_func(preds, labels)
        loss.backward()
        print(f"\r{id}: loss={loss.item()}", end="") 

        optimizer.step()

        loss_values.append(loss.item())
    print()
    loss_values = np.asarray(loss_values)
    print(f"Average loss={np.mean(loss_values)}")

    print("-" * os.get_terminal_size().columns)
    return loss_values
