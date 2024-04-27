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
    lr_schedule_values=None,
    start_step: int = 0,
):
    print("-" * os.get_terminal_size().columns)
    print(f"Epoch {epoch}:")
    model.train()
    loss_func = nn.CrossEntropyLoss()
    loss_values = []

    it = start_step
    if lr_schedule_values is not None:
        print(f"Start lr={lr_schedule_values[it]}")
    for videos, labels in tqdm(dataloader):
        if lr_schedule_values is not None:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule_values[it]

        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        preds = model(videos)

        loss = loss_func(preds, labels)
        loss.backward()

        optimizer.step()

        loss_values.append(loss.item())
        it = it + 1
    print()
    if lr_schedule_values is not None:
        print(f"End lr={lr_schedule_values[it-1]}")
    loss_values = np.asarray(loss_values)
    print(f"Average loss={np.mean(loss_values)}")

    print("-" * os.get_terminal_size().columns)
    return loss_values


def valid_one_epoch(
    model: nn.Module,
    dataloader: Iterable,
    device: torch.device,
):
    print("- " * (os.get_terminal_size().columns // 2))
    print("Start validation")

    model.eval()
    correct = 0
    total = 0
    loss_func = nn.CrossEntropyLoss()
    loss_values = []
    with torch.no_grad():
        for videos, labels in tqdm(dataloader):
            videos = videos.to(device)
            labels = labels.to(device)
            preds = model(videos)
            loss = loss_func(preds, labels)
            loss_values.append(loss.item())

            preds = torch.argmax(preds, dim=-1)

            correct += torch.sum(preds == labels)
            total += len(preds)

    loss_values = np.asarray(loss_values)
    acc = correct / total
    if isinstance(acc, torch.Tensor):
        acc = acc.item()
    print(f"Acc: {acc:.4f}")
    print(f"Avg Loss: {np.mean(loss_values):.4f}")
    print("- " * (os.get_terminal_size().columns // 2))
    return acc, loss_values

