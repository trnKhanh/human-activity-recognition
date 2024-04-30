import os
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn,
    dataloader: DataLoader,
    device: torch.device,
    start_step: int,
    lr_schedule=None,
):
    print("-" * os.get_terminal_size().columns)
    print(f"Epoch {epoch}: ")
    model.train()
    cur_step = start_step
    if lr_schedule is not None:
        print(f"  Start lr: {lr_schedule(cur_step):.6f}")
    loss_values = []
    for samples, labels in tqdm(dataloader):
        if lr_schedule is not None:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule(cur_step)

        samples = samples.to(device)
        labels = labels.to(device)

        preds = model(samples)
        optimizer.zero_grad()
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())
        cur_step += 1

    if lr_schedule is not None:
        print(f"  End lr: {lr_schedule(cur_step - 1):.6f}")
    avg_loss = np.mean(np.array(loss_values))
    print(f"  Avg Loss: {avg_loss:.6f}")
    print("-" * os.get_terminal_size().columns)

    return avg_loss, loss_values


def valid_one_epoch(
    model: nn.Module, loss_fn, dataloader: DataLoader, device: torch.device
):

    print("- " * (os.get_terminal_size().columns // 2))
    print(f"Validation")
    model.eval()

    loss_values = []
    correct_count = 0
    total_count = 0
    for samples, labels in tqdm(dataloader):
        samples = samples.to(device)
        labels = labels.to(device)

        preds = model(samples)
        loss = loss_fn(preds, labels)
        
        pred_classes = torch.argmax(preds, dim=-1)
        correct_count += torch.sum(pred_classes == labels).item()
        total_count += len(preds)

        loss_values.append(loss.item())

    avg_loss = np.mean(np.array(loss_values))
    acc = correct_count / total_count
    print(f"  Avg Loss: {avg_loss:.6f}")
    print(f"  Accuracy: {acc:.6f}")
    print("- " * (os.get_terminal_size().columns // 2))

    return avg_loss, acc
