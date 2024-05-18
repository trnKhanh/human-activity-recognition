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
    model.train()
    cur_step = start_step

    loss_values = []
    correct_count = 0
    total_count = 0
    with tqdm(dataloader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        if lr_schedule is not None:
            lr_schedule.step()
        for samples, labels in tepoch:

            samples = samples.to(device)
            labels = labels.to(device)

            preds = model(samples)
            optimizer.zero_grad()
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()

            loss_values.append(loss.item())

            with torch.no_grad():
                pred_classes = torch.argmax(preds, dim=1)
                correct_count += (pred_classes == labels).sum().item()
                total_count += len(labels)
                tepoch.set_postfix(
                    lr=optimizer.param_groups[0]["lr"],
                    avg_loss=np.mean(np.array(loss_values)),
                    acc=correct_count / total_count,
                )
                cur_step += 1

        avg_loss = sum(loss_values) / len(loss_values)

    acc = correct_count / total_count
    return avg_loss, acc 


def valid_one_epoch(
    model: nn.Module, loss_fn, dataloader: DataLoader, device: torch.device
):
    model.eval()

    preds_arr = []
    labels_arr = []

    loss_values = []
    correct_count = 0
    total_count = 0
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description("Validation")
            for samples, labels in tepoch:
                samples = samples.to(device)
                labels = labels.to(device)

                preds = model(samples)
                loss = loss_fn(preds, labels)

                loss_values.append(loss.item())

                pred_classes = torch.argmax(preds, dim=1)
                correct_count += (pred_classes == labels).sum().item()

                preds_arr.extend(pred_classes.tolist())
                labels_arr.extend(labels.tolist())

                total_count += len(labels)
                tepoch.set_postfix(
                    avg_loss=np.mean(np.array(loss_values)),
                    acc=correct_count / total_count,
                )

            avg_loss = sum(loss_values) / len(loss_values)
            acc = correct_count / total_count

    return avg_loss, acc, preds_arr, labels_arr
