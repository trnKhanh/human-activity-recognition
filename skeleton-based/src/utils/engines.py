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
    with tqdm(dataloader, unit="batch", ncols=0) as tepoch:
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

    score = None
    with torch.no_grad():
        with tqdm(dataloader, unit="batch", ncols=0) as tepoch:
            tepoch.set_description("Validation")
            for samples, labels in tepoch:
                samples = samples.to(device)
                labels = labels.to(device)

                preds = model(samples)

                score = preds if score is None else torch.cat([score, preds])

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

    return avg_loss, acc, preds_arr, labels_arr, score


def valid_essemble_one_epoch(
    models: nn.ModuleList, dataloaders: list[DataLoader], device: torch.device
):
    dataloader_iters = []
    for dataloader in dataloaders:
        dataloader_iters.append(iter(dataloader))

    models.eval()

    preds_arr = []
    labels_arr = []

    correct_count = 0
    total_count = 0
    with torch.no_grad():
        with tqdm(range(len(dataloaders[0])), unit="batch", ncols=0) as tepoch:
            tepoch.set_description("Validation")
            for _ in tepoch:
                probs = torch.zeros(0)
                labels = torch.empty(0)
                for i in range(len(models)):
                    samples, labels = next(dataloader_iters[i])

                    samples = samples.to(device)
                    labels = labels.to(device)
                    preds = models[i](samples)
                    # preds = preds.softmax(dim=-1)
                    probs = probs + preds if probs.size(0) != 0 else preds

                pred_classes = torch.argmax(probs, dim=1)
                correct_count += (pred_classes == labels).sum().item()

                preds_arr.extend(pred_classes.tolist())
                labels_arr.extend(labels.tolist())

                total_count += len(labels)
                tepoch.set_postfix(
                    acc=correct_count / total_count,
                )

            acc = correct_count / total_count

    return acc, preds_arr, labels_arr
