import os
import json
from argparse import ArgumentParser

import numpy as np

from datasets.NTUDataset import NTUDataset
from models.net import STGCN
from models.optims import CosineSchedule
from models.utils import build_A
from utils.graph import get_ntu_graph_args
from utils.engines import train_one_epoch, valid_one_epoch
from utils.checkpoints import load_checkpoint, save_checkpoint

import torch
from torch.utils.data import DataLoader
from torch import optim, nn


def create_args():
    parser = ArgumentParser()

    parser.add_argument("--train", action="store_true", help="Whether to train")
    parser.add_argument(
        "--device", default="cpu", type=str, help="Device to use (default: cpu)"
    )
    parser.add_argument(
        "--log-path", default="", type=str, help="Where to save log"
    )
    # Dataset
    parser.add_argument(
        "--data-path", required=True, type=str, help="Path to dataset"
    )
    parser.add_argument(
        "--extra-data-path",
        default="",
        type=str,
        help="Path to extra dataset (i.e NTU120)",
    )
    parser.add_argument(
        "--split",
        default="x-subject",
        choices=["x-subject", "x-view", "x-setup"],
        help="Split evaluation (default: x-subject)",
    )
    parser.add_argument(
        "--batch-size", default=64, type=int, help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--num-workers",
        default=1,
        type=int,
        help="Number of workers used to load data (default: 1)",
    )
    # Optimizer
    parser.add_argument(
        "--base-lr",
        default=0.005,
        type=float,
        help="Base learning rate (default: 0.005)",
    )
    parser.add_argument(
        "--target-lr",
        default=0.0001,
        type=float,
        help="Target learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--warmup-epochs",
        default=40,
        type=int,
        help="Warm up epochs (default: 40)",
    )
    # Training
    parser.add_argument(
        "--epochs",
        default=1000,
        type=int,
        help="Epochs to train (defult: 1000)",
    )
    parser.add_argument(
        "--start-epoch", default=1, type=int, help="Start epoch (defult: 1)"
    )
    # Checkpoint
    parser.add_argument(
        "--save-path", default="", type=str, help="Where to save checkpoint"
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="Whether to save best checkpoint",
    )
    parser.add_argument(
        "--resume", default="", type=str, help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--save-freq",
        default=10,
        type=int,
        help="How often to save checkpoint (default: 10)",
    )
    # Model
    parser.add_argument(
        "--num-classes",
        default=120,
        type=int,
        help="Number of classes (default: 120)",
    )
    parser.add_argument(
        "--dropout-rate",
        default=0,
        type=float,
        help="Dropout rate (default: 0)",
    )
    parser.add_argument(
        "--importance",
        action="store_true",
        help="Whether to use importance for graph edges",
    )
    return parser.parse_args()


def main(args):
    train_dataset = NTUDataset(
        data_path=args.data_path,
        extra_data_path=args.extra_data_path,
        mode="train",
        split=args.split,
    )

    valid_dataset = NTUDataset(
        data_path=args.data_path,
        extra_data_path=args.extra_data_path,
        mode="valid",
        split=args.split,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    args.device = torch.device(args.device)

    print("=" * os.get_terminal_size().columns)
    print("Dataset")
    print(f"  Data path: {args.data_path}")
    if len(args.extra_data_path) > 0:
        print(f"  Extra data path: {args.extra_data_path}")
    print(f"  Train size: {len(train_dataset)} samples")
    print(f"  Valid size: {len(valid_dataset)} samples")
    print("=" * os.get_terminal_size().columns)

    graph_args = get_ntu_graph_args("spatial")
    A = build_A(graph_args)
    A = A.to(args.device)

    model = STGCN(
        3,
        args.num_classes,
        A=A,
        dropout_rate=args.dropout_rate,
        importance=args.importance,
    )
    model.to(args.device)
    num_params = sum([p.numel() for p in model.parameters()])
    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr)

    steps_per_epoch = len(train_dataset) // args.batch_size
    warmup_steps = args.warmup_epochs * steps_per_epoch
    max_steps = (
        args.epochs - args.start_epoch + 1
    ) * steps_per_epoch - warmup_steps
    lr_scheduler = CosineSchedule(
        warmup_steps=warmup_steps,
        base_lr=args.base_lr,
        target_lr=args.target_lr,
        max_steps=max_steps,
    )
    if len(args.resume) > 0 and os.path.isfile(args.resume):
        args.start_epoch = load_checkpoint(
            args.resume, model, optimizer, lr_scheduler
        )
    loss_fn = nn.CrossEntropyLoss()
    print("=" * os.get_terminal_size().columns)
    print("Model")
    print(model)
    print(f"  Number of parameters: {num_params}")
    print("=" * os.get_terminal_size().columns)

    log = []
    if len(args.log_path) > 0:
        os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    if len(args.save_path) > 0:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    min_loss = np.Inf
    for e in range(args.start_epoch, args.epochs + 1):
        train_avg_loss, train_loss_values = train_one_epoch(
            epoch=e,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            dataloader=train_dataloader,
            device=args.device,
            start_step=(e - 1) * steps_per_epoch + 1,
            lr_schedule=lr_scheduler,
        )
        valid_avg_loss, valid_acc = valid_one_epoch(
            model=model,
            loss_fn=loss_fn,
            dataloader=valid_dataloader,
            device=args.device,
        )
        if len(args.log_path) > 0:
            log.append(
                {
                    "train_avg_loss": train_avg_loss,
                    "train_loss_values": train_loss_values,
                    "valid_avg_loss": valid_avg_loss,
                    "valid_acc": valid_acc,
                }
            )
            with open(args.log_path, "w", encoding="utf-8") as f:
                json.dump(log, f, ensure_ascii=False, indent=4)
        if len(args.save_path) > 0:
            if args.save_best:
                if valid_avg_loss < min_loss:
                    save_checkpoint(
                        args.save_path, model, optimizer, lr_scheduler, e
                    )
            else:
                if (e % args.save_freq) == 0 or e == args.epochs:
                    save_checkpoint(
                        args.save_path, model, optimizer, lr_scheduler, e
                    )

        min_loss = min(min_loss, valid_avg_loss)

    print("=" * os.get_terminal_size().columns)


if __name__ == "__main__":
    args = create_args()
    main(args)
