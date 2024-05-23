import os
import json
from argparse import ArgumentParser

import numpy as np

from src.datasets.NTUDataset import NTUDataset
from src.models.net import STGCN

from src.utils.optims import CosineSchedule
from src.utils.engines import (
    train_one_epoch,
    valid_one_epoch,
    valid_essemble_one_epoch,
)
from src.utils.checkpoints import load_checkpoint, save_checkpoint

import torch
from torch.utils.data import DataLoader
from torch import optim, nn


def create_args():
    parser = ArgumentParser()

    parser.add_argument("--train", action="store_true", help="Whether to train")
    parser.add_argument(
        "--valid", action="store_true", help="Whether to validate"
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="Device to use (default: cpu)"
    )
    parser.add_argument(
        "--log-path", default="", type=str, help="Where to save log"
    )
    parser.add_argument(
        "--eval-log-path",
        default="",
        type=str,
        help="Where to save final evaluation log",
    )
    # Dataset
    parser.add_argument(
        "--features",
        default=["j"],
        nargs="+",
        type=str,
        help="Features to use as inputs. If model ensemble is used, this should match the order of given models",
    )
    parser.add_argument(
        "--length-t",
        default=64,
        type=int,
        help="Number of frames in each sample",
    )
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
    parser.add_argument(
        "--max-epochs",
        default=80,
        type=int,
        help="Max epochs in cosine schedule (default: 80)",
    )
    # Training
    parser.add_argument(
        "--epochs",
        default=100,
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
        "--save-freq",
        default=10,
        type=int,
        help="How often to save checkpoint (default: 10)",
    )
    parser.add_argument(
        "--save-best-path",
        default="",
        type=str,
        help="Where to save best checkpoint",
    )
    parser.add_argument(
        "--save-best-acc-path",
        default="",
        type=str,
        help="Where to save highest accuracy checkpoint",
    )
    parser.add_argument(
        "--resume", default="", type=str, help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--load-ckpt", default="", type=str, help="Load checkpoint"
    )
    # Model
    parser.add_argument(
        "--ensemble",
        default=[],
        nargs="*",
        type=str,
        help="List of models used in multi-stream",
    )
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
        "--adaptive",
        action="store_true",
        help="Whether to use adaptive graph for graph edges",
    )
    return parser.parse_args()


def main(args):
    train_datasets = []
    valid_datasets = []
    train_dataloaders = []
    valid_dataloaders = []
    for feature in args.features:
        train_datasets.append(
            NTUDataset(
                data_path=args.data_path,
                extra_data_path=args.extra_data_path,
                mode="train",
                split=args.split,
                features=feature,
                length_t=args.length_t,
            )
        )

        valid_datasets.append(
            NTUDataset(
                data_path=args.data_path,
                extra_data_path=args.extra_data_path,
                mode="valid",
                split=args.split,
                features=feature,
                length_t=args.length_t,
            )
        )

        train_dataloaders.append(
            DataLoader(
                dataset=train_datasets[-1],
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        )
        valid_dataloaders.append(
            DataLoader(
                dataset=valid_datasets[-1],
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        )
    args.device = torch.device(args.device)

    print("=" * os.get_terminal_size().columns)
    print("Datasets")
    print(f"  Data path: {args.data_path}")
    if len(args.extra_data_path) > 0:
        print(f"  Extra data path: {args.extra_data_path}")
    for id in range(len(args.features)):
        print("-" * os.get_terminal_size().columns)
        print(f"  Feature: {args.features[id]}")
        print(f"  Train size: {len(train_datasets[id])} samples")
        print(f"  Valid size: {len(valid_datasets[id])} samples")
        print("-" * os.get_terminal_size().columns)
    print("=" * os.get_terminal_size().columns)

    num_features = args.features[0].count(",") + 1
    model = STGCN(
        3 * num_features,
        args.num_classes,
        act_layer=nn.ReLU,
        dropout_rate=args.dropout_rate,
        adaptive=args.adaptive,
    )
    model.to(args.device)
    num_params = sum([p.numel() for p in model.parameters()])
    optimizer = optim.SGD(
        model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0004
    )

    steps_per_epoch = len(train_datasets[0]) // args.batch_size
    warmup_steps = args.warmup_epochs
    max_steps = args.max_epochs
    lr_scheduler = CosineSchedule(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        base_lr=args.base_lr,
        target_lr=args.target_lr,
        max_steps=max_steps,
        cur_step=args.start_epoch - 1,
    )
    if len(args.load_ckpt) > 0 and os.path.isfile(args.load_ckpt):
        state_dict = torch.load(args.load_ckpt, map_location=args.device)
        if "model" in state_dict:
            model.load_state_dict(state_dict["model"])
        else:
            model.load_state_dict(state_dict)
        print(f"Loaded model from {args.load_ckpt}")

    min_loss = np.Inf
    max_acc = -np.Inf

    if len(args.resume) > 0 and os.path.isfile(args.resume):
        args.start_epoch, min_loss = load_checkpoint(
            args.resume, model, optimizer, lr_scheduler, args.device,
        )
        print(f"Resuming from epoch {args.start_epoch}, min_loss={min_loss}")

    loss_fn = nn.CrossEntropyLoss()

    if args.train:
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
        for e in range(args.start_epoch, args.epochs + 1):
            train_avg_loss, train_acc = train_one_epoch(
                epoch=e,
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                dataloader=train_dataloaders[0],
                device=args.device,
                start_step=(e - 1) * steps_per_epoch + 1,
                lr_schedule=lr_scheduler,
            )
            valid_avg_loss, valid_acc, _, _ = valid_one_epoch(
                model=model,
                loss_fn=loss_fn,
                dataloader=valid_dataloaders[0],
                device=args.device,
            )
            if len(args.log_path) > 0:
                log.append(
                    {
                        "epoch": e,
                        "train_avg_loss": train_avg_loss,
                        "train_acc": train_acc,
                        "valid_avg_loss": valid_avg_loss,
                        "valid_acc": valid_acc,
                    }
                )
                with open(args.log_path, "w", encoding="utf-8") as f:
                    json.dump(log, f, ensure_ascii=False, indent=2)

            if valid_avg_loss < min_loss and len(args.save_best_path) > 0:
                save_checkpoint(
                    args.save_best_path,
                    model,
                    optimizer,
                    lr_scheduler,
                    e,
                    valid_avg_loss,
                )

            if valid_acc > max_acc and len(args.save_best_acc_path) > 0:
                save_checkpoint(
                    args.save_best_acc_path,
                    model,
                    optimizer,
                    lr_scheduler,
                    e,
                    valid_avg_loss,
                )
            min_loss = min(min_loss, valid_avg_loss)
            max_acc = max(max_acc, valid_acc)

            if len(args.save_path) > 0:
                if (e % args.save_freq) == 0 or e == args.epochs:
                    save_checkpoint(
                        args.save_path,
                        model,
                        optimizer,
                        lr_scheduler,
                        e,
                        min_loss,
                    )

    if args.valid:
        valid_avg_loss, valid_acc, preds, labels = valid_one_epoch(
            model=model,
            loss_fn=loss_fn,
            dataloader=valid_dataloaders[0],
            device=args.device,
        )
        print(f"Evalution accuracy: {valid_acc}")
        if len(args.eval_log_path) > 0:
            os.makedirs(os.path.dirname(args.eval_log_path), exist_ok=True)
            with open(args.eval_log_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"preds": preds, "labels": labels},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
    if len(args.ensemble) > 0:
        assert len(args.features) == len(args.ensemble)
        models = nn.ModuleList()
        for i in range(len(args.ensemble)):
            num_features = args.features[i].count(",") + 1
            models.append(
                STGCN(
                    3 * num_features,
                    args.num_classes,
                    act_layer=nn.ReLU,
                    dropout_rate=args.dropout_rate,
                    adaptive=args.adaptive,
                )
            )

            state_dict = torch.load(args.ensemble[i], map_location=args.device)
            if "model" in state_dict:
                models[-1].load_state_dict(state_dict["model"])
            else:
                models[-1].load_state_dict(state_dict)
            print(f"Loaded model from {args.ensemble[i]}")
        models.to(args.device)
        valid_acc, preds, labels = valid_essemble_one_epoch(
            models, valid_dataloaders, args.device
        )
        print(f"Evalution accuracy: {valid_acc}")
        if len(args.eval_log_path) > 0:
            os.makedirs(os.path.dirname(args.eval_log_path), exist_ok=True)
            with open(args.eval_log_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"preds": preds, "labels": labels},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )


if __name__ == "__main__":
    args = create_args()
    main(args)
