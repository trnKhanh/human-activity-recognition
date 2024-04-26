import os
import argparse

import torch
from torch.utils.data import DataLoader

from timm.models import create_model

from datasets import VideoDataset
from train import train_one_epoch
from optim import create_optimizer, cosine_scheduler

from model import VisionTransformer

from utils import save_model, load_model


def create_args():
    parser = argparse.ArgumentParser()
    # Data argument
    parser.add_argument(
        "--anno-path", required=True, type=str, help="Path to annotations"
    )
    parser.add_argument(
        "--data-path", required=True, type=str, help="Path to dataset"
    )
    parser.add_argument(
        "--input-size",
        default=224,
        type=int,
        help="Size of input (default: 224)",
    )
    parser.add_argument(
        "--clip-size",
        default=16,
        type=int,
        help="Number of frames per clip (default: 16)",
    )
    parser.add_argument(
        "--sampling-rate",
        default=2,
        type=int,
        help="Sampling rate (default: 2)",
    )

    # Model argument
    parser.add_argument(
        "--patch-size",
        default=16,
        type=int,
        help="Size of spatial patch (default: 16)",
    )
    parser.add_argument(
        "--tube-size",
        default=2,
        type=int,
        help="Size of temporal patch (default: 2)",
    )
    parser.add_argument(
        "--attn-drop-rate",
        default=0,
        type=float,
        help="Attention dropout rate (default: 0)",
    )
    parser.add_argument(
        "--drop-rate",
        default=0,
        type=float,
        help="Dropout rate for FC layers (default: 0)",
    )
    parser.add_argument(
        "--drop-path-rate",
        default=0,
        type=float,
        help="Drop path rate (default: 0)",
    )
    parser.add_argument(
        "--head-drop-rate",
        default=0,
        type=float,
        help="Head dropout rate (default: 0)",
    )

    # Optimizer arguments
    parser.add_argument(
        "--optimizer",
        default="adamw",
        type=str,
        help="Optimizer used for training (default: adamw)",
    )
    parser.add_argument(
        "--epochs", default=500, type=int, help="Epoch to train (default: 500)"
    )
    parser.add_argument(
        "--lr",
        default=1e-4,
        type=float,
        help="Learning rate used for training (default: 1e-4)",
    )
    parser.add_argument(
        "--min-lr",
        default=1e-5,
        type=float,
        help="Lowerbound for learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--warmup-epochs",
        default=40,
        type=int,
        help="Warmup epochs (default: 40)",
    )
    parser.add_argument(
        "--start-warmup-lr",
        default=0,
        type=float,
        help="Warmup learning rate (default: 0)",
    )
    parser.add_argument(
        "--weight-decay",
        default=0.01,
        type=float,
        help="Weight decay (default: 0.01)",
    )

    # Train arguments
    parser.add_argument(
        "--start-epoch", default=0, type=int, help="Epoch to start (default: 0)"
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Whether to auto resume from checkpoint file",
    )
    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        help="Batch size used for training/testing (default: 64)",
    )
    parser.add_argument(
        "--ckpt-save-freq",
        default=10,
        type=int,
        help="How frequency to save checkpoint",
    )
    parser.add_argument(
        "--load-ckpt", action="store_true", help="Whether to load checkpoints"
    )
    parser.add_argument(
        "--ckpt-dir", default="", type=str, help="Where to save checkpoints"
    )
    parser.add_argument(
        "--save-best-ckpt",
        default=True,
        type=bool,
        help="Whether to save best checkpoints, having highest acc on validation set",
    )

    # Other arguments
    parser.add_argument(
        "--device", default="cpu", type=str, help="Device to use (default: cpu)"
    )

    return parser.parse_args()


def main(args):
    SSV2_IMAGE_SIZE = (240, 428)
    train_dataset = VideoDataset(
        anno_path=args.anno_path,
        data_path=args.data_path,
        crop_size=args.input_size,
        img_size=SSV2_IMAGE_SIZE,
        transform=None,
        num_frames=args.clip_size,
        sampling_rate=args.sampling_rate,
        mode="train",
    )
    valid_dataset = VideoDataset(
        anno_path=args.anno_path,
        data_path=args.data_path,
        crop_size=args.input_size,
        img_size=SSV2_IMAGE_SIZE,
        transform=None,
        num_frames=args.clip_size,
        sampling_rate=args.sampling_rate,
        mode="validation",
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True
    )
    print("=" * os.get_terminal_size().columns)
    print("Dataset:")
    print(f"  Annotation path: {args.anno_path}")
    print(f"  Data path: {args.data_path}")
    print(f"  Input size: {args.input_size}")
    print(f"  Clip size: {args.clip_size}")
    print(f"  Sampling rate: {args.sampling_rate}")
    print("=" * os.get_terminal_size().columns)

    device = torch.device(args.device)
    model = create_model(
        "vit_base_patch16_224",
        pretrained=False,
        attn_drop_rate=args.attn_drop_rate,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        head_drop_rate=args.head_drop_rate,
    )
    model.to(device)
    model.train()
    num_params = sum(p.numel() for p in model.parameters())
    print("=" * os.get_terminal_size().columns)
    print(f"Device: {args.device}")
    print(f"Model: {str(model)}")
    print(f"Number of params: {num_params}")
    print("=" * os.get_terminal_size().columns)

    optimizer = create_optimizer(args, model)
    print("=" * os.get_terminal_size().columns)
    print(f"Optimizer: {str(optimizer)}")
    print("=" * os.get_terminal_size().columns)

    if args.load_ckpt and len(args.ckpt_dir) > 0:
        load_model(args.ckpt_dir, args, model, optimizer)

    print("=" * os.get_terminal_size().columns)
    print("Training:")
    print(f"Start epoch: {args.start_epoch}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Checkpoint saving freq: {args.ckpt_save_freq}")
    print(f"Save best checkpoint: {args.save_best_ckpt}")
    print("=" * os.get_terminal_size().columns)

    niter_per_epoch = len(train_dataset) // args.batch_size
    lr_schedule_values = cosine_scheduler(
        epochs=args.epochs,
        base_lr=args.lr,
        min_lr=args.min_lr,
        niter_per_epoch=niter_per_epoch,
        warmup_epochs=args.warmup_epochs,
        start_warmup_lr=args.start_warmup_lr,
    )

    for i in range(args.start_epoch, args.epochs):
        train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            epoch=i,
            lr_schedule_values=lr_schedule_values,
            start_step=i * niter_per_epoch,
        )
        if len(args.ckpt_dir) > 0:
            if (i + 1) % args.ckpt_save_freq == 0 or i + 1 == args.epochs:
                save_model(
                    args.ckpt_dir,
                    args,
                    i,
                    model,
                    optimizer,
                )


if __name__ == "__main__":
    args = create_args()
    main(args)
