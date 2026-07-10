#!/usr/bin/env python3
"""
COCO Micro-LR Refinement for AvianRaptorNet.

Loads the best existing COCO checkpoint (default: avian_coco_refined_ema.pth)
and continues training with an extremely small learning rate to push the
validation accuracy to its practical limit.

Usage:
    python coco_micro_refinement.py
    python coco_micro_refinement.py --lr 5e-7 --epochs 40 --ema-decay 0.9999
"""

import argparse
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .coco import (
    CocoCropDataset,
    ModelEMA,
    AvianRaptorNet_Fast,
    build_model,
    evaluate,
    get_or_build_crops,
    set_seed,
)

# ============================================================
# Hardware Tuning
# ============================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# ============================================================
# Configuration
# ============================================================
DATA_DIR = "./data/coco_subset_20k"

# Start from the best checkpoint produced by the main pipeline.
BEST_CHECKPOINT = "avian_coco_refined_ema.pth"
MICRO_SAVE = "avian_coco_micro.pth"
MICRO_EMA_SAVE = "avian_coco_micro_ema.pth"
LOG_FILE = "coco_micro_training.log"

NUM_CLASSES = 80
IMG_SIZE = 128
BATCH_SIZE = 12
NUM_WORKERS = 0
PIN_MEMORY = True
VAL_RATIO = 0.10
SEED = 42

# Micro refinement hyper-parameters
MICRO_EPOCHS = 30
MICRO_LR = 1e-6
MICRO_WD = 1e-6          # very low weight decay
MICRO_LABEL_SMOOTHING = 0.0
MIXUP_PROB = 0.0         # disable strong augment regularization
GRAD_CLIP_NORM = 1.0
EMA_DECAY = 0.9999       # slower EMA update for stability at micro LR
PATIENCE = 10            # early-stopping patience (epochs)
DELTA = 0.05             # minimum improvement to reset patience


def get_transforms(phase: str, img_size: int):
    """Minimal augmentation for micro refinement."""
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if phase == "train":
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.15)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(*stats),
        ])

    # val
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])


def append_log(message: str) -> None:
    """Append a timestamped line to the log file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


@torch.no_grad()
def evaluate_loss(model, dataloader, criterion, device):
    """Return average validation loss plus accuracy."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    for inputs, targets in dataloader:
        inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast("cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        total_loss += loss.item() * targets.size(0)
    return (total_loss / total if total > 0 else 0.0,
            100.0 * correct / total if total > 0 else 0.0)


def train_micro(model, train_loader, val_loader, device, args):
    print("\n========== MICRO-LR REFINEMENT STAGE ==========")
    print(f"[INFO] Starting LR: {args.lr:.2e}, WD: {args.weight_decay:.2e}, "
          f"EMA decay: {args.ema_decay}, Epochs: {args.epochs}")
    append_log(f"START lr={args.lr:.2e} wd={args.weight_decay:.2e} "
               f"ema_decay={args.ema_decay} epochs={args.epochs}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    scaler = GradScaler("cuda")
    ema = ModelEMA(model, decay=args.ema_decay)

    best_acc = args.baseline_acc
    best_ema_acc = args.baseline_acc
    best_state = None
    best_ema_state = None
    epochs_without_improvement = 0
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Micro Epoch {epoch+1}/{args.epochs}")
        for inputs, targets in pbar:
            inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                "Loss": f"{running_loss / (total / args.batch_size + 1e-6):.4f}",
                "TrainAcc": f"{100.0 * correct / total:.2f}%",
                "LR": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        scheduler.step()
        train_acc = 100.0 * correct / total if total > 0 else 0.0
        val_loss, val_acc = evaluate_loss(model, val_loader, criterion, device)
        ema_val_acc = evaluate(ema.ema, val_loader, device)

        status = (
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}% | EMA Val Acc: {ema_val_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Best raw: {best_acc:.2f}% | Best EMA: {best_ema_acc:.2f}%"
        )
        print(status)
        append_log(status)

        improved = False
        if val_acc > best_acc + DELTA:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            print(f"  --> New best raw accuracy: {best_acc:.2f}%")
            append_log(f"NEW_BEST_RAW {best_acc:.2f}%")
            improved = True

        if ema_val_acc > best_ema_acc + DELTA:
            best_ema_acc = ema_val_acc
            best_ema_state = copy.deepcopy(ema.state_dict())
            print(f"  --> New best EMA accuracy: {best_ema_acc:.2f}%")
            append_log(f"NEW_BEST_EMA {best_ema_acc:.2f}%")
            improved = True

        if improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"[INFO] Early stopping triggered after {epoch+1} epochs "
                  f"(no improvement for {args.patience} epochs).")
            append_log(f"EARLY_STOP epoch={epoch+1}")
            break

    # Save best checkpoints (fallback to the loaded checkpoint if no improvement).
    if best_state is None:
        print("[INFO] Micro refinement did not improve raw accuracy; "
              f"keeping loaded checkpoint as {MICRO_SAVE}")
        best_state = model.state_dict()
    if best_ema_state is None:
        print("[INFO] Micro refinement did not improve EMA accuracy; "
              f"keeping loaded EMA as {MICRO_EMA_SAVE}")
        best_ema_state = ema.state_dict()

    torch.save(best_state, MICRO_SAVE)
    torch.save(best_ema_state, MICRO_EMA_SAVE)

    elapsed = time.time() - start
    summary = (
        f"[INFO] Micro stage finished in {elapsed / 3600:.2f}h. "
        f"Best raw: {best_acc:.2f}% ({MICRO_SAVE}), "
        f"Best EMA: {best_ema_acc:.2f}% ({MICRO_EMA_SAVE})"
    )
    print(summary)
    append_log(summary)
    return best_acc, best_ema_acc


def main():
    parser = argparse.ArgumentParser(description="Micro-LR COCO refinement")
    parser.add_argument("--checkpoint", type=str, default=BEST_CHECKPOINT,
                        help="Best checkpoint to start micro refinement from")
    parser.add_argument("--epochs", type=int, default=MICRO_EPOCHS,
                        help="Number of micro-refinement epochs")
    parser.add_argument("--lr", type=float, default=MICRO_LR,
                        help="Micro learning rate")
    parser.add_argument("--weight-decay", type=float, default=MICRO_WD,
                        help="Weight decay for micro stage")
    parser.add_argument("--ema-decay", type=float, default=EMA_DECAY,
                        help="EMA decay coefficient (higher = slower update)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout probability (should match training checkpoint)")
    parser.add_argument("--drop-path-rate", type=float, default=0.0,
                        help="Stochastic depth rate (should match training checkpoint)")
    parser.add_argument("--label-smoothing", type=float, default=MICRO_LABEL_SMOOTHING)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP_NORM)
    parser.add_argument("--patience", type=int, default=PATIENCE,
                        help="Early-stopping patience")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--target-img-count", type=int, default=20000)
    parser.add_argument("--min-area", type=int, default=1024)
    args = parser.parse_args()

    set_seed(args.seed)

    print(
        f"[INFO] Micro refinement config: checkpoint={args.checkpoint}, "
        f"epochs={args.epochs}, lr={args.lr:.2e}, wd={args.weight_decay:.2e}, "
        f"ema_decay={args.ema_decay}, batch={args.batch_size}, img_size={args.img_size}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    # Data
    all_crops = get_or_build_crops(target_img_count=args.target_img_count, min_area=args.min_area)
    if len(all_crops) == 0:
        raise RuntimeError("No valid COCO crops found. Check data directory.")

    random.shuffle(all_crops)
    split_idx = int(len(all_crops) * (1 - VAL_RATIO))
    train_crops = all_crops[:split_idx]
    val_crops = all_crops[split_idx:]
    print(f"[INFO] Train crops: {len(train_crops)} | Val crops: {len(val_crops)}")

    train_dataset = CocoCropDataset(train_crops, transform=get_transforms("train", args.img_size))
    val_dataset = CocoCropDataset(val_crops, transform=get_transforms("val", args.img_size))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=PIN_MEMORY,
    )

    # Model
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model = build_model(
        device,
        num_classes=NUM_CLASSES,
        weights_path=args.checkpoint,
        dropout=args.dropout,
        drop_path_rate=args.drop_path_rate,
    )
    baseline_acc = evaluate(model, val_loader, device)
    print(f"[INFO] Loaded checkpoint val accuracy (online): {baseline_acc:.2f}%")

    # Determine a realistic EMA baseline if an EMA checkpoint was loaded.
    ema_baseline = baseline_acc
    if "ema" in os.path.basename(args.checkpoint).lower():
        ema_baseline = baseline_acc
    args.baseline_acc = max(baseline_acc, ema_baseline)
    print(f"[INFO] Baseline accuracy for micro refinement: {args.baseline_acc:.2f}%")
    append_log(f"LOAD checkpoint={args.checkpoint} baseline={args.baseline_acc:.2f}%")

    best_acc, best_ema_acc = train_micro(model, train_loader, val_loader, device, args)

    print("\n========== MICRO REFINEMENT SUMMARY ==========")
    print(f"Baseline accuracy:      {args.baseline_acc:.2f}%")
    print(f"Micro best raw:         {best_acc:.2f}%  -> {MICRO_SAVE}")
    print(f"Micro best EMA:         {best_ema_acc:.2f}%  -> {MICRO_EMA_SAVE}")
    print("==============================================")


if __name__ == "__main__":
    main()
