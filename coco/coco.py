#!/usr/bin/env python3
"""
COCO Object-Crop Fine-Tuning for AvianRaptorNet (Enhanced).

Loads the CIFAR-100 pretrained AvianRaptorNet-Fast checkpoint, replaces the
classifier head for 80 COCO categories, and performs an aggressive two-phase
fine-tuning with modern regularization and training tricks:

  1. Primary transfer learning   -> avian_coco_primary.pth (+ EMA variant)
  2. Ultra-low LR refinement   -> avian_coco_refined.pth (+ EMA variant)

Usage:
    python coco.py
    python coco.py --primary-epochs 100 --refine-epochs 30 --img-size 160 --batch-size 8
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy
import pickle
import random
import time
from typing import List, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import zipfile
from PIL import Image
from pycocotools.coco import COCO
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from core.avian_model import AvianRaptorNet_Fast

# ============================================================
# Hardware Tuning (RTX 30-series)
# ============================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# ============================================================
# Configuration
# ============================================================
DATA_DIR = "./data/coco_subset_20k"
IMG_DIR = os.path.join(DATA_DIR, "images")
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANN_FILE = os.path.join(DATA_DIR, "annotations/instances_train2017.json")

PRETRAINED_WEIGHTS = "avian_raptor_fast_best.pth"
PRIMARY_SAVE = "avian_coco_primary.pth"
REFINED_SAVE = "avian_coco_refined.pth"
EMA_PRIMARY_SAVE = "avian_coco_primary_ema.pth"
EMA_REFINED_SAVE = "avian_coco_refined_ema.pth"


def get_crops_cache_path(target_img_count: int, min_area: int) -> str:
    return os.path.join(DATA_DIR, f"crops_cache_img{target_img_count}_area{min_area}.pkl")


NUM_CLASSES = 80
IMG_SIZE = 128           # default; can be raised to 160 on 8 GB with smaller batch
BATCH_SIZE = 12
NUM_WORKERS = 0          # avoid forked CUDA-context overhead on 8 GB VRAM
PIN_MEMORY = True

# Primary stage
PRIMARY_EPOCHS = 80
PRIMARY_LR = 0.01
PRIMARY_MOMENTUM = 0.9
PRIMARY_WD = 1e-4
PRIMARY_LABEL_SMOOTHING = 0.1
WARMUP_EPOCHS = 5

# Refinement stage
REFINE_EPOCHS = 25
REFINE_LR = 1e-5
REFINE_WD = 1e-4

# Regularization
DROPOUT = 0.2
DROP_PATH_RATE = 0.0

# Augmentation
MIXUP_ALPHA = 0.8
CUTMIX_ALPHA = 1.0
RANDOM_ERASING_PROB = 0.1

# Training
GRAD_CLIP_NORM = 1.0
EMA_DECAY = 0.999

VAL_RATIO = 0.10
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 1. Data Preparation
# ============================================================
def download_file(url: str, target_path: str, timeout: int = 300) -> None:
    """Download a file with a progress bar."""
    if os.path.exists(target_path):
        return
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    print(f"[INFO] Downloading {os.path.basename(target_path)}...")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(target_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def ensure_annotations() -> None:
    """Make sure COCO train2017 annotations are available locally."""
    if os.path.exists(ANN_FILE):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    ann_zip = os.path.join(DATA_DIR, "annotations.zip")
    download_file(ANN_URL, ann_zip)
    print("[INFO] Extracting annotations...")
    with zipfile.ZipFile(ann_zip, "r") as z:
        z.extractall(DATA_DIR)


def build_crop_list(target_img_count: int = 20000, min_area: int = 1024) -> List[Tuple[str, List[int], int]]:
    """
    Build a list of object crops from the COCO subset.
    Each item is (image_path, [x, y, w, h], label_index).
    """
    ensure_annotations()
    os.makedirs(IMG_DIR, exist_ok=True)

    coco = COCO(ANN_FILE)
    cats = coco.loadCats(coco.getCatIds())
    cat_id_to_label = {cat["id"]: i for i, cat in enumerate(cats)}

    all_ids = list(coco.imgs.keys())
    random.shuffle(all_ids)

    crops = []
    print(f"[INFO] Scanning up to {target_img_count} images for valid object crops...")
    for img_id in tqdm(all_ids[:target_img_count]):
        img_info = coco.loadImgs(img_id)[0]
        fname = img_info["file_name"]
        fpath = os.path.join(IMG_DIR, fname)

        if not os.path.exists(fpath):
            continue

        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            if ann["area"] < min_area:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 1 or h <= 1:
                continue
            label = cat_id_to_label[ann["category_id"]]
            crops.append((fpath, [float(x), float(y), float(w), float(h)], label))

    print(f"[INFO] Total object crops collected: {len(crops)}")
    return crops


def get_or_build_crops(target_img_count: int = 20000, min_area: int = 1024) -> List[Tuple[str, List[int], int]]:
    """Load cached crop list or build and cache it."""
    cache_path = get_crops_cache_path(target_img_count, min_area)
    if os.path.exists(cache_path):
        print(f"[INFO] Loading cached crop list from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    crops = build_crop_list(target_img_count, min_area)
    with open(cache_path, "wb") as f:
        pickle.dump(crops, f)
    print(f"[INFO] Cached crop list to {cache_path}")
    return crops


class CocoCropDataset(Dataset):
    """On-the-fly object-crop classification dataset."""

    def __init__(self, crops: List[Tuple[str, List[int], int]], transform=None):
        self.crops = crops
        self.transform = transform

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, index):
        fpath, bbox, label = self.crops[index]
        try:
            img = Image.open(fpath).convert("RGB")
            x, y, w, h = bbox
            img_w, img_h = img.size

            x = max(0, int(x))
            y = max(0, int(y))
            w = min(int(w), img_w - x)
            h = min(int(h), img_h - y)

            if w <= 1 or h <= 1:
                raise ValueError("Invalid crop")

            crop = img.crop((x, y, x + w, y + h))
            if self.transform:
                crop = self.transform(crop)
            return crop, label
        except Exception:
            # Single retry with a neighboring sample
            return self.__getitem__((index + 1) % len(self))


# ============================================================
# 2. Training Utilities
# ============================================================
class ModelEMA:
    """Exponential Moving Average of model weights (and BN buffers)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for param in self.ema.parameters():
            param.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for ema_param, param in zip(self.ema.parameters(), model.parameters()):
                ema_param.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
            # Keep BatchNorm running stats in sync with the online model.
            for ema_buf, buf in zip(self.ema.buffers(), model.buffers()):
                if ema_buf.shape == buf.shape:
                    ema_buf.copy_(buf.detach())

    def state_dict(self):
        return self.ema.state_dict()


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """Returns mixed inputs, pairs of targets, and lambda."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def rand_bbox(size, lam):
    """Generate a random bounding box for CutMix."""
    _, _, H, W = size
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation and return mixed inputs and target pairs."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    y_a, y_b = y, y[index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to exactly match the pixel ratio.
    lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in dataloader:
        inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast("cuda"):
            outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total if total > 0 else 0.0


def get_transforms(phase: str, img_size: int):
    """Return train/val transforms with aggressive regularization."""
    # ImageNet normalization for natural images
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if phase == "train_primary":
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.25)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(10),
            transforms.RandAugment(num_ops=2, magnitude=7),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(*stats),
            transforms.RandomErasing(p=RANDOM_ERASING_PROB),
        ])

    if phase == "train_refine":
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.2)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(*stats),
        ])

    # val
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.25)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])


# ============================================================
# 3. Model Setup
# ============================================================
def build_model(device, num_classes=NUM_CLASSES, weights_path=None,
                dropout=DROPOUT, drop_path_rate=DROP_PATH_RATE):
    """Build AvianRaptorNet-Fast, optionally load pretrained weights / head."""
    model = AvianRaptorNet_Fast(
        num_classes=num_classes, dropout=dropout, drop_path_rate=drop_path_rate
    ).to(device)
    needs_head_swap = True

    if weights_path:
        if os.path.exists(weights_path):
            print(f"[INFO] Loading pretrained weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)

            # Detect whether the checkpoint matches the target number of classes.
            head_key = "classifier_head.4.weight"
            checkpoint_classes = state_dict[head_key].shape[0] if head_key in state_dict else num_classes

            if checkpoint_classes == num_classes:
                # Same-class checkpoint: load everything as-is, keep the trained head.
                model.load_state_dict(state_dict, strict=True)
                needs_head_swap = False
            elif checkpoint_classes == 100 and num_classes != 100:
                # CIFAR-100 pretrained backbone -> transplant backbone only.
                pretrained_model = AvianRaptorNet_Fast(
                    num_classes=100, dropout=dropout, drop_path_rate=drop_path_rate
                ).to(device)
                pretrained_model.load_state_dict(state_dict, strict=True)
                model_state = model.state_dict()
                pretrained_state = pretrained_model.state_dict()
                for key in model_state.keys():
                    if "classifier_head.4" not in key:
                        model_state[key] = pretrained_state[key]
                model.load_state_dict(model_state, strict=True)
                del pretrained_model
            else:
                # Fallback: load everything that matches, ignore mismatched head.
                model.load_state_dict(state_dict, strict=False)
        else:
            print(f"[WARN] Pretrained weights not found at {weights_path}; training from scratch.")

    if needs_head_swap:
        # Replace classifier head when loading a mismatched pretrained backbone.
        in_features = model.classifier_head[4].in_features
        model.classifier_head[4] = nn.Linear(in_features, num_classes).to(device)
    return model.to(memory_format=torch.channels_last)


# ============================================================
# 4. Training Phases
# ============================================================
def build_scheduler(optimizer, epochs, warmup_epochs, steps_per_epoch):
    """Linear warmup followed by cosine annealing."""
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_primary(model, train_loader, val_loader, device, ema_decay=EMA_DECAY, grad_clip=GRAD_CLIP_NORM):
    print("\n========== PRIMARY FINE-TUNING STAGE ==========")
    criterion = nn.CrossEntropyLoss(label_smoothing=PRIMARY_LABEL_SMOOTHING)
    optimizer = optim.SGD(
        model.parameters(),
        lr=PRIMARY_LR,
        momentum=PRIMARY_MOMENTUM,
        weight_decay=PRIMARY_WD,
        nesterov=True,
    )
    scheduler = build_scheduler(
        optimizer,
        epochs=PRIMARY_EPOCHS,
        warmup_epochs=WARMUP_EPOCHS,
        steps_per_epoch=len(train_loader),
    )
    scaler = GradScaler("cuda")
    ema = ModelEMA(model, decay=ema_decay)

    best_acc = 0.0
    best_ema_acc = 0.0
    start = time.time()

    for epoch in range(PRIMARY_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Primary Epoch {epoch+1}/{PRIMARY_EPOCHS}")
        for inputs, targets in pbar:
            inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Prefer Mixup (70%) over CutMix (30%) for this small model.
            if np.random.rand() < 0.7:
                inputs, targets_a, targets_b, lam = mixup_data(
                    inputs, targets, alpha=MIXUP_ALPHA, device=device
                )
            else:
                inputs, targets_a, targets_b, lam = cutmix_data(
                    inputs, targets, alpha=CUTMIX_ALPHA
                )

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            scheduler.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (
                lam * predicted.eq(targets_a).sum().float()
                + (1 - lam) * predicted.eq(targets_b).sum().float()
            ).item()

            pbar.set_postfix({
                "Loss": f"{running_loss / (total / BATCH_SIZE + 1e-6):.3f}",
                "TrainAcc": f"{100.0 * correct / total:.1f}%",
                "LR": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        val_acc = evaluate(model, val_loader, device)
        ema_val_acc = evaluate(ema.ema, val_loader, device)
        print(
            f"Epoch {epoch+1}/{PRIMARY_EPOCHS} | "
            f"Train Acc: {100.0 * correct / total:.2f}% | "
            f"Val Acc: {val_acc:.2f}% | EMA Val Acc: {ema_val_acc:.2f}% | "
            f"Best raw: {best_acc:.2f}% | Best EMA: {best_ema_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), PRIMARY_SAVE)
            print(f"  --> New best raw validation accuracy: {best_acc:.2f}%. Saved {PRIMARY_SAVE}")

        if ema_val_acc > best_ema_acc:
            best_ema_acc = ema_val_acc
            torch.save(ema.state_dict(), EMA_PRIMARY_SAVE)
            print(f"  --> New best EMA validation accuracy: {best_ema_acc:.2f}%. Saved {EMA_PRIMARY_SAVE}")

    elapsed = time.time() - start
    print(
        f"[INFO] Primary stage finished in {elapsed / 3600:.2f}h. "
        f"Best raw val acc: {best_acc:.2f}%, Best EMA: {best_ema_acc:.2f}%"
    )
    return best_acc, best_ema_acc


def train_refinement(model, train_loader, val_loader, device, baseline_acc,
                     ema_decay=EMA_DECAY, grad_clip=GRAD_CLIP_NORM):
    print("\n========== ULTRA-LOW LR REFINEMENT STAGE ==========")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=REFINE_LR, weight_decay=REFINE_WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=REFINE_EPOCHS)
    scaler = GradScaler("cuda")
    ema = ModelEMA(model, decay=ema_decay)

    best_acc = baseline_acc
    best_ema_acc = baseline_acc
    best_state = None
    best_ema_state = None
    start = time.time()

    for epoch in range(REFINE_EPOCHS):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Refine Epoch {epoch+1}/{REFINE_EPOCHS}")
        for inputs, targets in pbar:
            inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            running_loss += loss.item()
            pbar.set_postfix({
                "Loss": f"{running_loss / (pbar.n + 1):.4f}",
                "LR": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        scheduler.step()
        val_acc = evaluate(model, val_loader, device)
        ema_val_acc = evaluate(ema.ema, val_loader, device)
        print(
            f"Epoch {epoch+1}/{REFINE_EPOCHS} | "
            f"Val Acc: {val_acc:.2f}% | EMA Val Acc: {ema_val_acc:.2f}% | "
            f"Best raw: {best_acc:.2f}% | Best EMA: {best_ema_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
            print(f"  --> New best raw accuracy: {best_acc:.2f}%")

        if ema_val_acc > best_ema_acc:
            best_ema_acc = ema_val_acc
            best_ema_state = ema.state_dict()
            print(f"  --> New best EMA accuracy: {best_ema_acc:.2f}%")

    # Persist best refined states. Fall back to primary checkpoints if refinement did not improve.
    if best_state is None:
        print(f"[INFO] Refinement did not improve over raw baseline {baseline_acc:.2f}%; "
              f"keeping primary best as {REFINED_SAVE}")
        best_state = torch.load(PRIMARY_SAVE, map_location="cpu", weights_only=True)
    if best_ema_state is None:
        fallback_ema = EMA_PRIMARY_SAVE if os.path.exists(EMA_PRIMARY_SAVE) else PRIMARY_SAVE
        print(f"[INFO] Refinement did not improve over EMA baseline; "
              f"keeping primary EMA best as {EMA_REFINED_SAVE}")
        best_ema_state = torch.load(fallback_ema, map_location="cpu", weights_only=True)

    torch.save(best_state, REFINED_SAVE)
    torch.save(best_ema_state, EMA_REFINED_SAVE)
    print(f"[INFO] Best refined checkpoint saved to {REFINED_SAVE} (raw {best_acc:.2f}%)")
    print(f"[INFO] Best refined EMA checkpoint saved to {EMA_REFINED_SAVE} (EMA {best_ema_acc:.2f}%)")

    elapsed = time.time() - start
    print(f"[INFO] Refinement stage finished in {elapsed / 3600:.2f}h. "
          f"Best raw val acc: {best_acc:.2f}%, Best EMA: {best_ema_acc:.2f}%")
    return best_acc, best_ema_acc


# ============================================================
# 5. Main
# ============================================================
def main():
    # These globals are overridden from command-line arguments below.
    global PRIMARY_EPOCHS, REFINE_EPOCHS, BATCH_SIZE, NUM_WORKERS, IMG_SIZE
    global DROPOUT, DROP_PATH_RATE, PRIMARY_LR, PRIMARY_WD, REFINE_LR, REFINE_WD
    global MIXUP_ALPHA, CUTMIX_ALPHA, PRIMARY_LABEL_SMOOTHING, GRAD_CLIP_NORM, EMA_DECAY, SEED
    global WARMUP_EPOCHS

    parser = argparse.ArgumentParser(description="COCO fine-tuning for AvianRaptorNet")
    parser.add_argument("--target-img-count", type=int, default=20000,
                        help="Number of COCO images to scan for object crops")
    parser.add_argument("--min-area", type=int, default=1024,
                        help="Minimum object area for a crop")
    parser.add_argument("--primary-epochs", type=int, default=PRIMARY_EPOCHS,
                        help="Epochs for primary fine-tuning stage")
    parser.add_argument("--refine-epochs", type=int, default=REFINE_EPOCHS,
                        help="Epochs for ultra-low LR refinement stage")
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS,
                        help="Linear warmup epochs for primary stage")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size for both training and validation")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="DataLoader worker processes")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE,
                        help="Input resolution (default 128; try 160 with smaller batch)")
    parser.add_argument("--dropout", type=float, default=DROPOUT,
                        help="Classifier dropout probability")
    parser.add_argument("--drop-path-rate", type=float, default=DROP_PATH_RATE,
                        help="Stochastic depth drop-path rate")
    parser.add_argument("--primary-lr", type=float, default=PRIMARY_LR,
                        help="Primary stage peak learning rate")
    parser.add_argument("--primary-wd", type=float, default=PRIMARY_WD,
                        help="Primary stage weight decay")
    parser.add_argument("--refine-lr", type=float, default=REFINE_LR,
                        help="Refinement stage learning rate")
    parser.add_argument("--refine-wd", type=float, default=REFINE_WD,
                        help="Refinement stage weight decay")
    parser.add_argument("--mixup-alpha", type=float, default=MIXUP_ALPHA,
                        help="Mixup alpha (0 to disable)")
    parser.add_argument("--cutmix-alpha", type=float, default=CUTMIX_ALPHA,
                        help="CutMix alpha (0 to disable)")
    parser.add_argument("--label-smoothing", type=float, default=PRIMARY_LABEL_SMOOTHING,
                        help="Label smoothing for primary stage")
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP_NORM,
                        help="Gradient clipping max norm (0 to disable)")
    parser.add_argument("--ema-decay", type=float, default=EMA_DECAY,
                        help="EMA decay coefficient")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed")
    parser.add_argument("--skip-primary", action="store_true",
                        help="Skip primary training and refine from an existing "
                             f"{PRIMARY_SAVE} checkpoint")
    args = parser.parse_args()

    PRIMARY_EPOCHS = args.primary_epochs
    REFINE_EPOCHS = args.refine_epochs
    WARMUP_EPOCHS = args.warmup_epochs
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    IMG_SIZE = args.img_size
    DROPOUT = args.dropout
    DROP_PATH_RATE = args.drop_path_rate
    PRIMARY_LR = args.primary_lr
    PRIMARY_WD = args.primary_wd
    REFINE_LR = args.refine_lr
    REFINE_WD = args.refine_wd
    MIXUP_ALPHA = args.mixup_alpha
    CUTMIX_ALPHA = args.cutmix_alpha
    PRIMARY_LABEL_SMOOTHING = args.label_smoothing
    GRAD_CLIP_NORM = args.grad_clip
    EMA_DECAY = args.ema_decay
    SEED = args.seed

    set_seed(SEED)

    print(
        f"[INFO] Effective config: BATCH_SIZE={BATCH_SIZE}, NUM_WORKERS={NUM_WORKERS}, "
        f"IMG_SIZE={IMG_SIZE}, PRIMARY_EPOCHS={PRIMARY_EPOCHS}, REFINE_EPOCHS={REFINE_EPOCHS}, "
        f"WARMUP_EPOCHS={WARMUP_EPOCHS}, DROPOUT={DROPOUT}, DROP_PATH_RATE={DROP_PATH_RATE}, "
        f"MIXUP={MIXUP_ALPHA}, CUTMIX={CUTMIX_ALPHA}, LABEL_SMOOTHING={PRIMARY_LABEL_SMOOTHING}, "
        f"GRAD_CLIP={GRAD_CLIP_NORM}, EMA_DECAY={EMA_DECAY}, SEED={SEED}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    # Build or load crop list
    all_crops = get_or_build_crops(target_img_count=args.target_img_count, min_area=args.min_area)
    if len(all_crops) == 0:
        raise RuntimeError("No valid COCO crops found. Check data directory.")

    # Train/val split
    random.shuffle(all_crops)
    split_idx = int(len(all_crops) * (1 - VAL_RATIO))
    train_crops = all_crops[:split_idx]
    val_crops = all_crops[split_idx:]
    print(f"[INFO] Train crops: {len(train_crops)} | Val crops: {len(val_crops)}")

    # Datasets & loaders
    train_dataset_primary = CocoCropDataset(train_crops, transform=get_transforms("train_primary", IMG_SIZE))
    train_dataset_refine = CocoCropDataset(train_crops, transform=get_transforms("train_refine", IMG_SIZE))
    val_dataset = CocoCropDataset(val_crops, transform=get_transforms("val", IMG_SIZE))

    train_loader_primary = DataLoader(
        train_dataset_primary,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    n_params = sum(p.numel() for p in AvianRaptorNet_Fast(
        num_classes=NUM_CLASSES, dropout=DROPOUT, drop_path_rate=DROP_PATH_RATE
    ).parameters())
    print(f"[INFO] Model parameters: {n_params / 1e6:.2f}M")

    if args.skip_primary:
        if not os.path.exists(PRIMARY_SAVE):
            raise FileNotFoundError(
                f"--skip-primary requires an existing {PRIMARY_SAVE} checkpoint."
            )
        print(f"[INFO] Skipping primary stage; loading existing {PRIMARY_SAVE}")
        primary_best = None
        primary_ema_best = None
    else:
        # Primary stage
        model = build_model(
            device,
            num_classes=NUM_CLASSES,
            weights_path=PRETRAINED_WEIGHTS,
            dropout=DROPOUT,
            drop_path_rate=DROP_PATH_RATE,
        )
        baseline_acc = evaluate(model, val_loader, device)
        print(f"[INFO] Post-head-swap random-initialized baseline val acc: {baseline_acc:.2f}%")
        torch.cuda.empty_cache()

        primary_best, primary_ema_best = train_primary(
            model, train_loader_primary, val_loader, device,
            ema_decay=EMA_DECAY, grad_clip=GRAD_CLIP_NORM
        )
        del train_loader_primary
        torch.cuda.empty_cache()

    # Refinement stage: load primary best and create a fresh loader
    print(f"\n[INFO] Loading primary best checkpoint for refinement: {PRIMARY_SAVE}")

    train_loader_refine = DataLoader(
        train_dataset_refine,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )
    model = build_model(
        device,
        num_classes=NUM_CLASSES,
        weights_path=PRIMARY_SAVE,
        dropout=DROPOUT,
        drop_path_rate=DROP_PATH_RATE,
    )
    # Recover primary best accuracy from the freshly loaded checkpoint.
    primary_best_loaded = evaluate(model, val_loader, device)
    print(f"[INFO] Loaded primary checkpoint val acc: {primary_best_loaded:.2f}%")
    if primary_best is None:
        primary_best = primary_best_loaded
        primary_ema_best = primary_best_loaded

    refined_best, refined_ema_best = train_refinement(
        model, train_loader_refine, val_loader, device,
        baseline_acc=max(primary_best, primary_ema_best),
        ema_decay=EMA_DECAY, grad_clip=GRAD_CLIP_NORM
    )

    print("\n========== TRAINING SUMMARY ==========")
    print(f"Primary best val accuracy:  {primary_best:.2f}%  -> {PRIMARY_SAVE}")
    print(f"Primary best EMA accuracy:  {primary_ema_best:.2f}%  -> {EMA_PRIMARY_SAVE}")
    print(f"Refined best val accuracy:  {refined_best:.2f}%  -> {REFINED_SAVE}")
    print(f"Refined best EMA accuracy:  {refined_ema_best:.2f}%  -> {EMA_REFINED_SAVE}")
    print("======================================")


if __name__ == "__main__":
    main()
