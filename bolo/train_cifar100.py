"""Train BOLO-CIFAR100 with standard CIFAR augmentation and mixed precision."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from bolo.cifar100_model import BoloCIFAR100

ROOT = Path(__file__).resolve().parent.parent
MEAN, STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device, non_blocking=True))
            correct += (logits.argmax(1).cpu() == targets).sum().item()
            total += targets.numel()
    return correct / total


def main():
    parser = argparse.ArgumentParser(description="Train BOLO-CIFAR100")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3, help="fine-tuning learning rate")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default="bolo_cifar100_best.pth")
    parser.add_argument("--data", default=str(ROOT / "data"))
    parser.add_argument(
        "--init-weights", default=str(ROOT.parent / "aviannet" / "avian_raptor_fast_best.pth"),
        help="AvianRaptorNet-Fast checkpoint to reform; use an empty string for scratch training",
    )
    parser.add_argument("--train-backbone", action="store_true", help="also fine-tune original AvianRaptorNet weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9), transforms.ToTensor(),
        transforms.Normalize(MEAN, STD), transforms.RandomErasing(p=0.15),
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_set = datasets.CIFAR100(args.data, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR100(args.data, train=False, download=True, transform=test_tf)
    loader_args = {"num_workers": args.workers, "pin_memory": device.type == "cuda"}
    if args.workers:
        loader_args["persistent_workers"] = True
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, **loader_args)
    test_loader = DataLoader(test_set, 512, shuffle=False, **loader_args)

    model = BoloCIFAR100().to(device, memory_format=torch.channels_last)
    if args.init_weights:
        model.load_aviannet_weights(args.init_weights)
        print(f"loaded AvianRaptorNet-Fast weights: {args.init_weights}")
    if not args.train_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
        print("backbone frozen; training only the inserted BOLO motion layers")
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = evaluate(model, test_loader, device)
    torch.save(model.state_dict(), args.output)
    print(f"device={device}; parameters={sum(p.numel() for p in model.parameters()):,}; baseline top1={best:.2%}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()
        scheduler.step()
        accuracy = evaluate(model, test_loader, device)
        print(f"epoch {epoch:03d}/{args.epochs}: loss={loss_sum / len(train_loader):.4f}, top1={accuracy:.2%}")
        if accuracy > best:
            best = accuracy
            torch.save(model.state_dict(), args.output)
            print(f"saved {args.output} (top1={best:.2%})")


if __name__ == "__main__":
    main()
