#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torch.amp import autocast, GradScaler
import sys
import os

# Configure PyTorch allocator to prevent VRAM fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 0. Import Huge Model Definition
try:
    from cifar100_huge import AvianMidbrainNet_Huge
except ImportError:
    print("[ERROR] 'cifar100_huge.py' not found.")
    sys.exit(1)

# 1. Model Preparation
def prepare_model(pretrained_path, num_classes, device):
    print(f"[INFO] Loading model weights from: {pretrained_path}")
    # Initialize model architecture
    model = AvianMidbrainNet_Huge(num_classes=100, cortical_dropout_rate=0.2)
    
    # Load pretrained weights
    try:
        checkpoint = torch.load(pretrained_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # Remove DataParallel 'module.' prefix if present
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print("[INFO] Weights loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load weights: {e}")
        sys.exit(1)

    print("[INFO] Configuration: Full fine-tuning (Backbone unfrozen).")

    # Replace Classifier Head for Target Task
    # Input dimension: 1536 (1024 Tecto + 512 Wulst)
    fused_dim = 1536
    model.classifier = nn.Sequential(
        nn.Linear(fused_dim, 1024),
        nn.GELU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.GELU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    
    # Initialize new head weights
    for m in model.classifier.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)
            
    return model.to(device)

# 2. Training Logic with Gradient Accumulation
def train_epoch(model, loader, criterion, opt, scaler, device, accum_steps):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    log_interval = len(loader) // 5
    
    # Ensure optimizer is zeroed at the start
    opt.zero_grad()
    
    for i, (img, label) in enumerate(loader):
        img, label = img.to(device), label.to(device)
        
        # Forward Pass with Mixed Precision
        with autocast('cuda'):
            out = model(img)
            loss = criterion(out, label)
            # Normalize loss for gradient accumulation
            loss = loss / accum_steps 
            
        # Backward Pass
        scaler.scale(loss).backward()
        
        # Update weights every 'accum_steps' iterations
        if (i + 1) % accum_steps == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
        
        # Metrics calculation (rescale loss for display)
        loss_val = loss.item() * accum_steps
        total_loss += loss_val * img.size(0)
        _, pred = out.max(1)
        total += label.size(0)
        correct += pred.eq(label).sum().item()
        
        if i % log_interval == 0 and i > 0:
            print(f"   > Batch {i}/{len(loader)} | Loss: {loss_val:.4f} | Acc: {100.*correct/total:.2f}%")
            
    return total_loss/total, correct/total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for img, label in loader:
            img, label = img.to(device), label.to(device)
            out = model(img)
            loss = criterion(out, label)
            total_loss += loss.item() * img.size(0)
            _, pred = out.max(1)
            total += label.size(0)
            correct += pred.eq(label).sum().item()
            
    return total_loss/total, correct/total

# 3. Main Execution
def main():
    parser = argparse.ArgumentParser(description="Caltech-256 Full Fine-tuning with Gradient Accumulation")
    parser.add_argument("--pretrained-path", type=str, required=True, help="Path to source .pth file")
    parser.add_argument("--batch-size", type=int, default=8, help="Micro-batch size per step")
    parser.add_argument("--accum-steps", type=int, default=4, help="Number of steps to accumulate gradients")
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--output", type=str, default="avian_caltech256_best.pth", help="Output checkpoint path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Calculate and display effective batch size
    eff_batch = args.batch_size * args.accum_steps
    print(f"[INIT] Target: Caltech-256 Object Recognition")
    print(f"[INFO] Effective Batch Size: {eff_batch} (Micro-batch: {args.batch_size}, Accumulation: {args.accum_steps})")

    # Input Resolution
    size = 224 
    
    # Data Augmentation
    transform_train = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")), 
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    transform_val = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")), 
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Dataset Loading
    print("[DATA] Downloading Caltech-256...")
    try:
        full_dataset = torchvision.datasets.Caltech256(root='./data', download=True, transform=transform_train)
    except RuntimeError:
        print("[ERROR] Download failed. Check internet connection or manual download.")
        sys.exit(1)

    # Train/Val Split (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Caltech-256 Classes: 256 categories + 1 clutter = 257
    num_classes = 257 
    print(f"[DATA] Train: {len(train_dataset)}, Val: {len(val_dataset)}, Classes: {num_classes}")

    # Model Setup
    model = prepare_model(args.pretrained_path, num_classes, device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler('cuda')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    print("[START] Training Process Started...")
    
    for epoch in range(args.epochs):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, args.accum_steps)
        v_loss, v_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), args.output)
            print(f"Ep [{epoch+1}/{args.epochs}] Train: {t_acc:.4f} | Val: {v_acc:.4f} [NEW BEST]")
        else:
            print(f"Ep [{epoch+1}/{args.epochs}] Train: {t_acc:.4f} | Val: {v_acc:.4f}")

    print(f"[DONE] Best Validation Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
