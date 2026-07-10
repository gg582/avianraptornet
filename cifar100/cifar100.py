#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

# ============================================================
# Configuration & Hardware Tuning (RTX 3070 Ti)
# ============================================================
# Enable TF32 for Ampere GPUs (RTX 30 series)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# ============================================================
# 1. JIT-Compiled Bio-Components
# ============================================================

@torch.jit.script
def biomish_activation(x):
    """
    JIT-fused BioMish activation for maximum throughput.
    """
    return x * torch.tanh(F.softplus(x))

class BioMish(nn.Module):
    def forward(self, x):
        return biomish_activation(x)

class ChannelAttention(nn.Module):
    """
    Lightweight Channel Attention using 1x1 Convs (faster than Linear).
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            BioMish(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

# ============================================================
# 2. Lightweight Blocks (GhostNet Style + Raptor)
# ============================================================

class RaptorFovealLite(nn.Module):
    """
    Lightweight Dual-Fovea Block.
    Parallel processing: Detail (3x3) + Context (Dilated)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 2
        
        # Central Fovea (Detail)
        self.central = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            BioMish()
        )
        
        # Peripheral Fovea (Context) - Dilated
        self.peripheral = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=2, dilation=2, groups=4, bias=False), # Groups for speed
            nn.BatchNorm2d(mid),
            BioMish()
        )
        
        # Fusion
        self.fusion = nn.Conv2d(mid * 2, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        c = self.central(x)
        p = self.peripheral(x)
        return self.bn(self.fusion(torch.cat([c, p], dim=1)))

class FeatherBlock(nn.Module):
    """
    Residual Inverted Bottleneck (GhostNet-inspired) for extreme lightweighting.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.use_res_connect = (stride == 1 and in_ch == out_ch)
        
        # Pointwise expansion
        exp_size = in_ch * 2
        self.conv = nn.Sequential(
            # Expansion
            nn.Conv2d(in_ch, exp_size, 1, bias=False),
            nn.BatchNorm2d(exp_size),
            BioMish(),
            
            # Depthwise (Spatial)
            nn.Conv2d(exp_size, exp_size, 3, stride=stride, padding=1, groups=exp_size, bias=False),
            nn.BatchNorm2d(exp_size),
            BioMish(),
            
            # Channel Attention
            ChannelAttention(exp_size),
            
            # Projection (Pointwise Linear)
            nn.Conv2d(exp_size, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# ============================================================
# 3. Avian RaptorNet Fast (~4M Params)
# ============================================================

class AvianRaptorNet_Fast(nn.Module):
    """
    Highly Optimized Architecture (~4M Params).
    Designed for high throughput on GPU while maintaining accuracy.
    """
    def __init__(self, num_classes=100):
        super().__init__()

        # --- 1. Retina (Fast Downsampling) ---
        self.retina = nn.Sequential(
            nn.Conv2d(3, 48, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48),
            BioMish()
        )

        # --- 2. Raptor Vision Stage (Feature Enrichment) ---
        self.raptor_eye = RaptorFovealLite(48, 96)

        # --- 3. Body (Feather Blocks) ---
        self.body = nn.Sequential(
            FeatherBlock(96, 128, stride=2),
            FeatherBlock(128, 128, stride=1),
            FeatherBlock(128, 256, stride=2),
            FeatherBlock(256, 256, stride=1),
            FeatherBlock(256, 256, stride=1),
            FeatherBlock(256, 512, stride=2),
            FeatherBlock(512, 512, stride=1),
        )

        # --- 4. Head ---
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 1x1 Conv replacing large Linear
        self.classifier_head = nn.Sequential(
            nn.Conv2d(512, 768, 1, bias=False),
            BioMish(),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(768, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.retina(x)
        x = self.raptor_eye(x)
        x = self.body(x)
        x = self.global_pool(x)
        x = self.classifier_head(x)
        return x

# ============================================================
# 4. Mixup & Data Utilities
# ============================================================

def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ============================================================
# 5. Training Loop (Optimized)
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='AvianRaptorNet Fast (Lightweight) Training')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate') # Lower LR for lightweight
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
    parser.add_argument('--batch-size', default=256, type=int, help='batch size') # Higher batch size for speed
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.get_device_capability()[0] >= 8:
        print("Ampere GPU detected: TF32 Enabled.")

    # Data Augmentation
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    # Persistent workers for speed
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=8, pin_memory=True, persistent_workers=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=200, shuffle=False, 
                            num_workers=4, pin_memory=True)

    # Model Init
    print("Building AvianRaptorNet Fast (Lightweight)...")
    model = AvianRaptorNet_Fast(num_classes=100).to(device)
    
    # VITAL: Channels Last Memory Format
    model = model.to(memory_format=torch.channels_last)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {n_params/1e6:.2f}M")

    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    # Training
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # Optim: Channels last for inputs
            inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Mixup
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=1.0, device=device)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a).sum().float() + (1 - lam) * predicted.eq(targets_b).sum().float()).item()

        # Validation
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                with autocast():
                    outputs = model(inputs)
                
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        acc = 100. * test_correct / test_total
        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'avian_raptor_fast_best.pth')

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Loss: {train_loss/(batch_idx+1):.3f} | "
              f"Train Acc: {100.*correct/total:.2f}% | "
              f"Test Acc: {acc:.2f}% | "
              f"Best: {best_acc:.2f}%")

    total_time = time.time() - start_time
    print(f"Training Finished. Total Time: {total_time/3600:.2f} hours. Best Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
