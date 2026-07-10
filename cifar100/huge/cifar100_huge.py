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
# Enable TF32 for Ampere GPUs (RTX 30 series) for huge speedup
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# ============================================================
# 1. JIT-Compiled Bio-Components
# ============================================================

@torch.jit.script
def biomish_activation(x):
    """
    JIT-fused BioMish activation.
    Formula: x * tanh(softplus(x))
    """
    return x * torch.tanh(F.softplus(x))

class BioMish(nn.Module):
    def forward(self, x):
        return biomish_activation(x)

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel recalibration.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            BioMish(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# ============================================================
# 2. Raptor Dual-Flow Bottleneck (Core Architecture)
# ============================================================

class RaptorBottleneck(nn.Module):
    """
    Raptor Hybrid Bottleneck Block (ResNet-style).
    
    Structure:
    1. 1x1 Conv (Compression)
    2. Dual-Flow Processing (Parallel):
       - Foveal Path: Standard 3x3 Conv (Detail focus)
       - Peripheral Path: Dilated 3x3 Conv (Context focus, only if stride==1)
    3. 1x1 Conv (Expansion x4)
    4. SE Block (Attention)
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        width = planes 
        
        # 1. Compression
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        
        # 2. Dual Flow
        # Path A: Foveal (Standard)
        self.conv2_fovea = nn.Conv2d(width, width, kernel_size=3, stride=stride, 
                                     padding=1, bias=False)
        self.bn2_fovea = nn.BatchNorm2d(width)
        
        # Path B: Peripheral (Dilated) - Context
        # Only apply dual path when stride is 1 to maintain feature map alignment easily
        self.use_dual = (stride == 1)
        if self.use_dual:
            self.conv2_periph = nn.Conv2d(width, width, kernel_size=3, stride=1, 
                                          padding=2, dilation=2, bias=False)
            self.bn2_periph = nn.BatchNorm2d(width)

        # 3. Expansion
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.act = BioMish()
        self.se = SEBlock(planes * self.expansion)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.act(self.bn1(self.conv1(x)))
        
        # Parallel Execution
        fovea = self.bn2_fovea(self.conv2_fovea(out))
        
        if self.use_dual:
            periph = self.bn2_periph(self.conv2_periph(out))
            # Fuse paths (Additive fusion)
            out = self.act(fovea + periph)
        else:
            out = self.act(fovea)
            
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        
        out += identity
        out = self.act(out)
        return out

# ============================================================
# 3. Avian RaptorNet Medium (ResNet-50 Scale)
# ============================================================

class AvianRaptorNet_Medium(nn.Module):
    """
    Medium Scale Model (~26M Params).
    Target: CIFAR-100 > 80% Accuracy.
    Based on ResNet-50 topology [3, 4, 6, 3] with Raptor Blocks.
    """
    def __init__(self, num_classes=100):
        super().__init__()
        self.in_planes = 64
        
        # Stem (Initial processing)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = BioMish()
        
        # Stages (Body)
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        
        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * RaptorBottleneck.expansion, num_classes)

        # Weight Initialization
        self._initialize_weights()

    def _make_layer(self, planes, blocks, stride):
        layers = []
        layers.append(RaptorBottleneck(self.in_planes, planes, stride))
        self.in_planes = planes * RaptorBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(RaptorBottleneck(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
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
# 5. Training Loop
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='AvianRaptorNet Medium CIFAR-100 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Check for Ampere/TF32
    if torch.cuda.get_device_capability()[0] >= 8:
        print("Ampere GPU detected: TF32 Enabled.")
    
    # --------------------------------------------------------
    # Data Preparation (Heavy Augmentation for High Accuracy)
    # --------------------------------------------------------
    print("Preparing Data...")
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10), # Critical for 80%+
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=8, pin_memory=True, persistent_workers=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, 
                            num_workers=4, pin_memory=True)

    # --------------------------------------------------------
    # Model & Optimizer
    # --------------------------------------------------------
    print("Building AvianRaptorNet Medium...")
    model = AvianRaptorNet_Medium(num_classes=100).to(device)
    
    # VITAL: Channels Last Memory Format for RTX 3070 Ti (NCHW -> NHWC optimization)
    model = model.to(memory_format=torch.channels_last)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {n_params/1e6:.2f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------
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

            # Mixup augmentation (50% chance or always, let's do always for stability)
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
            # Accuracy calc for mixup is approximate during training
            correct += (lam * predicted.eq(targets_a).sum().float() + (1 - lam) * predicted.eq(targets_b).sum().float()).item()

        # Validation
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        acc = 100. * test_correct / test_total
        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'avian_raptor_medium_best.pth')

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Loss: {train_loss/(batch_idx+1):.3f} | "
              f"Train Acc: {100.*correct/total:.2f}% | "
              f"Test Acc: {acc:.2f}% | "
              f"Best: {best_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.4f}")

    total_time = time.time() - start_time
    print(f"Training Finished. Total Time: {total_time/3600:.2f} hours. Best Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
