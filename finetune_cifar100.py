#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.amp import autocast, GradScaler

# Ensure avian_model.py is in the same directory
from avian_model import AvianRaptorNet_Fast

def main():
    parser = argparse.ArgumentParser(description='Safe Refine AvianRaptorNet')
    parser.add_argument('--weights', type=str, required=True, help='Path to best .pth checkpoint')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for refinement')
    parser.add_argument('--lr', type=float, default=1e-5, help='Ultra low LR to preserve weights')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Enable TF32 for Ampere GPUs (RTX 30 series+)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Data Preparation
    # CIFAR-100 Mean/Std
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])
    
    # Use weak augmentation for refinement to preserve existing knowledge
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize Model
    print(f"Loading model from {args.weights}...")
    model = AvianRaptorNet_Fast(num_classes=100).to(device)
    
    # Load weights (Use strict=True to ensure structural integrity)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    
    # Optimize memory format for Tensor Cores
    model = model.to(memory_format=torch.channels_last)

    # [IMPORTANT] Check baseline performance before training
    # This ensures the loaded weights are correct and prevents "catastrophic forgetting" illusion
    print(">>> Checking baseline accuracy before training...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = inputs.to(device, memory_format=torch.channels_last)
            targets = targets.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    baseline_acc = 100. * correct / total
    print(f">>> Baseline Accuracy: {baseline_acc:.2f}%")
    print("    (If this is ~1%, your weights are not loading correctly)")
    
    # Setup Optimizer with very low Learning Rate
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler('cuda')
    
    # Start Refinement Training
    print(f">>> Starting Careful Refinement (LR={args.lr})...")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        for inputs, targets in trainloader:
            inputs = inputs.to(device, memory_format=torch.channels_last)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs = inputs.to(device, memory_format=torch.channels_last)
                targets = targets.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss/len(trainloader):.4f} | Test Acc: {acc:.2f}%")
        
        # Save only if performance improves over the baseline
        if acc > baseline_acc:
            print(f"  --> Accuracy Improved ({baseline_acc:.2f}% -> {acc:.2f}%). Saving model...")
            baseline_acc = acc
            torch.save(model.state_dict(), "avian_raptor_refined_safe.pth")

if __name__ == '__main__':
    main()
