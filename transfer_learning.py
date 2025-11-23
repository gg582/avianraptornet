#!/usr/bin/env python3
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.amp import autocast, GradScaler

# Import the model definition
from avian_model import AvianRaptorNet_Fast

def main():
    parser = argparse.ArgumentParser(description='Train AvianRaptorNet on Custom Dataset')
    parser.add_argument('--data-dir', type=str, required=True, help='Root directory containing train/ and val/ folders')
    parser.add_argument('--weights', type=str, required=True, help='Path to CIFAR-100 pretrained weights')
    parser.add_argument('--img-size', type=int, default=64, help='Resize images to this dimension')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--freeze-body', action='store_true', help='Freeze feature extractor layers')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Enable TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Data Preparation (ImageFolder)
    # Using CIFAR-100 stats for normalization as we use pretrained weights
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    
    transform_train = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')

    print(f"Loading data from {args.data_dir}...")
    trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
    valset = torchvision.datasets.ImageFolder(val_dir, transform=transform_val)
    
    num_classes = len(trainset.classes)
    print(f"Detected {num_classes} classes.")

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize Model
    print("Loading pretrained model...")
    # Initialize with 100 classes first to load weights matching the checkpoint
    model = AvianRaptorNet_Fast(num_classes=100).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    # Replace the classification head
    print(f"Reinitializing classifier for {num_classes} classes...")
    # The classifier head structure is: Conv1x1 -> BioMish -> Dropout -> Flatten -> Linear
    # We need to replace the last Linear layer (index 4)
    in_features = model.classifier_head[4].in_features
    model.classifier_head[4] = nn.Linear(in_features, num_classes).to(device)

    # Freeze body if requested
    if args.freeze_body:
        print("Freezing feature extractor layers...")
        for name, param in model.named_parameters():
            if "classifier_head" not in name:
                param.requires_grad = False

    model = model.to(memory_format=torch.channels_last)

    # Optimizer & Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler('cuda')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("Starting training on custom dataset...")
    best_acc = 0.0
    
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
            for inputs, targets in valloader:
                inputs = inputs.to(device, memory_format=torch.channels_last)
                targets = targets.to(device)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = 100. * correct / total
        scheduler.step()
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "avian_raptor_custom.pth")
            
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss/len(trainloader):.4f} | Val Acc: {acc:.2f}%")

if __name__ == '__main__':
    main()
