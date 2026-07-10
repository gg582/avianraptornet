import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from core.avian_model import AvianRaptorNet_Fast

# --- Configuration ---
DATA_DIR = "./dataset/teacup_mobrew_cleaned"
MODEL_SAVE_PATH = "teacup_avian_raptor.pth"
PRETRAINED_WEIGHTS = "avian_raptor_fast_best.pth"
BATCH_SIZE = 2
EPOCHS = 200
LEARNING_RATE = 2e-4
NUM_CLASSES = 3 # shape_structure, detail_texture, style_culture

# --- Data Preparation ---
# ... (transforms and data loading remain the same)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load dataset and split into train/val
full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms['train'])
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Apply validation transforms to val_dataset
val_dataset.dataset.transform = data_transforms['val']

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
class_names = full_dataset.classes

print(f"Dataset loaded: {len(full_dataset)} images, Classes: {class_names}")

# --- Model Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AvianRaptorNet_Fast(num_classes=100).to(device)

if os.path.exists(PRETRAINED_WEIGHTS):
    print(f"Loading pre-trained weights from {PRETRAINED_WEIGHTS}...")
    state_dict = torch.load(PRETRAINED_WEIGHTS, map_location=device)
    model.load_state_dict(state_dict)
else:
    print("Warning: Pre-trained weights not found. Starting from scratch.")

# Replace classifier head for the new classes
# Original classifier_head[4] is nn.Linear(768, 100)
num_ftrs = model.classifier_head[4].in_features
model.classifier_head[4] = nn.Linear(num_ftrs, len(class_names)).to(device)

model = model.to(memory_format=torch.channels_last)

# Setup Optimizer & Scaler for AMP
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = GradScaler('cuda')

# --- Training Loop ---
print(f"Starting fine-tuning on {device} (RTX 3070 optimized)...")

best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device, memory_format=torch.channels_last), labels.to(device)

        optimizer.zero_grad()
        with autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
    
    scheduler.step()
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device, memory_format=torch.channels_last), labels.to(device)
            with autocast('cuda'):
                outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_acc = 100. * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")
    
    if val_acc > best_acc:
        best_acc = val_acc
        print(f"  --> Saving best model with {best_acc:.2f}% accuracy...")
        torch.save({
            'model_state': model.state_dict(),
            'classes': class_names
        }, MODEL_SAVE_PATH)

print(f"Fine-tuning complete. Best Val Acc: {best_acc:.2f}%")
