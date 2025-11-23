import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import argparse
import sys
from ultralytics import YOLO  # pip install ultralytics

# ============================================================
# 1. MODEL DEFINITION (Must match training EXACTLY)
# ============================================================

# JIT-compiled Activation
@torch.jit.script
def biomish_activation(x):
    return x * torch.tanh(F.softplus(x))

class BioMish(nn.Module):
    def forward(self, x): return biomish_activation(x)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False), BioMish(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False), nn.Sigmoid()
        )
    def forward(self, x): return x * self.fc(self.avg_pool(x))

class RaptorFovealLite(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 2
        self.central = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1, bias=False), nn.BatchNorm2d(mid), BioMish()
        )
        self.peripheral = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=2, dilation=2, groups=4, bias=False),
            nn.BatchNorm2d(mid), BioMish()
        )
        self.fusion = nn.Conv2d(mid * 2, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        c, p = self.central(x), self.peripheral(x)
        return self.bn(self.fusion(torch.cat([c, p], dim=1)))

class FeatherBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.use_res_connect = (stride == 1 and in_ch == out_ch)
        exp_size = in_ch * 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, exp_size, 1, bias=False), nn.BatchNorm2d(exp_size), BioMish(),
            nn.Conv2d(exp_size, exp_size, 3, stride=stride, padding=1, groups=exp_size, bias=False),
            nn.BatchNorm2d(exp_size), BioMish(), ChannelAttention(exp_size),
            nn.Conv2d(exp_size, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)
        )
    def forward(self, x): return x + self.conv(x) if self.use_res_connect else self.conv(x)

class AvianRaptorNet_Fast(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.retina = nn.Sequential(nn.Conv2d(3, 48, 3, 1, 1, bias=False), nn.BatchNorm2d(48), BioMish())
        self.raptor_eye = RaptorFovealLite(48, 96)
        self.body = nn.Sequential(
            FeatherBlock(96, 128, 2), FeatherBlock(128, 128, 1), FeatherBlock(128, 256, 2),
            FeatherBlock(256, 256, 1), FeatherBlock(256, 256, 1), FeatherBlock(256, 512, 2),
            FeatherBlock(512, 512, 1),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # [IMPORTANT] This structure must match the saved weights
        self.classifier_head = nn.Sequential(
            nn.Conv2d(512, 768, 1, bias=False), 
            BioMish(), 
            nn.Dropout(0.2), 
            nn.Flatten(), 
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        x = self.retina(x)
        x = self.raptor_eye(x)
        x = self.body(x)
        x = self.global_pool(x)
        x = self.classifier_head(x)
        return x

# ============================================================
# 2. CONFIGURATION & LABELS
# ============================================================
WEIGHTS_PATH = 'avian_raptor_fast_best.pth' # Ensure this file exists
NUM_CLASSES = 100 # CIFAR-100 Standard
IMG_SIZE = 32     # CIFAR Standard

# CIFAR-100 English Labels
CIFAR100_LABELS = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 
    'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 
    'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 
    'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 
    'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 
    'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 
    'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# ============================================================
# 3. MAIN INFERENCE FUNCTION
# ============================================================
def run_inference(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Loading CIFAR-100 Model on {device}...")

    # --- A. Load AvianRaptorNet ---
    model = AvianRaptorNet_Fast(num_classes=NUM_CLASSES).to(device)
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        print("✅ CIFAR Weights Loaded Successfully!")
    except FileNotFoundError:
        print(f"❌ Error: Weights file '{WEIGHTS_PATH}' not found.")
        sys.exit(1)
    except RuntimeError as e:
        print(f"❌ Model mismatch error: {e}")
        print("Ensure the model definition in this script exactly matches the training script.")
        sys.exit(1)
    
    model.to(memory_format=torch.channels_last)
    model.eval()

    # --- B. Load YOLO Detector ---
    print("Loading Detector (YOLOv8 Nano)...")
    # Suppress YOLO's loud output
    detector = YOLO('yolov8n.pt')

    # --- C. Preprocessing ---
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)), # 32x32 for CIFAR
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    # --- D. Load Image ---
    print(f"Processing image: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Error: Could not open image '{image_path}'. Check the path.")
        sys.exit(1)

    # 1. Detect Objects (YOLO)
    # Running at higher resolution for better detection in static images
    results = detector(frame, conf=0.25, iou=0.45, verbose=False)

    detections_found = False
    for result in results:
        boxes = result.boxes
        for box in boxes:
            detections_found = True
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Clip to frame bounds
            h_img, w_img, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)

            # 2. Crop Object
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0: continue

            # 3. Classify with AvianRaptorNet
            # Convert BGR (OpenCV default) -> RGB (Model expects RGB)
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            input_tensor = preprocess(roi_rgb).unsqueeze(0).to(device)
            input_tensor = input_tensor.to(memory_format=torch.channels_last)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    output = model(input_tensor)
                    probs = F.softmax(output, dim=1)
                    conf, idx = probs.max(1)

            label = CIFAR100_LABELS[idx.item()]
            score = conf.item()

            # 4. Draw Results
            color = (0, 255, 0) # Green
            line_thickness = max(1, int(min(w_img, h_img) / 200))
            
            # Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)
            
            # Label Text
            text = f"{label} ({int(score*100)}%)"
            font_scale = line_thickness / 2.0
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness)
            
            # Text Background
            cv2.rectangle(frame, (x1, y1 - th - baseline), (x1 + tw, y1), color, -1)
            # Text itself
            cv2.putText(frame, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), line_thickness)

    if not detections_found:
        print("No objects detected by YOLO in this image.")

    # Show Result
    window_name = "AvianRaptorNet Static Inference"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Allow resizing
    cv2.imshow(window_name, frame)
    
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Optional: Save output
    output_path = "output_" + image_path
    cv2.imwrite(output_path, frame)
    print(f"Saved result to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AvianRaptorNet Static Image Demo')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image file')
    args = parser.parse_args()
    
    run_inference(args.image)
