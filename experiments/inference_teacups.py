import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
from core.avian_model import AvianRaptorNet_Fast

# --- Configuration ---
MODEL_PATH = "teacup_avian_raptor.pth"

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found. Please run fine_tune_teacup.py first.")
        sys.exit(1)

    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    class_names = checkpoint['classes']
    
    # Initialize AvianRaptorNet_Fast architecture with the correct number of classes
    model = AvianRaptorNet_Fast(num_classes=len(class_names))
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, class_names, device

# --- Inference Setup ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(model, class_names, device, image_path):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    try:
        img = Image.open(image_path).convert('RGB')
        img_t = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_t)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)
            
        result = class_names[preds[0]]
        print(f"Prediction for [{image_path}]: {result} ({conf.item()*100:.2f}%)")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Teacup Classification")
    parser.add_argument("--images", type=str, help="Comma-separated paths to images")
    args = parser.parse_args()

    model, class_names, device = load_model()

    if args.images:
        image_list = [img.strip() for img in args.images.split(",")]
        for img_path in image_list:
            predict(model, class_names, device, img_path)
    else:
        print("Enter image paths (one per line, empty line to exit):")
        while True:
            try:
                line = input("> ").strip()
                if not line:
                    break
                predict(model, class_names, device, line)
            except EOFError:
                break
