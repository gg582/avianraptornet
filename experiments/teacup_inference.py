import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from core.avian_model import AvianRaptorNet_Fast

# --- Configuration ---
MODEL_PATH = "teacup_avian_raptor.pth"

# Load checkpoint
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file {MODEL_PATH} not found. Please run fine_tune_teacup.py first.")
    sys.exit(1)

checkpoint = torch.load(MODEL_PATH)
class_names = checkpoint['classes']

# --- Model Setup ---
# Initialize AvianRaptorNet_Fast architecture with the correct number of classes
model = AvianRaptorNet_Fast(num_classes=len(class_names))

# Load the trained weights
model.load_state_dict(checkpoint['model_state'])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# --- Inference Setup ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image_path):
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
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python teacup_inference.py <image_path>")
    else:
        predict(sys.argv[1])
