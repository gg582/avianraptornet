import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
from shutil import move

# --- Configuration ---
SOURCE_DIR = "./dataset/morus_alba"
CLEANED_DIR = "./dataset/morus_alba_cleaned"
GARBAGE_DIR = "./dataset/morus_alba_garbage"

# Create directories if they do not exist
for folder in [CLEANED_DIR, GARBAGE_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# --- Model Setup ---
# Using 'weights' instead of 'pretrained' to resolve DeprecationWarning
weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.eval()

# Image transformation matching ResNet50's expected input
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def is_plant_related(image_path):
    """
    Evaluates if an image contains plant-related content using Top-K predictions.
    """
    try:
        # Load and convert image to RGB
        img = Image.open(image_path).convert('RGB')
        input_tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
        
        # Convert output to probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Check Top-5 predictions to be less restrictive
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        
        # ImageNet plant-related indices (Broad range covering trees, leaves, and fruits)
        # Typically 936 to 970 includes many botanical categories
        plant_indices = set(range(910, 995)) 
        
        for i in range(5):
            if top5_catid[i].item() in plant_indices:
                return True
        return False
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

# --- Main Execution ---
def main():
    print("Starting data cleansing with updated ResNet50 weights...")
    
    if not os.path.exists(SOURCE_DIR):
        print(f"Source directory {SOURCE_DIR} not found.")
        return

    files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in files:
        file_path = os.path.join(SOURCE_DIR, filename)
        
        # Perform classification-based filtering
        if is_plant_related(file_path):
            move(file_path, os.path.join(CLEANED_DIR, filename))
            print(f"[KEEP] {filename}")
        else:
            move(file_path, os.path.join(GARBAGE_DIR, filename))
            print(f"[REJECT] {filename}")

    print(f"Cleansing complete. Check {CLEANED_DIR} for results.")

if __name__ == "__main__":
    main()
