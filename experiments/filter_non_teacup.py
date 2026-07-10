import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
from shutil import move

# --- Configuration ---
SOURCE_BASE_DIR = "./dataset/teacup_mobrew"
CLEANED_BASE_DIR = "./dataset/teacup_mobrew_cleaned"
GARBAGE_BASE_DIR = "./dataset/teacup_mobrew_garbage"

# Create base directories
for folder in [CLEANED_BASE_DIR, GARBAGE_BASE_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# --- Model Setup ---
weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def is_teacup_related(image_path):
    """
    Evaluates if an image contains teacup-related content.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        input_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
        
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        
        # Relevant ImageNet indices:
        # 849: teapot
        # 504: coffee mug
        # 968: cup
        # 725: pitcher, ewer
        # 489: chalice
        # 553: goblet
        # 810: soup bowl
        # 923: plate
        # 734: platter
        # 907: wine glass
        # 449: beer glass
        teacup_indices = {849, 504, 968, 725, 489, 553, 810, 923, 734, 907, 449}
        
        for i in range(5):
            if top5_catid[i].item() in teacup_indices:
                return True
        return False
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def main():
    print(f"Starting teacup data cleansing on {device}...")
    
    if not os.path.exists(SOURCE_BASE_DIR):
        print(f"Source directory {SOURCE_BASE_DIR} not found.")
        return

    for category in os.listdir(SOURCE_BASE_DIR):
        cat_src_path = os.path.join(SOURCE_BASE_DIR, category)
        if not os.path.isdir(cat_src_path):
            continue
            
        cat_clean_path = os.path.join(CLEANED_BASE_DIR, category)
        cat_garbage_path = os.path.join(GARBAGE_BASE_DIR, category)
        
        if not os.path.exists(cat_clean_path): os.makedirs(cat_clean_path)
        if not os.path.exists(cat_garbage_path): os.makedirs(cat_garbage_path)

        print(f"Processing category: {category}")
        files = [f for f in os.listdir(cat_src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        for filename in files:
            file_path = os.path.join(cat_src_path, filename)
            if is_teacup_related(file_path):
                move(file_path, os.path.join(cat_clean_path, filename))
                # print(f"[KEEP] {filename}")
            else:
                move(file_path, os.path.join(cat_garbage_path, filename))
                print(f"[REJECT] {filename}")

    print(f"Cleansing complete.")

if __name__ == "__main__":
    main()
