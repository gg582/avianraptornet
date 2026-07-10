import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from sklearn.cluster import KMeans
from shutil import copy2
from core.avian_model import AvianRaptorNet_Fast

# --- Configuration ---
SOURCE_BASE_DIR = "./dataset/teacup_mobrew_cleaned"
OUTPUT_DIR = "./dataset/teacup_clusters"
NUM_CLUSTERS = 5 # For more granular discovery within categories
PRETRAINED_WEIGHTS = "avian_raptor_fast_best.pth"

# --- Feature Extraction Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load AvianRaptorNet_Fast and repurpose it for feature extraction
full_model = AvianRaptorNet_Fast(num_classes=100)
if os.path.exists(PRETRAINED_WEIGHTS):
    state_dict = torch.load(PRETRAINED_WEIGHTS, map_location=device)
    full_model.load_state_dict(state_dict)

# We want features after Flatten() but before Linear(768, 100)
class AvianFeatureExtractor(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.retina = original_model.retina
        self.raptor_eye = original_model.raptor_eye
        self.body = original_model.body
        self.global_pool = original_model.global_pool
        # Take all parts of classifier_head except the final Linear layer
        self.feature_head = torch.nn.Sequential(*(list(original_model.classifier_head.children())[:-1]))

    def forward(self, x):
        x = self.retina(x)
        x = self.raptor_eye(x)
        x = self.body(x)
        x = self.global_pool(x)
        x = self.feature_head(x)
        return x

model = AvianFeatureExtractor(full_model).eval().to(device)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def extract_features(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_t = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(img_t)
        return features.cpu().numpy().flatten()
    except Exception:
        return None

# --- Main Logic ---
def main():
    print(f"Extracting features from {SOURCE_BASE_DIR}...")
    
    image_paths = []
    for root, _, files in os.walk(SOURCE_BASE_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                image_paths.append(os.path.join(root, f))
                
    features_list = []
    valid_files = []

    for path in image_paths:
        feat = extract_features(path)
        if feat is not None:
            features_list.append(feat)
            valid_files.append(path)

    if not features_list:
        print("No features extracted. Ensure images exist in the source directory.")
        return

    # Perform K-Means Clustering
    print(f"Clustering {len(features_list)} images into {NUM_CLUSTERS} groups...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(np.array(features_list))

    # Organize files into cluster folders
    for i in range(NUM_CLUSTERS):
        os.makedirs(os.path.join(OUTPUT_DIR, f"cluster_{i}"), exist_ok=True)

    for path, label in zip(valid_files, labels):
        copy2(path, os.path.join(OUTPUT_DIR, f"cluster_{label}", os.path.basename(path)))

    print(f"Auto-clustering complete. Check results in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
