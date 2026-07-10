import os
import requests
from duckduckgo_search import DDGS
from PIL import Image
from io import BytesIO
import time

def download_teacups(base_path, categories, images_per_prompt=20):
    """
    Download teacup images categorized by prompts.
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    with DDGS() as ddgs:
        for category, prompts in categories.items():
            cat_path = os.path.join(base_path, category)
            if not os.path.exists(cat_path):
                os.makedirs(cat_path)
            
            print(f"--- Scraping Category: {category} ---")
            
            for prompt in prompts:
                print(f"Prompt: {prompt}")
                results = ddgs.images(
                    keywords=prompt,
                    region="wt-wt",
                    safesearch="off",
                    size="Medium"
                )

                count = 0
                for res in results:
                    if count >= images_per_prompt:
                        break
                    
                    url = res.get("image")
                    try:
                        headers = {"User-Agent": "Mozilla/5.0"}
                        resp = requests.get(url, headers=headers, timeout=5)
                        
                        if resp.status_code == 200:
                            img = Image.open(BytesIO(resp.content)).convert("RGB")
                            # Resize to 224x224 as suggested for efficiency
                            img = img.resize((224, 224), Image.Resampling.LANCZOS)
                            
                            # Create a unique filename
                            filename = f"{prompt.replace(' ', '_').replace('"', '')}_{count:03d}.jpg"
                            img.save(os.path.join(cat_path, filename), "JPEG")
                            count += 1
                            if count % 5 == 0:
                                print(f"  Saved {count} images for this prompt...")
                    except Exception as e:
                        continue
                
                # Small delay to be polite to the search engine
                time.sleep(1)

# Prompt categories from user
TEACUP_CATEGORIES = {
    "shape_structure": [
        "Vintage porcelain teacup with handle silhouette",
        "Handleless Asian style teacup bowl",
        "Footed teacup with pedestal base",
        "Scalloped edge bone china saucer",
        "Flared rim teacup vintage",
        "Antique teapot with gooseneck spout",
        "Classic ceramic teapot with C-shaped handle"
    ],
    "detail_texture": [
        "Translucent English bone china teacup illumination",
        "Crackled glaze celadon teacup close up",
        "Hand-painted floral pattern on vintage teacup",
        "Gold gilded rim teacup details",
        "Blue Willow pattern teacup porcelain",
        "Vintage teacup backstamp maker's mark",
        "Royal Albert teacup bottom stamp"
    ],
    "style_culture": [
        "Traditional Japanese chawan matcha bowl",
        "Chinese Jingdezhen blue and white porcelain tea set",
        "Traditional Korean style water cup",
        "Traditional Korean style celadon chawan",
        "Thai Sukhothai stoneware ceramics fish motif",
        "Vietnamese Chu Dau blue and white porcelain tea set",
        "Victorian ornate teacup and saucer set",
        "Mid-century modern stackable teacups and mugs",
        "Modernist glass teacup with infuser",
        "Rustic handmade pottery tea mug"
    ]
}

if __name__ == "__main__":
    download_teacups("./dataset/teacup_mobrew", TEACUP_CATEGORIES, images_per_prompt=150)
