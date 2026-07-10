#!/bin/bash

# Configuration
DATASET_RAW="./dataset/teacup_mobrew"
DATASET_CLEANED="./dataset/teacup_mobrew_cleaned"
MODEL_FILE="teacup_avian_raptor.pth"

# 1. Scrape data if it doesn't exist
if [ ! -d "$DATASET_RAW" ] || [ "$(ls -A $DATASET_RAW 2>/dev/null)" == "" ]; then
    echo "--- Step 1: Scraping teacup images ---"
    python3 experiments/scrape_teacup.py
else
    echo "--- Step 1: Dataset already exists, skipping scraping ---"
fi

# 2. Filter/Clean data if cleaned dataset is missing
if [ ! -d "$DATASET_CLEANED" ] || [ "$(ls -A $DATASET_CLEANED 2>/dev/null)" == "" ]; then
    echo "--- Step 2: Filtering non-teacup images ---"
    python3 experiments/filter_non_teacup.py
else
    echo "--- Step 2: Cleaned dataset already exists, skipping filtering ---"
fi

# 3. Fine-tuning
echo "--- Step 3: Fine-tuning the model ---"
export PYTORCH_ALLOC_CONF=expandable_segments:True
python3 experiments/fine_tune_teacup.py

# 4. Inference
echo "--- Step 4: Running inference ---"
# Test with a few images from the cleaned dataset if available
SAMPLE_IMAGE=$(find "$DATASET_CLEANED" -name "*.jpg" | head -n 1)
if [ -n "$SAMPLE_IMAGE" ]; then
    echo "Testing with sample image: $SAMPLE_IMAGE"
    python3 experiments/inference_teacups.py --images "$SAMPLE_IMAGE"
else
    echo "No sample image found for testing."
fi

echo "--- Workflow Complete ---"
echo "You can now use: python3 experiments/inference_teacups.py --images <path1>,<path2>"
