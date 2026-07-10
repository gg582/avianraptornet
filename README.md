# AvianRaptorNet: Bio-Inspired Efficient Vision Models

[![License](https://img.shields.io/badge/License-Apache\_2.0-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![SOTA](https://img.shields.io/badge/CIFAR100-81.28%25-brightgreen.svg)]()
[![Params](https://img.shields.io/badge/Params-3.23M-yellow.svg)]()

> **For Researchers:** If you use this model in your research, **you must cite** it. Failure to cite is a violation of the academic ethos. See the [Citation](#citation) section below.
>
> **For Enterprises:** This project is licensed under **Apache 2.0**. You are free to use, modify, and distribute it for commercial purposes without restriction.

---

## 1. Overview

**AvianRaptorNet** is a biologically inspired convolutional neural network designed for **extreme efficiency**. It mimics the **Dual-Fovea system of Raptors (Birds of Prey)**, allowing the model to process fine details (central fovea) and broad context (peripheral fovea) simultaneously using a lightweight architecture.

We have achieved **81.28% Top-1 Accuracy** on CIFAR-100 with only **3.23M parameters**, setting a new efficiency frontier beyond MobileNetV2 and GhostNet.

### Key Features
- **Extreme Efficiency:** 3.23M Params (~12MB). Designed for Edge AI (Jetson, Raspberry Pi, Mobile).
- **Raptor Dual-Flow Architecture:** Parallel Detail (3×3) and Context (Dilated) streams.
- **BioMish Activation:** JIT-compiled stochastic activation mimicking biological neuron heterogeneity.
- **Hardware Optimized:** Fully tuned for NVIDIA Ampere+ (TF32, Channels Last).

---

## 2. Main Results (Fast Model)

| Model                   | Params   | Top-1 Acc | Training Strategy                          |
|-------------------------|----------|-----------|--------------------------------------------|
| **AvianRaptorNet-Fast** | **3.23M** | **81.28%** | Mixup + AutoAugment + Safe Refinement     |
| **AvianRaptorNet-Fast** | **3.23M** | **80.42%** | Mixup + AutoAugment                       |
| DenseNet-BC(ref)        | 2.8M     | **80-82%**  | AutoAugment                             |
| MobileNetV2 (ref)       | ~3.4M    | ~73-74%   | Standard                                   |
| GhostNet (ref)          | ~5.2M    | ~74-77%   | Standard                                   |

> **Note:** 81.28% achieved by ultra-low LR (1e-5) refinement after convergence — safely settles into global minima.

---

## 3. Architecture

1. **Retina (Stem):** Fast initial encoding  
2. **Raptor Eye (Dual-Flow):**
   - Foveal Path → high-resolution details
   - Peripheral Path → dilated context
   - Pecten-inspired fusion attention
3. **Tectofugal Stream:** Feather Blocks (Ghost-style, minimal FLOPs)
4. **Wulst (Head):** Bio-inspired classifier

[Image: raptor dual fovea anatomy]

---

## 4. Experimental Models

- **AvianRaptorNet-Medium / Huge:** Larger experimental variants (under active development). Currently less efficient than Fast on CIFAR-100.

---

## 5. Teacup Classification (mobrew Project)

Repurposing AvianRaptorNet for high-fidelity teacup and tea set classification. This sub-project focuses on distinguishing fine patterns and physical silhouettes using the **Dual-Flow** architecture.

### Dataset Pipeline
1. **Scraping:** `scrape_teacup.py` - Categorized scraping using specialized prompts (Shape, Detail, Style).
2. **Filtering:** `filter_non_teacup.py` - Automated data cleansing using ResNet50 to ensure only teacup-related images are kept.
3. **Fine-tuning:** `fine_tune_teacup.py` - Transfer learning from CIFAR-100 weights to teacup categories, optimized for **RTX 3070** (AMP, Channels Last).
4. **Clustering:** `teacup_auto_cluster.py` - Unsupervised discovery of sub-categories using AvianRaptorNet as a feature extractor.

### Usage
```bash
# 1. Scrape data
python scrape_teacup.py

# 2. Clean data
python filter_non_teacup.py

# 3. Fine-tune model
python fine_tune_teacup.py

# 4. Run inference
python teacup_inference.py <image_path>
```

---

## 6. Installation & Usage

### Installation
```python3
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Training & Refinement
```python3
python3 cifar100.py
python3 finetune_cifar100.py --weights=path_to_weight.pth
```

---

## 6. COCO Object-Crop Fine-Tuning

This project includes a transfer-learning pipeline that fine-tunes the CIFAR-100 pretrained **AvianRaptorNet-Fast** on COCO object crops for the 80 COCO thing categories.

### How it works

- **Source checkpoint:** `avian_raptor_fast_best.pth` (CIFAR-100, 100 classes)
- **Target task:** classify cropped COCO objects into the 80 COCO categories
- **Pipeline:**
  1. Load the pretrained backbone and replace the classifier head from 100 → 80 classes.
  2. **Primary stage:** SGD with mixup + AutoAugment for 40 epochs → `avian_coco_primary.pth`
  3. **Refinement stage:** AdamW with an ultra-low learning rate (1e-5) for 10 epochs → `avian_coco_refined.pth`

### Hardware / tuning notes

These defaults are tuned for an **RTX 3070 8 GB** card. Larger resolutions or batch sizes run out of memory on this GPU.

| Setting | Value |
|---------|-------|
| Input resolution | 128 × 128 |
| Batch size | 12 |
| `num_workers` | 0 |
| Primary optimizer | SGD, lr=0.01, momentum=0.9, weight_decay=1e-4 |
| Refinement optimizer | AdamW, lr=1e-5, weight_decay=1e-4 |
| Primary epochs | 40 |
| Refinement epochs | 10 |

### Usage

```bash
# Full two-stage training (requires ./data/coco_subset_20k)
python coco.py

# If you already have avian_coco_primary.pth and only want to rerun refinement:
python coco.py --skip-primary
```

### Outputs

| Checkpoint | Val Top-1 Accuracy | File |
|------------|-------------------|------|
| Primary | **60.50%** | `avian_coco_primary.pth` |
| Refined | **62.34%** | `avian_coco_refined.pth` |

Both checkpoints can be loaded with `AvianRaptorNet_Fast(num_classes=80)`.

---

## 7. Citation

@misc{AvianRaptorNet2025,
  author       = {gg582},
  title        = {AvianRaptorNet: Bio-Inspired Lightweight Vision Model with Raptor Dual-Flow Architecture},
  year         = {2025},
  publisher    = {GitHub},
  note         = {Achieved 81.28% on CIFAR-100 with 3.23M parameters},
  howpublished = {\url{https://github.com/gg582/aviannet}}
}

---

## 8. License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
