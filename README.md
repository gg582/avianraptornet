# AvianRaptorNet: Bio-Inspired Vision Models

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![CIFAR100](https://img.shields.io/badge/CIFAR100-81.28%25-brightgreen.svg)]()
[![Params](https://img.shields.io/badge/Params-3.23M-yellow.svg)]()

---

## 1. Overview

**AvianRaptorNet** is a convolutional neural network inspired by the dual-fovea vision system of raptors (birds of prey). It processes fine details through a central foveal path and context through a dilated peripheral path in parallel.

The fast variant achieves **81.28% Top-1 Accuracy** on CIFAR-100 with **3.23M parameters**.

### Key Features

- **Lightweight:** 3.23M parameters (~12 MB).
- **Dual-Flow Architecture:** Parallel detail (3×3) and context (dilated) streams.
- **BioMish Activation:** JIT-compiled activation function.
- **Hardware Tuned:** Uses TF32 and channels-last memory format on NVIDIA Ampere GPUs.

---

## 2. Main Results (Fast Model)

| Model                   | Params   | Top-1 Acc | Training Strategy                          |
|-------------------------|----------|-----------|--------------------------------------------|
| AvianRaptorNet-Fast     | 3.23M    | 81.28%    | Mixup + AutoAugment + Safe Refinement      |
| AvianRaptorNet-Fast     | 3.23M    | 80.42%    | Mixup + AutoAugment                        |
| DenseNet-BC (ref)       | 2.8M     | 80–82%    | AutoAugment                                |
| MobileNetV2 (ref)       | ~3.4M    | ~73–74%   | Standard                                   |
| GhostNet (ref)          | ~5.2M    | ~74–77%   | Standard                                   |

> **Note:** The 81.28% result was obtained by applying an ultra-low learning-rate refinement stage after the main training run.

---

## 3. Architecture

1. **Retina (Stem):** Initial convolution and normalization.
2. **Raptor Eye (Dual-Flow):**
   - Foveal path → 3×3 convolution for local detail.
   - Peripheral path → dilated 3×3 convolution for context.
   - Fusion via 1×1 convolution and batch normalization.
3. **Body:** Feather Blocks, an inverted-bottleneck residual block with depthwise separable convolution and channel attention.
4. **Head:** Global average pooling followed by a 1×1 convolution, activation, dropout, flatten, and linear classifier.

Class diagrams for all model variants are available in [`uml/`](uml/).

---

## 4. Experimental Models

- **AvianRaptorNet-Medium:** A ResNet-50-scale experimental variant built with `RaptorBottleneck` blocks.

---

## 5. Teacup Classification (mobrew Project)

A transfer-learning experiment that fine-tunes AvianRaptorNet-Fast on teacup and tea-set images.

### Dataset Pipeline

1. **Scraping:** `experiments/scrape_teacup.py`
2. **Filtering:** `experiments/filter_non_teacup.py`
3. **Fine-tuning:** `experiments/fine_tune_teacup.py`
4. **Clustering:** `experiments/teacup_auto_cluster.py`

### Usage

```bash
python3 experiments/scrape_teacup.py
python3 experiments/filter_non_teacup.py
python3 experiments/fine_tune_teacup.py
python3 experiments/teacup_inference.py <image_path>
```

---

## 6. Installation & Usage

### Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Training & Refinement

```bash
python -m cifar100.cifar100
python cifar100/finetune_cifar100.py --weights=path_to_weight.pth
```

---

## 7. COCO Object-Crop Fine-Tuning

This project includes a transfer-learning pipeline that fine-tunes the CIFAR-100 pretrained **AvianRaptorNet-Fast** on cropped COCO objects for the 80 COCO thing categories.

### How it works

- **Source checkpoint:** `avian_raptor_fast_best.pth` (CIFAR-100, 100 classes)
- **Target task:** classify cropped COCO objects into the 80 COCO categories
- **Pipeline:**
  1. Load the pretrained backbone and replace the classifier head from 100 → 80 classes.
  2. **Primary stage:** SGD with linear warmup + cosine decay, mixup, and AutoAugment for 80 epochs → `avian_coco_primary.pth`
  3. **Refinement stage:** AdamW with an ultra-low learning rate (1e-5) for 25 epochs → `avian_coco_refined.pth`
  4. **Micro stage:** AdamW with a micro learning rate (1e-6) for up to 30 epochs, with early stopping → `avian_coco_micro.pth`

### Hardware / tuning notes

These defaults are tuned for an **RTX 3070 8 GB** card. Larger resolutions or batch sizes run out of memory on this GPU.

| Setting | Value |
|---------|-------|
| Input resolution | 128 × 128 |
| Batch size | 12 |
| `num_workers` | 0 |
| Primary optimizer | SGD, lr=0.01, momentum=0.9, weight_decay=1e-4 |
| Refinement optimizer | AdamW, lr=1e-5, weight_decay=1e-4 |
| Micro optimizer | AdamW, lr=1e-6, weight_decay=1e-6 |
| Primary epochs | 80 |
| Refinement epochs | 25 |
| Micro epochs | 30 (early stopping patience = 10) |

### Usage

```bash
# Full training (requires ./data/coco_subset_20k)
python -m coco.coco

# Micro refinement from the best refined checkpoint
python -m coco.coco_micro_refinement
```

### Outputs

| Stage | Val Top-1 Accuracy | Checkpoint |
|-------|-------------------|------------|
| Primary (EMA) | 60.57% | `avian_coco_primary_ema.pth` |
| Refined (EMA) | 61.79% | `avian_coco_refined_ema.pth` |
| Micro (raw) | 63.63% | `avian_coco_micro.pth` |
| Micro (EMA) | 63.22% | `avian_coco_micro_ema.pth` |

All checkpoints can be loaded with `AvianRaptorNet_Fast(num_classes=80)`.

---

## 8. Citation

```bibtex
@misc{AvianRaptorNet2025,
  author       = {gg582},
  title        = {AvianRaptorNet: Bio-Inspired Lightweight Vision Model with Raptor Dual-Flow Architecture},
  year         = {2025},
  publisher    = {GitHub},
  note         = {Achieved 81.28% on CIFAR-100 with 3.23M parameters},
  howpublished = {\url{https://github.com/gg582/aviannet}}
}
```

---

## 9. License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
