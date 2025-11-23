# AvianRaptorNet: Bio-Inspired Efficient Vision Models

[![License](https://img.shields.io/badge/License-Apache\_2.0-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![SOTA](https://img.shields.io/badge/CIFAR100-80.49%25-brightgreen.svg)]()
[![Params](https://img.shields.io/badge/Params-3.23M-yellow.svg)]()

> **For Researchers:** If you use this model in your research, **you must cite** it. Failure to cite is a violation of the academic ethos. See the [Citation](#citation) section below.
>
> **For Enterprises:** This project is licensed under **Apache 2.0**. You are free to use, modify, and distribute it for commercial purposes without restriction.

---

## 1. Overview

**AvianRaptorNet** is a biologically inspired convolutional neural network designed for **extreme efficiency**. It mimics the **Dual-Fovea system of Raptors (Birds of Prey)**, allowing the model to process fine details (central fovea) and broad context (peripheral fovea) simultaneously using a lightweight architecture.

We have achieved **80.49% Top-1 Accuracy** on CIFAR-100 with only **3.23M parameters**, setting a new efficiency frontier beyond MobileNetV2 and GhostNet.

### Key Features
- **Extreme Efficiency:** 3.23M Params (~12MB). Designed for Edge AI (Jetson, Raspberry Pi, Mobile).
- **Raptor Dual-Flow Architecture:** Parallel Detail (3×3) and Context (Dilated) streams.
- **BioMish Activation:** JIT-compiled stochastic activation mimicking biological neuron heterogeneity.
- **Hardware Optimized:** Fully tuned for NVIDIA Ampere+ (TF32, Channels Last).

---

## 2. Main Results (Fast Model)

| Model                   | Params   | Top-1 Acc | Training Strategy                          |
|-------------------------|----------|-----------|--------------------------------------------|
| **AvianRaptorNet-Fast** | **3.23M** | **80.49%** | Mixup + AutoAugment + Safe Refinement     |
| MobileNetV2 (ref)       | ~3.4M    | ~73-74%   | Standard                                   |
| GhostNet (ref)          | ~5.2M    | ~74-77%   | Standard                                   |

> **Note:** 80.49% achieved by ultra-low LR (1e-5) refinement after convergence — safely settles into global minima.

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

## 5. Usage

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

## 6. Citation

@misc{AvianRaptorNet2025,
  author       = {gg582},
  title        = {AvianRaptorNet: Bio-Inspired Lightweight Vision Model with Raptor Dual-Flow Architecture},
  year         = {2025},
  publisher    = {GitHub},
  note         = {Achieved 80.49% on CIFAR-100 with 3.23M parameters},
  howpublished = {\url{https://github.com/gg582/aviannet}}
}

---

## 7. License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
