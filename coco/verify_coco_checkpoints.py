#!/usr/bin/env python3
"""Quick load verification for COCO fine-tuned checkpoints."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from core.avian_model import AvianRaptorNet_Fast


def verify(checkpoint_path: str, expected_classes: int = 80):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AvianRaptorNet_Fast(num_classes=expected_classes).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)

    head_key = "classifier_head.4.weight"
    actual_classes = state_dict[head_key].shape[0]
    assert actual_classes == expected_classes, (
        f"Expected {expected_classes} classes, got {actual_classes}"
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] {checkpoint_path}: loaded {actual_classes}-class model, "
          f"{total_params / 1e6:.2f}M params")


if __name__ == "__main__":
    verify("avian_coco_primary.pth")
    verify("avian_coco_refined.pth")
