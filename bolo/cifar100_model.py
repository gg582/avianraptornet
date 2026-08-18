"""Reformed AvianRaptorNet-Fast for CIFAR-100 with BOLO motion attention."""

from pathlib import Path

import torch
import torch.nn as nn

from bolo.modules import MotionStrideAttention

ROOT = Path(__file__).resolve().parent.parent


def _identity_initialize(module):
    """Make a MotionStrideAttention block an exact residual identity initially."""
    final_bn = module.correction[-1]
    nn.init.zeros_(final_bn.weight)
    nn.init.zeros_(final_bn.bias)


class BoloCIFAR100(nn.Module):
    """AvianRaptorNet-Fast reformed with identity-initialized BOLO layers.

    The original proven CIFAR-100 model is retained without architectural
    changes to its feature extractor or classifier. Dual-stride motion
    correction is inserted after the 256-channel stage and immediately before
    classification. Both correction branches start at zero, so loading an
    AvianRaptorNet-Fast checkpoint preserves its output exactly at step zero.
    """

    def __init__(self, num_classes=100, dropout=0.2, drop_path_rate=0.0):
        super().__init__()
        from core.avian_model import AvianRaptorNet_Fast

        self.backbone = AvianRaptorNet_Fast(
            num_classes=num_classes, dropout=dropout, drop_path_rate=drop_path_rate
        )
        self.mid_motion = MotionStrideAttention(256)
        self.final_motion = MotionStrideAttention(512)
        _identity_initialize(self.mid_motion)
        _identity_initialize(self.final_motion)

    def load_aviannet_weights(self, weights):
        """Load a plain AvianRaptorNet-Fast state dict into the retained backbone."""
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.backbone.load_state_dict(state, strict=True)

    def forward(self, x):
        x = self.backbone.retina(x)
        x = self.backbone.raptor_eye(x)
        for index, block in enumerate(self.backbone.body):
            x = block(x)
            if index == 4:
                x = self.mid_motion(x)
        x = self.final_motion(x)
        x = self.backbone.global_pool(x)
        return self.backbone.classifier_head(x)
