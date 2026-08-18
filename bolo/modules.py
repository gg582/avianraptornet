"""BOLO custom modules.

Ports AvianRaptorNet's bio-inspired blocks (BioMish, RaptorFoveal,
FeatherBlock) to the ultralytics parser conventions, and defines the
dual-stride motion correction layer (MotionStrideAttention).

Per the ultralytics `parse_model` conventions, custom modules are designed
to be channel-preserving: YAML args are passed verbatim to the constructor,
so they use `__init__(self, c1, ...)`; channel changes and downsampling are
left to the standard Conv layers. Only pure ONNX-exportable ops are used
(no scripted functions or dynamic control flow).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BioMish(nn.Module):
    """BioMish activation: x * tanh(softplus(x))."""

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class ChannelAttention(nn.Module):
    """AvianRaptorNet channel attention (BioMish-activated variant)."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channels, max(channels // reduction, 8), 1, bias=False),
            BioMish(),
            nn.Conv2d(max(channels // reduction, 8), channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(F.adaptive_avg_pool2d(x, 1))


class RaptorFoveal(nn.Module):
    """Raptor dual-fovea block (channel-preserving).

    A central (foveal) 3x3 conv captures fine detail while a peripheral
    dilated 3x3 conv captures context in parallel; the two streams are
    fused with a 1x1 conv. Downsampling / channel changes are handled by
    the surrounding standard Conv layers.
    """

    def __init__(self, c1):
        super().__init__()
        mid = c1 // 2
        groups = 4 if (c1 % 4 == 0 and mid % 4 == 0) else 1
        self.central = nn.Sequential(
            nn.Conv2d(c1, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            BioMish(),
        )
        self.peripheral = nn.Sequential(
            nn.Conv2d(c1, mid, 3, padding=2, dilation=2, groups=groups, bias=False),
            nn.BatchNorm2d(mid),
            BioMish(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(mid * 2, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
        )

    def forward(self, x):
        c = self.central(x)
        p = self.peripheral(x)
        return self.fusion(torch.cat([c, p], dim=1))


class FeatherBlock(nn.Module):
    """Feather inverted-bottleneck residual block (channel-preserving).

    1x1 expansion -> depthwise 3x3 -> channel attention -> 1x1 projection,
    with a residual connection and optional stochastic depth.
    """

    def __init__(self, c1, expansion=2, drop_path_rate=0.0):
        super().__init__()
        exp = c1 * expansion
        self.conv = nn.Sequential(
            nn.Conv2d(c1, exp, 1, bias=False),
            nn.BatchNorm2d(exp),
            BioMish(),
            nn.Conv2d(exp, exp, 3, padding=1, groups=exp, bias=False),
            nn.BatchNorm2d(exp),
            BioMish(),
            ChannelAttention(exp),
            nn.Conv2d(exp, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.drop_path_rate = drop_path_rate

    def forward(self, x):
        y = self.conv(x)
        if self.training and self.drop_path_rate > 0.0:
            keep = 1.0 - self.drop_path_rate
            mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
            y = y.div(keep) * mask
        return x + y


class MotionStrideAttention(nn.Module):
    """Dual-stride motion correction layer (channel-preserving).

    Captures motion blur / momentum at two different spatial strides:

    - Large-stride (4x) branch: coarse motion energy map over wide regions
    - Small-stride (2x) branch: fine motion energy map over narrow regions

    Both energy maps are upsampled to the input resolution and summed, then
    a sigmoid produces the momentum attention map `a`. A depthwise residual
    correction branch `corr` (inverse-estimation / deblur role) is applied
    selectively only where the attention is high:

        out = x + a * corr(x)
    """

    def __init__(self, c1, reduction=4):
        super().__init__()
        r = max(c1 // reduction, 8)
        # Large-stride branch: stride-4 downsample -> coarse momentum estimate
        self.large_stride = nn.Sequential(
            nn.AvgPool2d(4),
            nn.Conv2d(c1, r, 1, bias=False),
            BioMish(),
            nn.Conv2d(r, r, 3, padding=1, groups=r, bias=False),
            BioMish(),
            nn.Conv2d(r, 1, 1),
        )
        # Small-stride branch: stride-2 downsample -> fine momentum estimate
        self.small_stride = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(c1, r, 1, bias=False),
            BioMish(),
            nn.Conv2d(r, r, 3, padding=1, groups=r, bias=False),
            BioMish(),
            nn.Conv2d(r, 1, 1),
        )
        # Correction (inverse / deblur) branch: depthwise 3x3 -> 1x1 residual
        self.correction = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            BioMish(),
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
        )

    def forward(self, x):
        size = x.shape[-2:]
        large = F.interpolate(self.large_stride(x), size=size, mode="nearest")
        small = F.interpolate(self.small_stride(x), size=size, mode="nearest")
        attn = torch.sigmoid(large + small)
        return x + attn * self.correction(x)
