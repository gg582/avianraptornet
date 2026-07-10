import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------
# JIT & Bio Components
# --------------------------------------------------------
@torch.jit.script
def biomish_activation(x):
    return x * torch.tanh(F.softplus(x))

class BioMish(nn.Module):
    def forward(self, x):
        return biomish_activation(x)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            BioMish(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

class DropPath(nn.Module):
    """Stochastic depth (drop path) regularization."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask

# --------------------------------------------------------
# Blocks
# --------------------------------------------------------
class RaptorFovealLite(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 2
        self.central = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            BioMish()
        )
        self.peripheral = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=2, dilation=2, groups=4, bias=False),
            nn.BatchNorm2d(mid),
            BioMish()
        )
        self.fusion = nn.Conv2d(mid * 2, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        c = self.central(x)
        p = self.peripheral(x)
        return self.bn(self.fusion(torch.cat([c, p], dim=1)))

class FeatherBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, drop_path_rate=0.0):
        super().__init__()
        self.use_res_connect = (stride == 1 and in_ch == out_ch)
        exp_size = in_ch * 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, exp_size, 1, bias=False),
            nn.BatchNorm2d(exp_size),
            BioMish(),
            nn.Conv2d(exp_size, exp_size, 3, stride=stride, padding=1, groups=exp_size, bias=False),
            nn.BatchNorm2d(exp_size),
            BioMish(),
            ChannelAttention(exp_size),
            nn.Conv2d(exp_size, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.drop_path = DropPath(drop_path_rate) if self.use_res_connect else nn.Identity()

    def forward(self, x):
        if self.use_res_connect:
            return x + self.drop_path(self.conv(x))
        return self.conv(x)

# --------------------------------------------------------
# AvianRaptorNet Fast
# --------------------------------------------------------
class AvianRaptorNet_Fast(nn.Module):
    def __init__(self, num_classes=100, dropout=0.2, drop_path_rate=0.0):
        super().__init__()
        self.retina = nn.Sequential(
            nn.Conv2d(3, 48, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48),
            BioMish()
        )
        self.raptor_eye = RaptorFovealLite(48, 96)

        # Linearly increase stochastic depth rate across residual blocks.
        blocks = [
            (96, 128, 2),
            (128, 128, 1),
            (128, 256, 2),
            (256, 256, 1),
            (256, 256, 1),
            (256, 512, 2),
            (512, 512, 1),
        ]
        num_res_blocks = sum(1 for in_ch, out_ch, stride in blocks if stride == 1 and in_ch == out_ch)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_res_blocks)] if num_res_blocks > 0 else []
        dpr_iter = iter(dpr)

        self.body = nn.Sequential(*[
            FeatherBlock(
                in_ch, out_ch, stride=stride,
                drop_path_rate=next(dpr_iter) if (stride == 1 and in_ch == out_ch) else 0.0
            )
            for in_ch, out_ch, stride in blocks
        ])
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier_head = nn.Sequential(
            nn.Conv2d(512, 768, 1, bias=False),
            BioMish(),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(768, num_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.retina(x)
        x = self.raptor_eye(x)
        x = self.body(x)
        x = self.global_pool(x)
        x = self.classifier_head(x)
        return x
