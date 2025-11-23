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
    def __init__(self, in_ch, out_ch, stride=1):
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

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

# --------------------------------------------------------
# AvianRaptorNet Fast
# --------------------------------------------------------
class AvianRaptorNet_Fast(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.retina = nn.Sequential(
            nn.Conv2d(3, 48, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48),
            BioMish()
        )
        self.raptor_eye = RaptorFovealLite(48, 96)
        self.body = nn.Sequential(
            FeatherBlock(96, 128, stride=2),
            FeatherBlock(128, 128, stride=1),
            FeatherBlock(128, 256, stride=2),
            FeatherBlock(256, 256, stride=1),
            FeatherBlock(256, 256, stride=1),
            FeatherBlock(256, 512, stride=2),
            FeatherBlock(512, 512, stride=1),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier_head = nn.Sequential(
            nn.Conv2d(512, 768, 1, bias=False),
            BioMish(),
            nn.Dropout(0.2),
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
