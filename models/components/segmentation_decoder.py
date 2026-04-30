
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, low_level_channels: int, num_classes: int):
        super().__init__()
        ## Project low-level features to 48 channels (paper)
        self.project_low = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        ## After concatenation: 256 (ASPP) + 48 (low-level) = 304
        self.refine = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, low: torch.Tensor, high: torch.Tensor, target_size) -> torch.Tensor:
        low_proj = self.project_low(low)
        ## Upsample ASPP output to match low-level feature size
        high_up = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = self.refine(torch.cat([high_up, low_proj], dim=1))
        logits = self.classifier(x)
        return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
