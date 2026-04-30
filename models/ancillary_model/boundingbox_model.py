import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundingBoxEncoder(nn.Module):
    def __init__(self, num_classes: int, low_level_channels: int = 256, high_level_channels: int = 256):
        super().__init__()
        ## Scale 1: attention for low-level features (stride 4)
        self.attn_low = nn.Sequential(
            nn.Conv2d(num_classes, low_level_channels, 3, padding=1),
            nn.Sigmoid(),
        )
        ## Scale 2: attention for high-level (ASPP) features (stride 16)
        self.attn_high = nn.Sequential(
            nn.Conv2d(num_classes, high_level_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, bb_mask: torch.Tensor, low_size, high_size):
        bb_low = F.interpolate(bb_mask.float(), size=low_size, mode="nearest")
        bb_high = F.interpolate(bb_mask.float(), size=high_size, mode="nearest")
        return self.attn_low(bb_low), self.attn_high(bb_high)
