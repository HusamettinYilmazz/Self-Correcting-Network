import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundingBoxEncoder(nn.Module):
    def __init__(self, num_classes: int=21, low_level_channels: int = 256, high_level_channels: int = 256, hidden_channels: int = 64):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Conv2d(num_classes, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )

        # Low-level attention head
        self.attn_low = nn.Sequential(
            nn.Conv2d(hidden_channels, low_level_channels, kernel_size=3, padding=1)
        )

        # High-level attention head
        self.attn_high = nn.Sequential(
            nn.Conv2d(hidden_channels, high_level_channels, kernel_size=3, padding=1)
        )

    def forward(self, bb_mask, low_features, high_features):
        """
        bb_mask:       [B,c,H,W]
        low_features:  [B,C_low,H/4,W/4]
        high_features: [B,C_high,H/16,W/16]
        """

        low_size = low_features.shape[-2:]
        high_size = high_features.shape[-2:]

        # Low scale
        bb_low = F.interpolate(
            bb_mask.float(),
            size=low_size,
            mode="nearest"
        )

        low_attn = self.shared(bb_low)
        low_attn = torch.sigmoid(self.attn_low(low_attn))

        # High scale
        bb_high = F.interpolate(
            bb_mask.float(),
            size=high_size,
            mode="nearest"
        )

        high_attn = self.shared(bb_high)
        high_attn = torch.sigmoid(self.attn_high(high_attn))

        # Residual attention
        low_features = low_features * (1.0 + low_attn)
        high_features = high_features * (1.0 + high_attn)

        return low_features, high_features
