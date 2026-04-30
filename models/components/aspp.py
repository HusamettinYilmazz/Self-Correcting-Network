import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPPModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.atrous6 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.atrous12 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.atrous18 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.bn = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(5)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        branches = [
            F.relu(self.bn[0](self.conv1x1(x))),
            F.relu(self.bn[1](self.atrous6(x))),
            F.relu(self.bn[2](self.atrous12(x))),
            F.relu(self.bn[3](self.atrous18(x))),
            F.interpolate(F.relu(self.bn[4](self.global_pool(x))), size=(h, w), mode="bilinear", align_corners=False),
        ]
        return self.project(torch.cat(branches, dim=1))
