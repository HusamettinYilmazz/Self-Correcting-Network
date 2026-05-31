import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.skip(x)

        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return self.act(x + res)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape

        y = self.pool(x).view(b, c)
        y = self.act(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)

        return x * y


class BoundingBoxEncoder(nn.Module):
    """
    Improved Bounding Box Attention Encoder:
    - image + bbox fusion
    - multi-scale residual attention
    - channel attention (SE block)
    """

    def __init__(self, num_classes, low_ch=256, high_ch=256, img_ch=256):
        super().__init__()

        # bbox projection
        self.box_low = nn.Conv2d(num_classes, low_ch, 1)
        self.box_high = nn.Conv2d(num_classes, high_ch, 1)

        # image feature projection
        self.img_low = nn.Conv2d(img_ch, low_ch, 1)
        self.img_high = nn.Conv2d(img_ch, high_ch, 1)

        # attention networks
        self.low_attn = nn.Sequential(
            ResidualBlock(low_ch * 2, low_ch, dilation=1),
            ResidualBlock(low_ch, low_ch, dilation=2),
            nn.Conv2d(low_ch, low_ch, 1),
            nn.Sigmoid()
        )

        self.high_attn = nn.Sequential(
            ResidualBlock(high_ch * 2, high_ch, dilation=1),
            ResidualBlock(high_ch, high_ch, dilation=2),
            nn.Conv2d(high_ch, high_ch, 1),
            nn.Sigmoid()
        )

        self.se_low = SEBlock(low_ch)
        self.se_high = SEBlock(high_ch)

    def forward(self, bb_mask, img_low_feat, img_high_feat, low_size, high_size):
        """
        bb_mask: (B, C+1, H, W)
        img_low_feat: encoder low-level feature map
        img_high_feat: ASPP / high-level feature map
        """

        bb_low = F.interpolate(bb_mask.float(), size=low_size, mode="nearest")
        bb_high = F.interpolate(bb_mask.float(), size=high_size, mode="nearest")

        bb_low = self.box_low(bb_low)
        bb_high = self.box_high(bb_high)

        img_low = self.img_low(img_low_feat)
        img_high = self.img_high(img_high_feat)

        low_fused = torch.cat([bb_low, img_low], dim=1)
        high_fused = torch.cat([bb_high, img_high], dim=1)

        attn_low = self.low_attn(low_fused)
        attn_high = self.high_attn(high_fused)

        attn_low = self.se_low(attn_low)
        attn_high = self.se_high(attn_high)

        return attn_low, attn_high
    