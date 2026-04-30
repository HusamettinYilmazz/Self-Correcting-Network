
import torch
import torch.nn as nn
import torchvision.models as tvm

from .aspp import ASPPModule

class DeepLabV3PlusEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet101", pretrained: bool = True):
        super().__init__()
        if backbone == "resnet101":
            base = tvm.resnet101(pretrained=pretrained, replace_stride_with_dilation=[False, True, True])
            self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4
            aspp_in = 2048
            self.low_level_channels = 256
        else:
            raise NotImplementedError(f"Backbone '{backbone}' not implemented. Add it here.")

        self.aspp = ASPPModule(aspp_in, out_channels=256)

    def forward(self, x: torch.Tensor):
        x = self.layer0(x)
        low = self.layer1(x)     ## (B, 256, H/4, W/4) passed to decoder
        x = self.layer2(low)
        x = self.layer3(x)
        x = self.layer4(x)
        high = self.aspp(x)      ## (B, 256, H/16, W/16)
        return low, high
