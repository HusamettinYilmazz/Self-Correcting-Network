import torch
import torch.nn as nn

from ..self_correction_interface import SelfCorrectionModule


class ConvCorrectionModule(SelfCorrectionModule):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_classes * 2, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=3, padding=1),
        )

    def forward(self, primary_logits: torch.Tensor, ancillary_logits: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([primary_logits, ancillary_logits], dim=1)
        logits = self.net(combined)
        return torch.softmax(logits, dim=1)
