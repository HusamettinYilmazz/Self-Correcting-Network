from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class SelfCorrectionModule(ABC, nn.Module):
    """
    All self-correction variants implement this interface.
    Input:  primary_logits   (B, C, H, W)
            ancillary_logits (B, C, H, W)
    Output: soft_labels      (B, C, H, W)  — probabilities summing to 1 on dim=1
    """

    @abstractmethod
    def forward(self, primary_logits: torch.Tensor, 
                ancillary_logits: torch.Tensor,) -> torch.Tensor:
        
        pass
