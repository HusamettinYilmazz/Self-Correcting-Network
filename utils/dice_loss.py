import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=255, ignore_background=True):
        super().__init__()

        self.smooth = smooth
        self.ignore_index = ignore_index
        self.ignore_background = ignore_background

    def forward(self, logits, targets):
        """
        logits:  [B, C, H, W]
        targets: [B, H, W]
        """

        num_classes = logits.shape[1]

        # probabilities
        probs = torch.softmax(logits, dim=1)

        # valid pixels mask
        valid_mask = (targets != self.ignore_index)

        # replace ignore_index temporarily
        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0

        # one-hot encode targets
        targets_one_hot = F.one_hot(
            safe_targets,
            num_classes=num_classes
        ).permute(0, 3, 1, 2).float()

        # apply valid mask
        valid_mask = valid_mask.unsqueeze(1)

        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask

        # flatten spatial dims
        probs = probs.reshape(probs.shape[0], probs.shape[1], -1)
        targets_one_hot = targets_one_hot.reshape(
            targets_one_hot.shape[0],
            targets_one_hot.shape[1],
            -1
        )

        # remove background if needed
        if self.ignore_background:
            probs = probs[:, 1:]
            targets_one_hot = targets_one_hot[:, 1:]

        # intersection + union
        intersection = (probs * targets_one_hot).sum(dim=2)

        union = probs.sum(dim=2) + targets_one_hot.sum(dim=2)

        dice = (2 * intersection + self.smooth) / (
            union + self.smooth
        )

        # ignore classes absent from GT
        target_sum = targets_one_hot.sum(dim=2)

        valid_classes = target_sum > 0

        dice = dice[valid_classes]

        # fallback safety
        if dice.numel() == 0:
            return logits.new_tensor(0.0)

        dice = dice.mean()

        return 1 - dice
