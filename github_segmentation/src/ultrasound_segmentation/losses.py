from __future__ import annotations

import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target)
        probability = logits.sigmoid()
        intersection = (probability * target).sum(dim=(1, 2, 3))
        denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + self.eps) / (denominator + self.eps)).mean()
        return bce + dice_loss
