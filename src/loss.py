"""
Imbalance-Aware Loss Functions Module
Provides Focal Loss and Weighted BCE Loss for industrial failure classification (~43:1 imbalance).
"""

import math
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class BinaryFocalLoss(nn.Module):
        """
        Binary Focal Loss for severe class imbalance.
        Down-weights easy/majority samples and focuses on rare failure events.
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        """

        def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            """
            Args:
                inputs: Logits tensor of shape (batch_size, 1) or (batch_size,)
                targets: Binary labels of shape (batch_size, 1) or (batch_size,)
            """
            inputs = inputs.view(-1)
            targets = targets.view(-1).float()

            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
            p = torch.sigmoid(inputs)
            p_t = p * targets + (1 - p) * (1 - targets)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

            focal_weight = alpha_t * (1.0 - p_t).pow(self.gamma)
            loss = focal_weight * bce_loss

            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            return loss

    class WeightedBCELoss(nn.Module):
        """Binary Cross Entropy with positive class weighting."""

        def __init__(self, pos_weight: float = 1.0):
            super().__init__()
            self.pos_weight = torch.tensor([pos_weight], dtype=torch.float32)

        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            inputs = inputs.view(-1)
            targets = targets.view(-1).float()
            weight = self.pos_weight.to(inputs.device)
            return F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=weight)

else:
    class BinaryFocalLoss:
        pass

    class WeightedBCELoss:
        pass
