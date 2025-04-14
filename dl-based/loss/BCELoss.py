"""Binary Cross Entropy loss for CTR prediction."""

import torch
import torch.nn as nn

from utils.register import Registers
from .base_loss import BaseLoss


@Registers.loss_registry.register
class BCELoss(BaseLoss):
    """
    Weighted Binary Cross Entropy loss.

    Args:
        pos_weight: Weight for positive samples (1)
        reduction: Reduction method
    """

    def __init__(self, pos_weight: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.name = "BCELoss"
        self.pos_weight = torch.tensor([pos_weight])
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight, reduction=reduction
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted BCE loss.

        Args:
            pred: Raw predictions (before sigmoid)
            target: Ground truth labels (0 or 1)

        Returns:
            loss value
        """
        return self.loss_fn(pred, target)
