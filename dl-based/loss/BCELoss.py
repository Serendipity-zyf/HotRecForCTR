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
        self.pos_weight_value = pos_weight
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted BCE loss.

        Args:
            pred: Raw predictions (before sigmoid)
            target: Ground truth labels (0 or 1)

        Returns:
            loss value
        """
        if target.dtype != pred.dtype:
            target = target.float()

        if pred.dim() > target.dim():
            pred = pred.squeeze(-1)
        elif pred.dim() < target.dim():
            pred = pred.unsqueeze(-1)

        pos_weight = torch.tensor([self.pos_weight_value], device=pred.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=self.reduction)
        return loss_fn(pred, target)
