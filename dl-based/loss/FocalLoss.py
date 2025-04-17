"""Focal Loss for CTR prediction."""

import torch
import torch.nn.functional as F

from utils.register import Registers
from .base_loss import BaseLoss


@Registers.loss_registry.register
class FocalLoss(BaseLoss):
    """
    Focal Loss for dealing with class imbalance.

    Args:
        alpha: Weight for positive samples (1)
        gamma: Focusing parameter
        reduction: Reduction method
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.name = "FocalLoss"
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal loss.

        Args:
            pred: Raw predictions (logits, before sigmoid)
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

        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pred_prob = torch.sigmoid(pred)
        # Calculate Probability for Focal Loss Weights
        p_t = torch.where(target == 1, pred_prob, 1 - pred_prob)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
