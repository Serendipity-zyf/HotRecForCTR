"""AUC metric for CTR prediction."""

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from utils.register import Registers


@Registers.metric_registry.register
class AUCMetric(nn.Module):
    """
    AUC (Area Under the ROC Curve) metric for CTR prediction.
    Uses sklearn's implementation for better performance and reliability.
    """

    def __init__(self):
        super().__init__()
        self.name = "AUC"

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate AUC score.

        Args:
            pred: Raw predictions (logits, before sigmoid)
            target: Ground truth labels (0 or 1)

        Returns:
            AUC score
        """
        # Convert to numpy arrays
        pred_prob = torch.sigmoid(pred).detach().cpu().numpy().flatten()
        target_np = target.detach().cpu().numpy().flatten()

        try:
            return roc_auc_score(target_np, pred_prob)
        except ValueError:
            # Handling the case with only one category
            return 0.5
