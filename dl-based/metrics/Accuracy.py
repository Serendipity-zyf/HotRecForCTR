"""Accuracy metric for CTR prediction."""

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from utils.register import Registers


@Registers.metric_registry.register
class AccuracyMetric(nn.Module):
    """
    Accuracy metric for CTR prediction.
    Uses sklearn's implementation for better performance and reliability.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initialize AccuracyMetric.

        Args:
            threshold: Classification threshold for converting probabilities to predictions
        """
        super().__init__()
        self.name = "Accuracy"
        self.threshold = threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate accuracy score.

        Args:
            pred: Raw predictions (logits, before sigmoid)
            target: Ground truth labels (0 or 1)

        Returns:
            Accuracy score
        """
        # Convert to numpy arrays
        pred_prob = torch.sigmoid(pred).detach().cpu().numpy().flatten()
        target_np = target.detach().cpu().numpy().flatten()

        try:
            return accuracy_score(target_np, pred_prob >= self.threshold)
        except ValueError:
            return 0.0
