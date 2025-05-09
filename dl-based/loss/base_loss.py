"""Base loss class for CTR prediction."""

import torch
import torch.nn as nn
from abc import ABC
from abc import abstractmethod


class BaseLoss(nn.Module, ABC):
    """Base loss class for CTR prediction."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate loss."""
        pass
