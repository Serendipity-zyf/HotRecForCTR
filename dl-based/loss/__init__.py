"""Loss functions for CTR prediction."""

from .BCELoss import BCELoss
from .FocalLoss import FocalLoss

__all__ = ["BCELoss", "FocalLoss"]
