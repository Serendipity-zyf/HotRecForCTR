"""Dataset implementations for CTR prediction."""

from .CriteoDataset import CriteoDataset
from .AmazonDataset import AmazonDataset

__all__ = ["CriteoDataset", "AmazonDataset"]
