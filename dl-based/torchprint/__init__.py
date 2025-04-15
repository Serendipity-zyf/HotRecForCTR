"""
Custom model analysis module for PyTorch models.

This module provides functionality similar to torchinfo but with custom formatting
and additional features.
"""

from .model_info import analyze_model
from .model_info import ModelAnalyzer

__all__ = ["analyze_model", "ModelAnalyzer"]
