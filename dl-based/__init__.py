"""
Deep learning-based recommendation system.

This package contains components for building and training deep learning-based
recommendation systems, particularly for CTR (Click-Through Rate) prediction.
"""

from .utils.register import build_from_config

__all__ = ["build_from_config"]
