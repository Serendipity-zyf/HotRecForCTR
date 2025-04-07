"""
Deep learning-based recommendation system.

This package contains components for building and training deep learning-based
recommendation systems, particularly for CTR (Click-Through Rate) prediction.
"""

from .utils.register import (
    MODEL_REGISTRY,
    LOSS_REGISTRY,
    METRIC_REGISTRY,
    SCHEDULER_REGISTRY,
    OPTIMIZER_REGISTRY,
    CONFIG_REGISTRY,
    build_from_config
)

__all__ = [
    'MODEL_REGISTRY',
    'LOSS_REGISTRY',
    'METRIC_REGISTRY',
    'SCHEDULER_REGISTRY',
    'OPTIMIZER_REGISTRY',
    'CONFIG_REGISTRY',
    'build_from_config'
]
