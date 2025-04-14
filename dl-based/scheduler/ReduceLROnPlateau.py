"""ReduceLROnPlateau scheduler implementation."""

import torch.optim as optim
from typing import Any
from typing import Dict

from utils.register import Registers


@Registers.scheduler_registry.register
class ReduceLROnPlateau(optim.lr_scheduler.ReduceLROnPlateau):
    """ReduceLROnPlateau scheduler implementation."""

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], optimizer: optim.Optimizer
    ) -> "ReduceLROnPlateau":
        """Create scheduler from config."""
        return cls(
            optimizer,
            mode=config["mode"],
            factor=config["factor"],
            patience=config["patience"],
        )
