"""ExponentialLR scheduler implementation."""

import torch.optim as optim
from typing import Any
from typing import Dict

from utils.register import Registers


@Registers.scheduler_registry.register
class ExponentialLR(optim.lr_scheduler.ExponentialLR):
    """ExponentialLR scheduler implementation."""

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], optimizer: optim.Optimizer
    ) -> "ExponentialLR":
        """Create scheduler from config."""
        return cls(optimizer, gamma=config["gamma"])
