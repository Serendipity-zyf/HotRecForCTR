"""StepLR scheduler implementation."""

import torch.optim as optim
from typing import Any
from typing import Dict

from utils.register import Registers


@Registers.scheduler_registry.register
class StepLR(optim.lr_scheduler.StepLR):
    """StepLR scheduler implementation."""

    def __init__(self, optimizer, step_size, gamma):
        super().__init__(optimizer, step_size, gamma)
        self.name = "StepLR"

    @classmethod
    def from_config(cls, config: Dict[str, Any], optimizer: optim.Optimizer) -> "StepLR":
        """Create scheduler from config."""
        return cls(optimizer, step_size=config["step_size"], gamma=config["gamma"])
