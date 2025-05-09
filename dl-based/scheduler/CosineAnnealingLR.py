"""CosineAnnealingLR scheduler implementation."""

import torch.optim as optim
from typing import Any
from typing import Dict

from utils.register import Registers


@Registers.scheduler_registry.register
class CosineAnnealingLR(optim.lr_scheduler.CosineAnnealingLR):
    """CosineAnnealingLR scheduler implementation."""

    def __init__(self, optimizer, T_max, eta_min):
        super().__init__(optimizer, T_max, eta_min)
        self.name = "CosineAnnealingLR"

    @classmethod
    def from_config(cls, config: Dict[str, Any], optimizer: optim.Optimizer) -> "CosineAnnealingLR":
        """Create scheduler from config."""
        params = {"T_max": config["T_max"], "eta_min": config["eta_min"]}
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        return cls(optimizer, **params)
