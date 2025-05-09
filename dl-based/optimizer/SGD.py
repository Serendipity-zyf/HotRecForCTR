"""SGD optimizer implementation with momentum."""

import torch
import torch.optim as optim
from typing import Any
from typing import Dict
from typing import Iterator
from utils.logger import ColorLogger
from utils.register import Registers

logger = ColorLogger(name="SGD")


@Registers.optimizer_registry.register
class SGD(optim.SGD):
    """SGD optimizer implementation with momentum support.

    Stochastic Gradient Descent with optional momentum, weight decay,
    dampening, and Nesterov momentum.
    """

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.9,
        dampening: float = 0,
        nesterov: bool = False,
    ):
        if momentum < 0.0:
            logger.warning(f"Invalid momentum value: {momentum}, setting to 0")
            momentum = 0.0

        if lr <= 0.0:
            raise ValueError(f"Learning rate must be positive, got: {lr}")

        if nesterov and (momentum <= 0 or dampening != 0):
            logger.warning("Nesterov momentum requires positive momentum and zero dampening")

        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            nesterov=nesterov,
        )
        self.name = "SGD"

    @classmethod
    def from_config(cls, params, config: Dict[str, Any]) -> "SGD":
        """Create optimizer from config dictionary."""
        sgd_params = {
            "lr": config.get("learning_rate", 1e-3),
            "momentum": config.get("momentum", 0.9),
            "dampening": config.get("dampening", 0),
            "weight_decay": config.get("weight_decay", 1e-4),
            "nesterov": config.get("nesterov", False),
        }

        return cls(params, **sgd_params)
