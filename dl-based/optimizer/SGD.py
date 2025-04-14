"""SGD optimizer implementation."""

import torch
import torch.optim as optim
from typing import Any
from typing import Dict
from typing import Iterator

from utils.register import Registers


@Registers.optimizer_registry.register
class SGD(optim.SGD):
    """SGD optimizer implementation."""

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
    ):
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    @classmethod
    def from_config(cls, params, config: Dict[str, Any]) -> "SGD":
        """Create optimizer from config."""
        sgd_params = {
            "lr": config["learning_rate"],
            "momentum": config["momentum"],
            "dampening": config["dampening"],
            "weight_decay": config["weight_decay"],
            "nesterov": config["nesterov"],
        }
        # Remove None values
        sgd_params = {k: v for k, v in sgd_params.items() if v is not None}
        return cls(params, **sgd_params)
