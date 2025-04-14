"""RMSprop optimizer implementation."""

import torch
import torch.optim as optim
from typing import Any, Dict, Iterator

from utils.register import Registers


@Registers.optimizer_registry.register
class RMSprop(optim.RMSprop):
    """RMSprop optimizer implementation."""

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr: float = 1e-3,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum: float = 0,
        centered: bool = False,
    ):
        super().__init__(
            params,
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
        )
        self.name = "RMSprop"

    @classmethod
    def from_config(cls, params, config: Dict[str, Any]) -> "RMSprop":
        """Create optimizer from config."""
        rmsprop_params = {
            "lr": config["learning_rate"],
            "alpha": config["alpha"],
            "eps": config["eps"],
            "weight_decay": config["weight_decay"],
            "momentum": config["momentum"],
            "centered": config["centered"],
        }
        # Remove None values
        rmsprop_params = {k: v for k, v in rmsprop_params.items() if v is not None}
        return cls(params, **rmsprop_params)
