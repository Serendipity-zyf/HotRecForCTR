"""Adam optimizer implementation."""

import torch
import torch.optim as optim
from typing import Any
from typing import Dict
from typing import Iterator

from utils.register import Registers


@Registers.optimizer_registry.register
class Adam(optim.Adam):
    """Adam optimizer implementation."""

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
        )
        self.name = "Adam"

    @classmethod
    def from_config(cls, params, config: Dict[str, Any]) -> "Adam":
        """Create optimizer from config."""
        # Only select the parameters required by the Adam optimizer.
        adam_params = {
            "lr": config["learning_rate"],
            "betas": config["betas"],
            "eps": config["eps"],
        }
        return cls(params, **adam_params)
