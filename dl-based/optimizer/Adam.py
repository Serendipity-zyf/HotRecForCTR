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
        weight_decay: float = 0,
    ):
        """Initialize the Adam optimizer.

        Args:
            params: Parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages of gradient and its square
            eps: Term added to the denominator to improve numerical stability
            weight_decay: Weight decay (L2 penalty)
        """
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Adam":
        """Create optimizer from config."""
        return cls(**config)
