"""ReduceLROnPlateau scheduler implementation."""

import torch

from typing import Any
from typing import Dict
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.register import Registers


@Registers.scheduler_registry.register
class ReduceLROnPlateau(ReduceLROnPlateau):
    """ReduceLROnPlateau scheduler implementation.

    Reduces learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: float = 0,
        eps: float = 1e-8,
        verbose: bool = False,
    ):
        # Ensure min_lr is properly set for all parameter groups
        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(f"Expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}")
        else:
            min_lr = [min_lr] * len(optimizer.param_groups)

        super().__init__(
            optimizer=optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
            verbose=verbose,
        )
        self.name = "ReduceLROnPlateau"

    @classmethod
    def from_config(cls, config: Dict[str, Any], optimizer: torch.optim.Optimizer) -> "ReduceLROnPlateau":
        """Create scheduler from config."""
        # Only select the parameters required by the ReduceLROnPlateau scheduler.
        scheduler_params = {
            "mode": config.get("mode", "min"),
            "factor": config.get("factor", 0.1),
            "patience": config.get("patience", 10),
            "threshold": config.get("threshold", 1e-4),
            "threshold_mode": config.get("threshold_mode", "rel"),
            "cooldown": config.get("cooldown", 0),
            "min_lr": config.get("min_lr", 0),
            "eps": config.get("eps", 1e-8),
            "verbose": config.get("verbose", False),
        }
        # Ensure min_lr is not None
        if scheduler_params["min_lr"] is None:
            scheduler_params["min_lr"] = 0
        return cls(optimizer, **scheduler_params)
