from dataclasses import dataclass

from utils.register import Registers

from .base_config import BaseModelConfig, BaseTrainerConfig


@Registers.config_registry.register
@dataclass
class FMConfig(BaseModelConfig):
    """Configuration for FM model."""

    type: str = "FMCTR"
    embed_dim: int = 10
    dropout_rate: float = 0.1
    hidden_size: int = 128


@Registers.config_registry.register
@dataclass
class FMTrainerConfig(BaseTrainerConfig):
    """Configuration for FM trainer."""

    type: str = "FMCTR"
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    scheduler_step: int = 10
