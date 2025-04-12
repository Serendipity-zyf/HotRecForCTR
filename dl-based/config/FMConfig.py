from typing import List
from typing import Optional
from dataclasses import field
from dataclasses import dataclass

from utils.register import Registers
from .base_config import BaseModelConfig
from .base_config import BaseTrainerConfig


@dataclass
@Registers.model_config_registry.register
class FMConfig(BaseModelConfig):
    """Configuration for FM model."""

    type: str = "FMConfig"
    feature_dims: List[int] = field(default_factory=list)
    dense_feature_dim: Optional[int] = None
    embed_dim: int = 8


@dataclass
@Registers.trainer_config_registry.register
class FMTrainerConfig(BaseTrainerConfig):
    """Configuration for FM trainer."""

    type: str = "FMTrainerConfig"
    epochs: int = 10
    train_batch_size: int = 64
    test_batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    scheduler_step: Optional[int] = 10
    scheduler_gamma: Optional[float] = 0.5
    device: str = "cpu"
