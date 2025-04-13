from typing import List
from typing import Optional
from pydantic import Field
from pydantic.types import PositiveInt
from pydantic.types import PositiveFloat

from utils.register import Registers
from .base_config import BaseModelConfig
from .base_config import BaseTrainerConfig


@Registers.model_config_registry.register
class FMConfig(BaseModelConfig):
    """Configuration for FM model."""

    Type: str = Field(default="FMConfig", const=True)
    feature_dims: List[PositiveInt]
    dense_feature_dim: Optional[PositiveInt] = None
    embed_dim: PositiveInt = 8


@Registers.trainer_config_registry.register
class FMTrainerConfig(BaseTrainerConfig):
    """Configuration for FM trainer."""

    Type: str = Field(default="FMTrainerConfig", const=True)
    epochs: PositiveInt = 10
    train_batch_size: PositiveInt = 64
    test_batch_size: PositiveInt = 128
    learning_rate: PositiveFloat = 1e-3
    weight_decay: float = Field(default=2e-5, ge=0)
    is_scheduler: bool = True
    scheduler_step: Optional[PositiveInt] = 10
    scheduler_gamma: Optional[float] = Field(default=0.5, gt=0)
    device: str = "cpu"
