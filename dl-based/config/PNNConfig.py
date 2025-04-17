from typing import List
from typing import Literal
from typing import Optional
from typing import Union
from pydantic.types import PositiveInt
from pydantic.types import PositiveFloat

from utils.register import Registers
from .base_config import BaseModelConfig
from .base_config import BaseTrainerConfig
from .base_config import OptimizerConfig
from .base_config import SchedulerConfig


@Registers.model_config_registry.register
class PNNConfig(BaseModelConfig):
    """Configuration for PNN model."""

    Name: Literal["PNN"] = "PNN"
    feature_dims: List[PositiveInt]
    dense_feature_dim: Optional[PositiveInt] = None
    embed_dim: PositiveInt = 8
    hidden_size: PositiveInt = 64
    units: PositiveInt = 32
    is_inner: bool = True
    is_outer: bool = True


PNNOptimizerConfig = OptimizerConfig(
    type="Adam",
    learning_rate=1e-3,
    weight_decay=2e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
)

PNNSchedulerConfig = SchedulerConfig(
    type="StepLR",
    step_size=10,
    gamma=0.9,
)


@Registers.trainer_config_registry.register
class PNNTrainerConfig(BaseTrainerConfig):
    """Configuration for PNN trainer."""

    Name: Literal["PNN"] = "PNN"
    epochs: PositiveInt = 30
    train_batch_size: PositiveInt = 512
    test_batch_size: PositiveInt = 1024
    grad_clip: PositiveFloat = 1.0
    device: Union[Literal["cpu", "cuda", "auto"], str, List[str]] = "cuda:0"
    is_scheduler: bool = True

    optimizer: OptimizerConfig = PNNOptimizerConfig
    scheduler: SchedulerConfig = PNNSchedulerConfig

    class Config:
        validate_assignment = True
