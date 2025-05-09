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
class DINConfig(BaseModelConfig):
    """Configuration for DIN model."""

    Name: Literal["DIN"] = "DIN"
    feature_dims: List[PositiveInt]
    dense_feature_dims: Optional[PositiveInt] = None
    embed_dim: PositiveInt = 16
    hidden_size: PositiveInt = 400
    dnn_layers: PositiveInt = 3
    dropout_rate: PositiveFloat = 0.0
    interact_feature_nums: Optional[PositiveInt] = None
    is_interact: bool = True


DINOptimizerConfig = OptimizerConfig(
    Name="Adam",
    embed_weight_decay=1e-6,
    dense_weight_decay=5e-5,
    learning_rate=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
)


DINSchedulerConfig = SchedulerConfig(
    Name="StepLR",
    step_size=5,
    gamma=0.9,
)


@Registers.trainer_config_registry.register
class DINTrainerConfig(BaseTrainerConfig):
    """Configuration for DIN trainer."""

    Name: Literal["DIN"] = "DIN"
    epochs: PositiveInt = 50
    train_batch_size: PositiveInt = 2
    test_batch_size: PositiveInt = 2
    grad_clip: PositiveFloat = 1.0
    patience: PositiveInt = 5
    device: Union[Literal["cpu", "cuda", "auto"], str, List[str]] = "cuda:0"
    is_scheduler: bool = True

    optimizer: OptimizerConfig = DINOptimizerConfig
    scheduler: SchedulerConfig = DINSchedulerConfig

    class Config:
        validate_assignment = True
