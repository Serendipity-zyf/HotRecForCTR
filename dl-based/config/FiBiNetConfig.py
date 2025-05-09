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
class FiBiNetConfig(BaseModelConfig):
    """Configuration for FiBiNet model."""

    Name: Literal["FiBiNet"] = "FiBiNet"
    feature_dims: List[PositiveInt]
    dense_feature_dims: Optional[PositiveInt] = None
    embed_dim: PositiveInt = 32
    hidden_size: PositiveInt = 400
    reduction_ratio: PositiveInt = 3
    bilinear_type: Literal["field-all", "field-each", "field-interaction"] = "field-all"
    dnn_layers: PositiveInt = 3
    dropout_rate: PositiveFloat = 0.05
    interact_feature_nums: Optional[PositiveInt] = None
    is_interact: bool = False


FiBiNetOptimizerConfig = OptimizerConfig(
    Name="Adam",
    embed_weight_decay=1e-6,
    dense_weight_decay=5e-5,
    learning_rate=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
)


FiBiNetSchedulerConfig = SchedulerConfig(
    Name="StepLR",
    step_size=5,
    gamma=0.9,
)


@Registers.trainer_config_registry.register
class FiBiNetTrainerConfig(BaseTrainerConfig):
    """Configuration for FiBiNet trainer."""

    Name: Literal["FiBiNet"] = "FiBiNet"
    epochs: PositiveInt = 50
    train_batch_size: PositiveInt = 512
    test_batch_size: PositiveInt = 1024
    grad_clip: PositiveFloat = 1.0
    patience: PositiveInt = 5
    device: Union[Literal["cpu", "cuda", "auto"], str, List[str]] = "cuda:0"
    is_scheduler: bool = True

    optimizer: OptimizerConfig = FiBiNetOptimizerConfig
    scheduler: SchedulerConfig = FiBiNetSchedulerConfig

    class Config:
        validate_assignment = True
