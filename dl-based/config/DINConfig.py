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
    user_num: PositiveInt
    item_num: PositiveInt
    cate_num: PositiveInt
    embed_dim: PositiveInt = 8
    hidden_size: PositiveInt = 200
    unit_hidden_size: PositiveInt = 36
    dense_nums: PositiveInt = 2
    use_attn: bool = True


DINOptimizerConfig = OptimizerConfig(
    Name="Adam",
    embed_weight_decay=0.0,
    dense_weight_decay=1e-5,
    learning_rate=3e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
)


# DINSchedulerConfig = SchedulerConfig(
#     Name="StepLR",
#     step_size=5,
#     gamma=0.9,
# )
DINSchedulerConfig = SchedulerConfig(
    Name="ReduceLROnPlateau",
    mode="min",
    factor=0.5,
    patience=1,
    threshold=1e-3,
    min_lr=1e-6,  # Explicitly set min_lr to avoid None value
)


@Registers.trainer_config_registry.register
class DINTrainerConfig(BaseTrainerConfig):
    """Configuration for DIN trainer."""

    Name: Literal["DIN"] = "DIN"
    epochs: PositiveInt = 50
    train_batch_size: PositiveInt = 256
    test_batch_size: PositiveInt = 1024
    grad_clip: PositiveFloat = 5.0
    patience: PositiveInt = 5
    device: Union[Literal["cpu", "cuda", "auto"], str, List[str]] = "cuda:1"
    is_scheduler: bool = True

    optimizer: OptimizerConfig = DINOptimizerConfig
    scheduler: SchedulerConfig = DINSchedulerConfig

    class Config:
        validate_assignment = True
