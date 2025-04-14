from typing import List
from typing import Literal
from typing import Optional
from typing import Union
from pydantic import Field
from pydantic.types import PositiveInt
from pydantic.types import PositiveFloat

from utils.register import Registers
from .base_config import BaseModelConfig
from .base_config import BaseTrainerConfig
from .base_config import OptimizerConfig
from .base_config import SchedulerConfig


@Registers.model_config_registry.register
class FMConfig(BaseModelConfig):
    """Configuration for FM model."""

    Type: Literal["FMCTR"] = "FMCTR"
    feature_dims: List[PositiveInt]
    dense_feature_dim: Optional[PositiveInt] = None
    embed_dim: PositiveInt = 8


FMOptimizerConfig = OptimizerConfig(
    type="Adam",
    learning_rate=1e-3,
    weight_decay=2e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
)

FMSchedulerConfig = SchedulerConfig(
    type="StepLR",
    step_size=5,
    gamma=0.8,
)


@Registers.trainer_config_registry.register
class FMTrainerConfig(BaseTrainerConfig):
    """Configuration for FM trainer."""

    Type: Literal["FMCTR"] = "FMCTR"
    epochs: PositiveInt = 10
    train_batch_size: PositiveInt = 64
    test_batch_size: PositiveInt = 128
    grad_clip: PositiveFloat = 1.0
    device: Union[Literal["cpu", "cuda", "auto"], str, List[str]] = "cuda:0"
    is_scheduler: bool = True

    optimizer: OptimizerConfig = FMOptimizerConfig
    scheduler: SchedulerConfig = FMSchedulerConfig

    class Config:
        validate_assignment = True
