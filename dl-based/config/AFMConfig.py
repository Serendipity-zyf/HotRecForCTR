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
class AFMConfig(BaseModelConfig):
    """Configuration for AFM model."""

    Name: Literal["AFM"] = "AFM"
    feature_dims: List[PositiveInt]
    dense_feature_dims: Optional[PositiveInt] = None
    embed_dim: PositiveInt = 64
    attn_factor_size: PositiveInt = 256
    attn_dropout_rate: PositiveFloat = 0.1
    interact_feature_nums: Optional[PositiveInt] = None
    is_interact: bool = False


AFMOptimizerConfig = OptimizerConfig(
    Name="Adam",
    embed_weight_decay=1e-5,
    dense_weight_decay=2e-5,
    learning_rate=3e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
)


AFMSchedulerConfig = SchedulerConfig(
    Name="StepLR",
    step_size=10,
    gamma=0.9,
)


@Registers.trainer_config_registry.register
class AFMTrainerConfig(BaseTrainerConfig):
    """Configuration for AFM trainer."""

    Name: Literal["AFM"] = "AFM"
    epochs: PositiveInt = 50
    train_batch_size: PositiveInt = 1024
    test_batch_size: PositiveInt = 1024
    grad_clip: PositiveFloat = 1.0
    patience: PositiveInt = 5
    device: Union[Literal["cpu", "cuda", "auto"], str, List[str]] = "cuda:2"
    is_scheduler: bool = True

    optimizer: OptimizerConfig = AFMOptimizerConfig
    scheduler: SchedulerConfig = AFMSchedulerConfig

    class Config:
        validate_assignment = True
