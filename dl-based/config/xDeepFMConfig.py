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
class xDeepFMConfig(BaseModelConfig):
    """Configuration for xDeepFM model."""

    Name: Literal["xDeepFM"] = "xDeepFM"
    feature_dims: List[PositiveInt]
    dense_feature_dims: Optional[PositiveInt] = None
    embed_dim: PositiveInt = 10
    hidden_size: PositiveInt = 400
    num_dnn_layers: PositiveInt = 2
    dnn_dropout: PositiveFloat = 0.1
    num_cin_layers: PositiveInt = 3
    feature_maps: PositiveInt = 200
    interact_feature_nums: Optional[PositiveInt] = None
    is_interact: bool = False


xDeepFMOptimizerConfig = OptimizerConfig(
    type="Adam",
    embed_weight_decay=2e-5,
    dense_weight_decay=2e-5,
    learning_rate=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
)

xDeepFMSchedulerConfig = SchedulerConfig(
    type="StepLR",
    step_size=10,
    gamma=0.9,
)


@Registers.trainer_config_registry.register
class xDeepFMTrainerConfig(BaseTrainerConfig):
    """Configuration for xDeepFM trainer."""

    Name: Literal["xDeepFM"] = "xDeepFM"
    epochs: PositiveInt = 100
    train_batch_size: PositiveInt = 512
    test_batch_size: PositiveInt = 1024
    grad_clip: PositiveFloat = 1.0
    patience: PositiveInt = 5
    device: Union[Literal["cpu", "cuda", "auto"], str, List[str]] = "cuda:1"
    is_scheduler: bool = True

    optimizer: OptimizerConfig = xDeepFMOptimizerConfig
    scheduler: SchedulerConfig = xDeepFMSchedulerConfig

    class Config:
        validate_assignment = True
