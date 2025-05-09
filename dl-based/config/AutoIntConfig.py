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
class AutoIntConfig(BaseModelConfig):
    """Configuration for AutoInt model."""

    Name: Literal["AutoInt"] = "AutoInt"
    feature_dims: List[PositiveInt]
    dense_feature_dims: Optional[PositiveInt] = None
    embed_dim: PositiveInt = 16
    attn_hidden_size: PositiveInt = 64
    num_heads: PositiveInt = 4
    num_atten_layers: PositiveInt = 1
    attn_dropout: PositiveFloat = 0.1
    use_dnn: bool = True
    dnn_layers: PositiveInt = 2
    dnn_hidden_size: PositiveInt = 256
    dnn_dropout: PositiveFloat = 0.1
    embedding_dropout: PositiveFloat = 0.1
    interact_feature_nums: Optional[PositiveInt] = None
    is_interact: bool = False


AutoIntOptimizerConfig = OptimizerConfig(
    Name="Adam",
    embed_weight_decay=1e-4,
    dense_weight_decay=1e-4,
    learning_rate=2e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
)


AutoIntSchedulerConfig = SchedulerConfig(
    Name="StepLR",
    step_size=10,
    gamma=0.9,
)


@Registers.trainer_config_registry.register
class AutoIntTrainerConfig(BaseTrainerConfig):
    """Configuration for AutoInt trainer."""

    Name: Literal["AutoInt"] = "AutoInt"
    epochs: PositiveInt = 100
    train_batch_size: PositiveInt = 256
    test_batch_size: PositiveInt = 1024
    grad_clip: PositiveFloat = 1.0
    patience: PositiveInt = 10
    device: Union[Literal["cpu", "cuda", "auto"], str, List[str]] = "cuda:1"
    is_scheduler: bool = True

    optimizer: OptimizerConfig = AutoIntOptimizerConfig
    scheduler: SchedulerConfig = AutoIntSchedulerConfig

    class Config:
        validate_assignment = True
