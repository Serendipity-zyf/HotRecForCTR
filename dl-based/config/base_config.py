from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Union
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic.types import PositiveInt
from pydantic.types import PositiveFloat
from pydantic import ValidationInfo


class BaseModelConfig(BaseModel):
    """Base configuration class."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.model_dump().items() if v is not None}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseModelConfig":
        """Create a configuration instance from the dictionary"""
        return cls(**config_dict)

    class Config:
        validate_assignment = True


class OptimizerConfig(BaseModel):
    """Base configuration class for optimizer."""

    Name: Literal["SGD", "Adam", "AdamW", "RMSprop"] = "Adam"
    embed_weight_decay: float = Field(default=1e-6, ge=0)
    dense_weight_decay: float = Field(default=5e-5, ge=0)
    learning_rate: PositiveFloat = 1e-3

    # Adam/AdamW parameters
    betas: Optional[tuple[float, float]] = (0.9, 0.999)
    eps: Optional[float] = 1e-8

    # SGD parameters
    momentum: Optional[float] = Field(default=None, ge=0, le=1)
    dampening: Optional[float] = Field(default=None, ge=0)
    nesterov: Optional[bool] = None

    # RMSprop parameters
    alpha: Optional[float] = Field(default=None, ge=0)
    centered: Optional[bool] = None

    @field_validator("*")
    @classmethod
    def validate_params(cls, v: Any, info: ValidationInfo) -> Any:
        field = info.field_name
        values = info.data

        if field != "Name" and v is not None:
            required_params = {
                "SGD": ["learning_rate", "weight_decay"],
                "Adam": ["learning_rate", "weight_decay", "betas", "eps"],
                "AdamW": ["learning_rate", "weight_decay", "betas", "eps"],
                "RMSprop": ["learning_rate", "weight_decay", "alpha", "eps"],
            }

            if values.get("Name") in required_params:
                required = required_params[values["Name"]]
                if field in required and v is None:
                    raise ValueError(f"Parameter '{field}' is required for optimizer '{values['Name']}'")
        return v

    class Config:
        validate_assignment = True


class SchedulerConfig(BaseModel):
    """Base configuration class for scheduler."""

    Name: Literal["StepLR", "ExponentialLR", "CosineAnnealingLR", "ReduceLROnPlateau"] = "StepLR"

    # StepLR parameters
    step_size: PositiveInt = 10
    gamma: float = Field(default=0.8, gt=0)

    # CosineAnnealingLR parameters
    T_max: Optional[PositiveInt] = None
    eta_min: Optional[float] = Field(default=None, ge=0)

    # ReduceLROnPlateau parameters
    mode: Optional[Literal["min", "max"]] = None
    factor: Optional[float] = Field(default=None, gt=0)
    patience: Optional[PositiveInt] = None
    min_lr: Optional[float] = Field(default=None, ge=0)

    @field_validator("*")
    @classmethod
    def validate_params(cls, v: Any, info: ValidationInfo) -> Any:
        field = info.field_name
        values = info.data

        # If type is being set, update Name to match

        if field != "Name" and v is not None:
            required_params = {
                "StepLR": ["step_size", "gamma"],
                "ExponentialLR": ["gamma"],
                "CosineAnnealingLR": ["T_max"],
                "ReduceLROnPlateau": ["mode", "factor", "patience", "min_lr"],
            }

            if values.get("Name") in required_params:
                required = required_params[values["Name"]]
                if field in required and v is None:
                    raise ValueError(f"Parameter '{field}' is required for scheduler '{values['Name']}'")
        return v

    class Config:
        validate_assignment = True


class BaseTrainerConfig(BaseModel):
    """Base configuration class for trainer."""

    epochs: PositiveInt
    train_batch_size: PositiveInt
    test_batch_size: PositiveInt
    grad_clip: PositiveFloat
    patience: PositiveInt
    device: Union[Literal["cpu", "cuda", "auto"], str, List[str]]
    is_scheduler: bool

    optimizer: OptimizerConfig
    scheduler: SchedulerConfig

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: Union[str, List[str]]) -> Union[str, List[str]]:
        if v in ["cpu", "cuda", "auto"]:
            return v
        if isinstance(v, str) and v.startswith("cuda:") and v[5:].isdigit():
            return v
        if isinstance(v, list):
            for device in v:
                if not (device.startswith("cuda:") and device[5:].isdigit()):
                    raise ValueError('Multi-GPU format must be ["cuda:0", "cuda:1", ...]')
            return v
        raise ValueError('device must be "cpu", "cuda", "auto", "cuda:N" or ["cuda:0", "cuda:1", ...]')

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.model_dump().items() if v is not None}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseTrainerConfig":
        """Create a configuration instance from the dictionary"""
        return cls(**config_dict)

    class Config:
        validate_assignment = True
