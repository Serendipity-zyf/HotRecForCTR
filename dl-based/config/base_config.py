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


class BaseModelConfig(BaseModel):
    """Base configuration class."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.items() if v is not None}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseModelConfig":
        """Create a configuration instance from the dictionary"""
        return cls(**config_dict)

    class Config:
        validate_assignment = True


class BaseTrainerConfig(BaseModel):
    """Base configuration class for trainer."""

    epochs: PositiveInt
    train_batch_size: PositiveInt
    test_batch_size: PositiveInt
    grad_clip: PositiveFloat
    is_scheduler: bool
    device: Union[Literal["cpu", "cuda", "auto"], str, List[str]] = "cpu"

    learning_rate: PositiveFloat
    weight_decay: float = Field(ge=0)
    scheduler_step: Optional[PositiveInt]
    scheduler_gamma: Optional[float] = Field(default=None, gt=0)

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
                    raise ValueError(
                        'Multi-GPU format must be ["cuda:0", "cuda:1", ...]'
                    )
            return v
        raise ValueError(
            'device must be "cpu", "cuda", "auto", "cuda:N" or ["cuda:0", "cuda:1", ...]'
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.items() if v is not None}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseTrainerConfig":
        """Create a configuration instance from the dictionary"""
        return cls(**config_dict)

    class Config:
        validate_assignment = True
