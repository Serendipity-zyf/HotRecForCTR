from typing import Any
from typing import Dict
from typing import Optional
from pydantic import BaseModel
from pydantic import Field
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
    learning_rate: PositiveFloat
    weight_decay: float = Field(ge=0)
    scheduler_step: Optional[PositiveInt] = None
    scheduler_gamma: Optional[float] = Field(default=None, gt=0)
    device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.items() if v is not None}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseTrainerConfig":
        """Create a configuration instance from the dictionary"""
        return cls(**config_dict)

    class Config:
        validate_assignment = True
