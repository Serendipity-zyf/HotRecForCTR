from typing import Any
from typing import Dict
from typing import Optional
from dataclasses import dataclass


@dataclass
class BaseModelConfig:
    """Base configuration class."""

    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls: dataclass, config_dict: Dict[str, Any]) -> "BaseModelConfig":
        """Create a configuration instance from the dictionary"""
        return cls(**config_dict)


@dataclass
class BaseTrainerConfig:
    """Base configuration class for trainer."""

    epochs: int
    train_batch_size: int
    test_batch_size: int
    learning_rate: float
    weight_decay: float
    scheduler_step: Optional[int]
    scheduler_gamma: Optional[float]
    device: str

    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls: dataclass, config_dict: Dict[str, Any]) -> "BaseTrainerConfig":
        """Create a configuration instance from the dictionary"""
        return cls(**config_dict)
