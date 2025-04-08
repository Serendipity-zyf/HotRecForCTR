from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class BaseModelConfig:
    """Base configuration class."""

    embed_dim: int = 10
    dropout_rate: float = 0.1
    hidden_size: int = 128

    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseModelConfig":
        """Create a configuration instance from the dictionary"""
        instance = cls(**config_dict)
        instance.validate()
        return instance


@dataclass
class BaseTrainerConfig:
    """Base configuration class for trainer."""

    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    scheduler_step: int = 10
    scheduler_gamma: float = 0.5
    device: str = "cuda:0"

    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseTrainerConfig":
        """Create a configuration instance from the dictionary"""
        instance = cls(**config_dict)
        instance.validate()
        return instance