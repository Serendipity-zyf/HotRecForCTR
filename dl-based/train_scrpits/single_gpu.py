import torch
import torch.nn as nn
from utils.logger import ColorLogger
from utils.register import Registers
from utils.register import build_from_config

logger = ColorLogger(name="TrainScript")


@Registers.train_script_registry.register
class SingleGPUTrainScript(object):
    """Single GPU / CPU training script."""

    def __init__(
        self,
        epochs: int,
        train_batch_size: int,
        test_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        is_scheduler: bool,
        scheduler_step: int,
        scheduler_gamma: float,
        device: str,
    ):
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.is_scheduler = is_scheduler
        self.scheduler_step = scheduler_step
        self.scheduler_gamma = scheduler_gamma
        self.device = device

    def train(self):
        pass

    @classmethod
    def from_config(cls, config: dict) -> "SingleGPUTrainScript":
        """Create model from config."""
        return build_from_config(config, Registers.train_script_registry)
