import torch
import torch.nn as nn

from typing import List
from config import DINConfig
from utils.logger import ColorLogger
from utils.register import Registers
from utils.register import build_from_config

logger = ColorLogger(name="DIN")


@Registers.model_registry.register
class DIN(nn.Module):
    """
    DIN: Deep Interest Network for Click-Through Rate Prediction

    Args:

    """

    def __init__(
        self,
    ):
        super(DIN, self).__init__()
        self.name = "DIN"

        # initialization
        def init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_weights)

    def forward(self) -> torch.Tensor:
        pass

    @classmethod
    def from_config(cls, config: DINConfig) -> "DIN":
        """Create model from config."""
        return build_from_config(config, Registers.model_registry)
