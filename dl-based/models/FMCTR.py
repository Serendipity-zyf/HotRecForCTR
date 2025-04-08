import torch.nn as nn
from config import FMConfig
from utils.logger import ColorLogger
from utils.register import Registers, build_from_config

logger = ColorLogger(name="FMCTR")


@Registers.model_registry.register
class FMCTR(nn.Module):
    def __init__(self):
        super(FMCTR, self).__init__()
        pass

    def forward(self, x):
        pass

    @classmethod
    def from_config(cls, config: FMConfig) -> "FMCTR":
        """Create model from config."""
        return build_from_config(config, Registers.model_registry)
