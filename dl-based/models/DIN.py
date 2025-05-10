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
        user_num: int,
        item_num: int,
        embed_dim: int,
        hidden_size: int,
        dropout_rate: float,
    ):
        super(DIN, self).__init__()
        self.name = "DIN"

        self.user_embedding = nn.Embedding(user_num, embed_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(item_num, embed_dim, padding_idx=0)

        # initialization
        def init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_weights)

    def forward(
        self,
        uid_idx: torch.Tensor,
        iid_idx: torch.Tensor,
        seq_idx: torch.Tensor,
        mask: torch.Tensor,
        dense: torch.Tensor,
    ) -> torch.Tensor:
        # user_embed: [batch_size, embed_dim]
        user_embed = self.user_embedding(uid_idx)
        # item_embed: [batch_size, embed_dim]
        item_embed = self.item_embedding(iid_idx)
        # seq_embed: [batch_size, seq_len, embed_dim]
        seq_embed = self.item_embedding(seq_idx)

    @classmethod
    def from_config(cls, config: DINConfig) -> "DIN":
        """Create model from config."""
        return build_from_config(config, Registers.model_registry)
