import torch
import torch.nn as nn

from typing import List

from config import FMConfig
from utils.logger import ColorLogger
from utils.register import Registers
from utils.register import build_from_config

logger = ColorLogger(name="FMCTR")


@Registers.model_registry.register
class FMCTR(nn.Module):
    """
    Factorization Machine for CTR prediction.

    Args:
        feature_dims (List[int]): List of feature dimensions.
        dense_feature_dim (int): Dimension of dense features.
        embed_dim (int): Dimension of embedding.
    """

    def __init__(self, feature_dims: List[int], dense_feature_dim: int, embed_dim: int):
        super(FMCTR, self).__init__()
        self.name = "FMCTR"
        self.embeddings = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in feature_dims])
        self.dense_layer = nn.Linear(dense_feature_dim, embed_dim)

    def forward(self, dense_x: torch.Tensor, discrete_x: torch.Tensor) -> torch.Tensor:
        # discrete_embeds shape's [batch_size, num_discrete_features, embed_dim]
        discret_embeds = torch.stack(
            [emb(discrete_x[:, i]) for i, emb in enumerate(self.embeddings)], dim=1
        )
        # dense_embed shape's [batch_size, embed_dim]
        dense_embed = self.dense_layer(dense_x)
        # embeds shape's [batch_size, num_discrete_features + 1, embed_dim]
        embeds = torch.cat([discret_embeds, dense_embed.unsqueeze(1)], dim=1)
        # FM Calculation
        # sum_of_embeds shape's [batch_size, embed_dim]
        sum_of_embeds = torch.sum(embeds, dim=1)
        # square_of_sum shape's [batch_size, embed_dim]
        square_of_sum = torch.square(sum_of_embeds)
        # sum_of_square shape's [batch_size, embed_dim]
        sum_of_square = torch.sum(torch.square(embeds), dim=1)
        # fm_output shape's [batch_size]
        fm_output = torch.sum(0.5 * (square_of_sum - sum_of_square), dim=1)
        return fm_output

    @classmethod
    def from_config(cls, config: FMConfig) -> "FMCTR":
        """Create model from config."""
        return build_from_config(config, Registers.model_registry)
