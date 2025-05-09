import torch
import torch.nn as nn

from typing import List

from config import FMConfig
from utils.logger import ColorLogger
from utils.register import Registers
from utils.register import build_from_config

logger = ColorLogger(name="FM")


@Registers.model_registry.register
class FM(nn.Module):
    """
    FM: Factorization Machine for CTR prediction.

    Args:
        feature_dims (List[int]): List of feature dimensions.
        dense_feature_dims (int): Dimension of dense features.
        embed_dim (int): Dimension of embedding.
        interact_feature_nums (int): Number of features to interact.
        is_interact (bool): Whether to use interaction features.
    """

    def __init__(
        self,
        feature_dims: List[int],
        dense_feature_dims: int,
        embed_dim: int,
        interact_feature_nums: int,
        is_interact: bool = False,
    ):
        super(FM, self).__init__()
        self.name = "FM"
        self.is_interact = is_interact
        self.interact_feature_nums = interact_feature_nums

        self.embed_nums = feature_dims if is_interact else feature_dims[:-interact_feature_nums]
        self.embeddings = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in self.embed_nums])
        self.dense_layer = nn.Linear(dense_feature_dims, 1)
        self.discrete_layer = nn.ModuleList([nn.Embedding(dim, 1) for dim in self.embed_nums])

        nn.init.xavier_uniform_(self.dense_layer.weight)
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.discrete_layer:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, dense_x: torch.Tensor, discrete_x: torch.Tensor) -> torch.Tensor:
        # FM first order Calculation
        # dense_embed shape's [batch_size, 1]
        dense_out = self.dense_layer(dense_x)
        # discrete_out shape's [batch_size, 1]
        discrete_out = torch.stack(
            [emb(discrete_x[:, i]) for i, emb in enumerate(self.discrete_layer)], dim=1
        ).sum(dim=1)

        # FM second order Calculation
        if not self.is_interact and self.interact_feature_nums > 0:
            discrete_x = discrete_x[:, : -self.interact_feature_nums]
        # embeds shape's [batch_size, num_discrete_features, embed_dim]
        embeds = torch.stack([emb(discrete_x[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
        # sum_of_embeds shape's [batch_size, embed_dim]
        sum_of_embeds = torch.sum(embeds, dim=1)
        # square_of_sum shape's [batch_size, embed_dim]
        square_of_sum = torch.square(sum_of_embeds)
        # sum_of_square shape's [batch_size, embed_dim]
        sum_of_square = torch.sum(torch.square(embeds), dim=1)
        # second_order_output shape's [batch_size, 1]
        second_order_output = torch.sum(0.5 * (square_of_sum - sum_of_square), dim=1, keepdim=True)
        fm_output = dense_out + discrete_out + second_order_output
        return fm_output

    @classmethod
    def from_config(cls, config: FMConfig) -> "FM":
        """Create model from config."""
        return build_from_config(config, Registers.model_registry)
