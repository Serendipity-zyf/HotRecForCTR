import torch
import torch.nn as nn

from typing import List
from config import DCNConfig
from utils.logger import ColorLogger
from utils.register import Registers
from utils.register import build_from_config

logger = ColorLogger(name="DCNv2")


@Registers.model_registry.register
class DCNv2(nn.Module):
    """
    Deep & Cross Network Version 2 for CTR prediction.

    Args:
        feature_dims (List[int]): List of feature dimensions.
        dense_feature_dims (int): Dimension of dense features.
        embed_dim (int): Dimension of embedding.
        hidden_size (int): Dimension of hidden layer.
        num_dnn_layers (int): Number of DNN layers.
        num_cross_layers (int): Number of cross layers.
        interact_feature_nums (int): Number of interactive features.
        is_interact (bool): Whether to use interactive features.
    """

    def __init__(
        self,
        feature_dims: List[int],
        dense_feature_dims: int,
        embed_dim: int,
        hidden_size: int,
        num_dnn_layers: int,
        num_cross_layers: int,
        interact_feature_nums: int,
        is_interact: bool = False,
    ):
        super(DCNv2, self).__init__()
        self.name = "DCNv2"
        self.dense_feature_dims = dense_feature_dims
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.is_interact = is_interact
        self.num_dnn_layers = num_dnn_layers
        self.num_cross_layers = num_cross_layers
        self.interact_feature_nums = interact_feature_nums
        self.embed_nums: List[int] = (
            feature_dims if is_interact else feature_dims[:-interact_feature_nums]
        )
        self.final_embed_dims = embed_dim * len(self.embed_nums) + dense_feature_dims

        # Layers definition
        self.embeddings = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in self.embed_nums])
        layers = []
        input_dim = self.final_embed_dims
        for i in range(num_dnn_layers):
            output_dim = hidden_size // (2**i)
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim

        self.dnn = nn.Sequential(*layers)
        self.cross = nn.ModuleList(
            [CrossLayer(self.final_embed_dims) for _ in range(num_cross_layers)]
        )
        self.output_layer = nn.Linear(
            hidden_size // (2 ** (num_dnn_layers - 1)) + self.final_embed_dims, 1
        )

        # initialization
        for i in range(num_dnn_layers):
            nn.init.xavier_uniform_(self.dnn[i * 2].weight)
            nn.init.zeros_(self.dnn[i * 2].bias)
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, dense_x: torch.Tensor, discrete_x: torch.Tensor) -> torch.Tensor:
        # x_embeds shape's [batch_size, embed_dim * self.num_features]
        if not self.is_interact and self.interact_feature_nums > 0:
            discrete_x = discrete_x[:, : -self.interact_feature_nums]
        x_embeds = torch.cat(
            [emb(discrete_x[:, i]) for i, emb in enumerate(self.embeddings)] + [dense_x], dim=1
        )
        # dense_output shape's [batch_size, hidden_size // (2**num_dnn_layers)]
        dense_output = self.dnn(x_embeds)
        # cross_output shape's [batch_size, self.final_embed_dims]
        cross_output = x_embeds.clone()
        for i in range(self.num_cross_layers):
            cross_output = self.cross[i](x_embeds, cross_output)
        # output shape's [batch_size, hidden_size // (2**num_dnn_layers) + self.final_embed_dims]
        out = torch.cat([dense_output, cross_output], dim=1)
        out = self.output_layer(out)
        return out

    @classmethod
    def from_config(cls, config: DCNConfig) -> "DCNv2":
        """Create model from config."""
        return build_from_config(config, Registers.model_registry)


class CrossLayer(nn.Module):
    """
    Cross Layer for DCNv2 model.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(input_dim, input_dim, bias=True)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)

    def forward(self, x0: torch.Tensor, xj: torch.Tensor) -> torch.Tensor:
        return x0 * self.W(xj) + xj
