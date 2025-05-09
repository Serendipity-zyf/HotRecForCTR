import torch
import torch.nn as nn

from typing import List
from config import WideDeepConfig
from utils.logger import ColorLogger
from utils.register import Registers
from utils.register import build_from_config

logger = ColorLogger(name="WideDeep")


@Registers.model_registry.register
class WideDeep(nn.Module):
    """
    WideDeep: Wide & Deep Learning for Recommender Systems

    Args:
        feature_dims (List[int]): List of feature dimensions.
        dense_feature_dims (int): Dimension of dense features.
        embed_dim (int): Dimension of embedding.
        hidden_size (int): Dimension of hidden layer.
        interact_feature_nums (int): Number of interactive features.
        is_interact (bool): Whether to use interactive features.
    """

    def __init__(
        self,
        feature_dims: List[int],
        dense_feature_dims: int,
        embed_dim: int,
        hidden_size: int,
        dnn_layers: int,
        dropout_rate: float,
        interact_feature_nums: int,
        is_interact: bool = False,
    ):
        super(WideDeep, self).__init__()
        self.name = "WideDeep"
        self.dense_feature_dims = dense_feature_dims
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.is_interact = is_interact
        self.single_feature_nums = len(feature_dims) - interact_feature_nums
        self.interact_feature_nums = interact_feature_nums
        if interact_feature_nums > 0:
            self.single_feature_dims = feature_dims[:-interact_feature_nums]
        else:
            self.single_feature_dims = feature_dims

        # dnn Layers definition
        self.dnn_embeddings = nn.ModuleList(
            [nn.Embedding(dim, embed_dim) for dim in self.single_feature_dims]
        )
        dnn_input_size = embed_dim * self.single_feature_nums + dense_feature_dims
        layers = []
        in_features = dnn_input_size
        for _ in range(dnn_layers):
            out_features = self.hidden_size
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = out_features

        self.dnn = nn.Sequential(*layers)
        self.dnn_output_layer = nn.Linear(in_features, 1)
        # wide Layers definition
        self.wide_embeddings = nn.ModuleList([nn.Embedding(dim, 1) for dim in feature_dims])
        wide_input_size = len(feature_dims)
        self.wide_output_layer = nn.Linear(wide_input_size, 1)

        # initialization
        def init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_weights)

    def forward(self, dense_x: torch.Tensor, discrete_x: torch.Tensor) -> torch.Tensor:
        if self.interact_feature_nums > 0:
            single_x = discrete_x[:, : -self.interact_feature_nums]
        else:
            single_x = discrete_x
        # dnn_embed_x: [batch_size, embed_dim * single_feature_nums]
        dnn_embed_x = torch.concat(
            [dense_x] + [emb(single_x[:, i]) for i, emb in enumerate(self.dnn_embeddings)], dim=-1
        )
        # dnn_out: [batch_size, 1]
        dnn_out = self.dnn(dnn_embed_x)
        dnn_out = self.dnn_output_layer(dnn_out)

        wide_embed_x = torch.concat(
            [emb(discrete_x[:, i]) for i, emb in enumerate(self.wide_embeddings)], dim=-1
        )
        # wide_out: [batch_size, 1]
        wide_out = self.wide_output_layer(wide_embed_x)
        return dnn_out + wide_out

    @classmethod
    def from_config(cls, config: WideDeepConfig) -> "WideDeep":
        """Create model from config."""
        return build_from_config(config, Registers.model_registry)
