import torch
import torch.nn as nn

from typing import List
from config import FiBiNetConfig
from utils.logger import ColorLogger
from utils.register import Registers
from utils.register import build_from_config

logger = ColorLogger(name="FiBiNet")


@Registers.model_registry.register
class FiBiNet(nn.Module):
    """
    FiBiNET: Combining Feature Importance and Bilinear feature
             Interaction for Click-Through Rate Prediction

    Args:
        feature_dims (List[int]): List of feature dimensions.
        dense_feature_dims (int): Dimension of dense features.
        embed_dim (int): Dimension of embedding.
        reduction_ratio (int): Reduction ratio for SENET.
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
        reduction_ratio: int,
        bilinear_type: str,
        dnn_layers: int,
        dropout_rate: float,
        interact_feature_nums: int,
        is_interact: bool = False,
    ):
        super(FiBiNet, self).__init__()
        self.name = "FiBiNet"
        self.dense_feature_dims = dense_feature_dims
        self.embed_dim = embed_dim
        self.reduction_ratio = reduction_ratio
        self.hidden_size = hidden_size
        self.bilinear_type = bilinear_type
        self.is_interact = is_interact
        self.interact_feature_nums = interact_feature_nums
        self.discrete_embed_nums: List[int] = (
            feature_dims if is_interact else feature_dims[:-interact_feature_nums]
        )
        self.discrete_nums = len(self.discrete_embed_nums)
        self.num_features = len(self.discrete_embed_nums) + 1
        self.reduction_size = max(1, len(self.discrete_embed_nums) // reduction_ratio)
        self.pair_nums = self.discrete_nums * (self.discrete_nums - 1) // 2

        # first order Layers definition
        self.dense_layer = nn.Linear(dense_feature_dims, 1)
        self.discrete_layer = nn.ModuleList([nn.Embedding(dim, 1) for dim in self.discrete_embed_nums])

        # second order Layers definition
        self.discrete_embeddings = nn.ModuleList(
            [nn.Embedding(dim, embed_dim) for dim in self.discrete_embed_nums]
        )
        self.SENETLayer = nn.Sequential(
            nn.Linear(self.discrete_nums, self.reduction_size),
            nn.ReLU(),
            nn.Linear(self.reduction_size, self.discrete_nums),
            nn.ReLU(),
        )
        self.bilinear_layer = BilinearInterationLayer(embed_dim, self.discrete_nums, bilinear_type)
        dnn_input_size = self.dense_feature_dims + 2 * self.pair_nums * self.embed_dim
        layers = []
        in_features = dnn_input_size
        for i in range(dnn_layers):
            out_features = self.hidden_size
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = out_features

        self.dnn = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_features, 1)

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
        if not self.is_interact and self.interact_feature_nums > 0:
            discrete_x = discrete_x[:, : -self.interact_feature_nums]
        # embed_x: [batch_size, num_feature, embed_dim]
        embed_x = torch.stack(
            [emb(discrete_x[:, i]) for i, emb in enumerate(self.discrete_embeddings)], dim=1
        )
        # avg_embed_x: [batch_size, num_feature]
        avg_embed_x = torch.mean(embed_x, dim=-1)
        # weights: [batch_size, num_feature]
        weights = self.SENETLayer(avg_embed_x)
        # weighted_embed_x: [batch_size, num_feature, embed_dim]
        weighted_embed_x = embed_x * weights.unsqueeze(-1)
        # bilinear_out: [batch_size, pair_nums, embed_dim]
        bilinear_out_1 = self.bilinear_layer(weighted_embed_x, weighted_embed_x)
        bilinear_out_2 = self.bilinear_layer(embed_x, embed_x)
        # flatten_bilinear_out: [batch_size, 2 * pair_nums * embed_dim]
        flatten_bilinear_out = torch.concat([bilinear_out_1, bilinear_out_2], dim=1).flatten(1)
        # dnn_input: [batch_size, dense_feature_dims + 2 * pair_nums * embed_dim]
        dnn_input = torch.cat([dense_x, flatten_bilinear_out], dim=1)
        # dnn_out: [batch_size, 1]
        dnn_out = self.dnn(dnn_input)
        dnn_out = self.output_layer(dnn_out)

        # firset order Calculation
        # dense_out: [batch_size, 1]
        dense_out = self.dense_layer(dense_x)
        # discrete_out: [batch_size, 1]
        discrete_out = torch.stack(
            [emb(discrete_x[:, i]) for i, emb in enumerate(self.discrete_layer)], dim=1
        ).sum(dim=1)
        return dense_out + discrete_out + dnn_out

    @classmethod
    def from_config(cls, config: FiBiNetConfig) -> "FiBiNet":
        """Create model from config."""
        return build_from_config(config, Registers.model_registry)


class BilinearInterationLayer(nn.Module):
    def __init__(self, embed_dim: int, discrete_nums: int, Type: str = "field-each"):
        super(BilinearInterationLayer, self).__init__()
        self.Type = Type
        self.D = embed_dim
        self.N = discrete_nums
        self.pair_nums = discrete_nums * (discrete_nums - 1) // 2

        if Type == "field-all":
            self.W = nn.Linear(embed_dim, embed_dim, bias=False)
            nn.init.xavier_uniform_(self.W.weight)
        elif Type == "field-each":
            weight = torch.empty(discrete_nums, embed_dim, embed_dim)
            nn.init.xavier_uniform_(weight)
            self.W = nn.Parameter(weight)
        else:
            weight = torch.empty(self.pair_nums, embed_dim, embed_dim)
            nn.init.xavier_uniform_(weight)
            self.W = nn.Parameter(weight)

        # Upper triangular coordinates (i<j)
        idx_i, idx_j = torch.triu_indices(self.N, self.N, offset=1)
        self.register_buffer("idx_i", idx_i, persistent=False)
        self.register_buffer("idx_j", idx_j, persistent=False)

    def forward(self, xi: torch.Tensor, xj: torch.Tensor) -> torch.Tensor:
        """
        xi, xj: B * N * D
        Returns: B * P * D
        """
        if self.Type == "field-all":
            # left_all: B * N * D
            left_all = self.W(xi)
            # left: B * PairNum * D
            left = left_all[:, self.idx_i]
        elif self.Type == "field-each":
            # einsum: (B,N,D) · (N,D,H) -> (B,N,H)    H -> D
            left_all = torch.einsum("b n d , n d h -> b n h", xi, self.W)
            left = left_all[:, self.idx_i]
        else:
            # einsum: (B,P,D) · (P,D,H) -> (B,P,H)    H -> D
            xi_pair = xi[:, self.idx_i]
            left = torch.einsum("b p d , p d h -> b p h", xi_pair, self.W)

        # right: B * PairNum * D
        right = xj[:, self.idx_j]
        return left * right
