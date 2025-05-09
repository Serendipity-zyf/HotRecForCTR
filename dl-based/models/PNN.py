import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

from config import PNNConfig
from utils.logger import ColorLogger
from utils.register import Registers
from utils.register import build_from_config

logger = ColorLogger(name="PNN")


@Registers.model_registry.register
class PNN(nn.Module):
    """
    Product-based Neural Networks for CTR prediction.

    Args:
        feature_dims (List[int]): List of feature dimensions.
        dense_feature_dims (int): Dimension of dense features.
        embed_dim (int): Dimension of embedding.
    """

    def __init__(
        self,
        feature_dims: List[int],
        dense_feature_dims: int,
        embed_dim: int,
        hidden_size: int,
        units: int,
        interact_feature_nums: int,
        is_interact: bool = False,
        is_inner: bool = True,
        is_outer: bool = True,
    ):
        super(PNN, self).__init__()
        self.name = "PNN"
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.units = units
        self.is_interact = is_interact
        self.interact_feature_nums = interact_feature_nums
        self.is_inner = is_inner
        self.is_outer = is_outer
        self.embed_nums = feature_dims if is_interact else feature_dims[:-interact_feature_nums]
        self.num_features = len(self.embed_nums) + 1

        # Layers definition
        self.dense_layer = nn.Linear(dense_feature_dims, embed_dim)
        self.embeddings = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in self.embed_nums])
        self.W_z = nn.Linear(self.num_features * embed_dim, units, bias=False)
        self.W_ip = torch.nn.Parameter(torch.randn(units, self.num_features))
        self.W_op = torch.nn.Parameter(torch.randn(self.embed_dim**2, units))

        units_num = 3 if self.is_inner and self.is_outer else 2
        self.hidden_layer_1 = nn.Linear(units * units_num, hidden_size)
        self.hidden_layer_2 = nn.Linear(hidden_size, hidden_size // 2)
        self.output_layer = nn.Linear(hidden_size // 2, 1)

        nn.init.xavier_uniform_(self.dense_layer.weight)
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, dense_x: torch.Tensor, discrete_x: torch.Tensor) -> torch.Tensor:
        batch_size = dense_x.size(0)
        # dense_embed shape's [batch_size, embed_dim]
        dense_embed = self.dense_layer(dense_x)
        # discrete_embeds shape's [batch_size, num_discrete_features, embed_dim]
        if not self.is_interact and self.interact_feature_nums > 0:
            discrete_x = discrete_x[:, : -self.interact_feature_nums]
        discret_embeds = torch.stack([emb(discrete_x[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
        # z shape's [batch_size, num_features, embed_dim]
        z = torch.cat([discret_embeds, dense_embed.unsqueeze(1)], dim=1)
        # flatten_z shape's [batch_size, num_features * embed_dim]
        flatten_z = z.reshape(batch_size, -1)
        pnn_out_list = []
        # PNN Calculation
        out_1 = self.W_z(flatten_z)  # out_1 shape's [batch_size, units]
        pnn_out_list.append(out_1)
        # IPNN Calculation
        if not self.is_inner and not self.is_outer:
            raise ValueError("At least one of self.inner and self.outer must be true.")
        if self.is_inner:
            # res_inner shape's [batch_size, units, num_features, embed_dim]
            res_inner = torch.einsum("b i j, k i-> b k i j", z, self.W_ip)
            # res_inner_sum shape's [batch_size, units, num_features]
            res_inner_sum = torch.sum(res_inner, dim=-1)
            out_2 = (res_inner_sum**2).sum(dim=-1)  # out_2 shape's [batch_size, units]
            pnn_out_list.append(out_2)
        if self.is_outer:
            # z_sum shape's [batch_size, embed_dim]
            z_sum = torch.sum(z, dim=1)
            # res_outer shape's [batch_size, embed_dim ** 2]
            res_outer = torch.einsum("b i, b j -> b i j", z_sum, z_sum).reshape(batch_size, -1)
            out_3 = torch.matmul(res_outer, self.W_op)  # out_3 shape's [batch_size, units]
            pnn_out_list.append(out_3)
        # out shape's [batch_size, units * j] j = 2 or 3
        out = torch.cat(pnn_out_list, dim=1)
        out = F.relu(self.hidden_layer_1(out))
        out = F.relu(self.hidden_layer_2(out))
        pnn_out = self.output_layer(out)
        return pnn_out

    @classmethod
    def from_config(cls, config: PNNConfig) -> "PNN":
        """Create model from config."""
        return build_from_config(config, Registers.model_registry)
