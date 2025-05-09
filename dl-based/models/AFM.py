import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from config import AFMConfig
from utils.logger import ColorLogger
from utils.register import Registers
from utils.register import build_from_config

logger = ColorLogger(name="AFM")


@Registers.model_registry.register
class AFM(nn.Module):
    """
    AFM: Attentional Factorization Machines: Learning the Weight of Feature
         Interactions via Attention Networks for CTR Prediction

    Args:
        feature_dims (List[int]): List of feature dimensions.
        dense_feature_dims (int): Dimension of dense features.
        embed_dim (int): Dimension of embedding.
        attn_factor_size (int): Size of attention factors.
        dropout_rate (float): Pair wise layer dropout rate.
        interact_feature_nums (int): Number of interactive features.
        is_interact (bool): Whether to use interactive features.
    """

    def __init__(
        self,
        feature_dims: List[int],
        dense_feature_dims: int,
        embed_dim: int,
        attn_factor_size: int,
        attn_dropout_rate: float,
        interact_feature_nums: int,
        is_interact: bool = False,
    ):
        super(AFM, self).__init__()
        self.name = "AFM"
        self.dense_feature_dims = dense_feature_dims
        self.embed_dim = embed_dim
        self.attn_factor_size = attn_factor_size
        self.is_interact = is_interact
        self.interact_feature_nums = interact_feature_nums

        # first order layers
        if not is_interact and interact_feature_nums:
            feature_dims = feature_dims[:-interact_feature_nums]
        self.dense_layer = nn.Linear(dense_feature_dims, 1)
        self.discrete_layer = nn.ModuleList([nn.Embedding(dim, 1) for dim in feature_dims])
        # embedding layers
        self.dense_embedding = nn.Linear(1, embed_dim)
        self.discrete_embeddings = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in feature_dims])

        # pair-wise interaction layer
        self.pair_wise_interaction = PairWiseInteractionLayer(
            embed_dim, len(feature_dims) + dense_feature_dims, attn_factor_size, attn_dropout_rate
        )
        self.output_layer = nn.Linear(2, 1)

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
        # AFM first order Calculation | dense_embed shape's [batch_size, 1]
        dense_out = self.dense_layer(dense_x)
        # discrete_out shape's [batch_size, 1]
        discrete_out = torch.stack(
            [emb(discrete_x[:, i]) for i, emb in enumerate(self.discrete_layer)], dim=1
        ).sum(dim=1)
        linear_logits = dense_out + discrete_out

        # Embedding layer | dense_embed shape's [batch_size, dense_feature_nums, embed_dim]
        dense_embed = self.dense_embedding(dense_x.unsqueeze(-1))
        # discrete_embed shape's [batch_size, discrete_feature_nums, embed_dim]
        discrete_embed = torch.stack(
            [emb(discrete_x[:, i]) for i, emb in enumerate(self.discrete_embeddings)], dim=1
        )
        # pair_wise_input shape's [batch_size, feature_nums, embed_dim]
        pair_wise_input = torch.cat([dense_embed, discrete_embed], dim=1)
        pair_wise_logits = self.pair_wise_interaction(pair_wise_input)
        logits = torch.cat([linear_logits, pair_wise_logits], dim=1)
        logits = self.output_layer(logits)
        return logits

    @classmethod
    def from_config(cls, config: AFMConfig) -> "AFM":
        """Create model from config."""
        return build_from_config(config, Registers.model_registry)


class PairWiseInteractionLayer(nn.Module):
    def __init__(
        self, embed_dim: int, feature_nums: int, attn_factor_size: int, attn_dropout_rate: float = 0.5
    ):
        super(PairWiseInteractionLayer, self).__init__()
        self.D = embed_dim
        self.N = feature_nums
        self.pair_nums = feature_nums * (feature_nums - 1) // 2
        self.W = nn.Linear(embed_dim, attn_factor_size)
        self.h = nn.Linear(attn_factor_size, self.pair_nums, bias=False)
        self.output_layer = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(attn_dropout_rate)

        idx_i, idx_j = torch.triu_indices(self.N, self.N, offset=1)
        self.register_buffer("idx_i", idx_i, persistent=False)
        self.register_buffer("idx_j", idx_j, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape's [batch_size, feature_nums, embed_dim]
        x_i = x[:, self.idx_i, :]
        x_j = x[:, self.idx_j, :]
        # pair_wise_x shape's [batch_size, pair_nums, embed_dim]
        pair_wise_x = x_i * x_j
        # attn_pair_wise_x shape's [batch_size, embed_dim]
        attn_pair_wise_x = torch.sum(pair_wise_x, dim=1)
        # attn_temp shape's [batch_size, attn_factor_size]
        attn_temp = F.relu(self.W(attn_pair_wise_x))
        # attn_temp shape's [batch_size, pair_nums]
        attn_weight = torch.softmax(self.h(attn_temp), dim=-1)
        attn_out = self.dropout(attn_weight)
        # out_x shape's [batch_size, pair_nums, embed_dim]
        out_x = attn_out.unsqueeze(-1) * pair_wise_x
        # out_x shape's [batch_size, embed_dim]
        out_x = torch.sum(out_x, dim=1)
        logits = self.output_layer(out_x)
        return logits
