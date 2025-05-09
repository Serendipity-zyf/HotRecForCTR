import torch
import torch.nn as nn

from typing import List
from config import DeepFMConfig
from utils.logger import ColorLogger
from utils.register import Registers
from utils.register import build_from_config

logger = ColorLogger(name="DeepFM")


@Registers.model_registry.register
class DeepFM(nn.Module):
    """
    DeepFM:

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
        super(DeepFM, self).__init__()
        self.name = "DeepFM"
        self.dense_feature_dims = dense_feature_dims
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.dnn_layers = dnn_layers
        self.dropout_rate = dropout_rate
        self.is_interact = is_interact
        self.interact_feature_nums = interact_feature_nums

        # Linear layers
        if not is_interact and interact_feature_nums:
            feature_dims = feature_dims[:-interact_feature_nums]
        self.dense_layer = nn.Linear(dense_feature_dims, 1)
        self.discrete_layer = nn.ModuleList([nn.Embedding(dim, 1) for dim in feature_dims])
        # embedding layer
        self.num_features = len(feature_dims) + dense_feature_dims
        self.discrete_embed_layer = nn.ModuleList([nn.Embedding(num, embed_dim) for num in feature_dims])
        # FM layer
        self.fm_layer = FMLayer()
        # DNN layer
        layers = []
        in_features = len(feature_dims) * embed_dim + dense_feature_dims
        for j in range(dnn_layers):
            out_features = self.hidden_size // (2**j)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = out_features
        self.dnn = nn.Sequential(*layers)
        self.dnn_output_layer = nn.Linear(in_features, 1)

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
        if not self.is_interact and self.interact_feature_nums:
            discrete_x = discrete_x[:, : -self.interact_feature_nums]
        # dense_embed shape's [batch_size, 1]
        dense_out = self.dense_layer(dense_x)
        # discrete_out shape's [batch_size, 1]
        discrete_out = torch.stack(
            [emb(discrete_x[:, i]) for i, emb in enumerate(self.discrete_layer)], dim=1
        ).sum(dim=1)
        linear_logits = dense_out + discrete_out
        # discrete_embed_x's shape: [batch_size, num_features, embed_dim]
        discrete_embed_x = torch.stack(
            [emb(discrete_x[:, i]) for i, emb in enumerate(self.discrete_embed_layer)], dim=1
        )
        fm_logits = self.fm_layer(discrete_embed_x)
        # dnn_input's shape: [batch_size, num_features * embed_dim + dense_feature_dims]
        batch_size = discrete_embed_x.size(0)
        dnn_input = torch.cat([dense_x, discrete_embed_x.reshape(batch_size, -1)], dim=1)
        dnn_output = self.dnn(dnn_input)
        dnn_logits = self.dnn_output_layer(dnn_output)
        return linear_logits + fm_logits + dnn_logits

    @classmethod
    def from_config(cls, config: DeepFMConfig) -> "DeepFM":
        """Create model from config."""
        return build_from_config(config, Registers.model_registry)


class FMLayer(nn.Module):
    """FM Layer."""

    def __init__(self):
        super(FMLayer, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input x's shape: [batch_size, num_features, embed_dim]
        sum_of_embeds = torch.sum(x, dim=1)  # [batch_size, embed_dim]
        square_of_sum = torch.square(sum_of_embeds)  # [batch_size, embed_dim]
        sum_of_square = torch.sum(torch.square(x), dim=1)  # [batch_size, embed_dim]
        fm_logits = torch.sum(0.5 * (square_of_sum - sum_of_square), dim=1, keepdim=True)  # [batch_size, 1]
        return fm_logits
