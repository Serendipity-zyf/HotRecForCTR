import torch
import torch.nn as nn

from typing import List
from config import NFMConfig
from utils.logger import ColorLogger
from utils.register import Registers
from utils.register import build_from_config

logger = ColorLogger(name="NFM")


@Registers.model_registry.register
class NFM(nn.Module):
    """
    NFM: Neural Factorization Machines for Sparse Predictive Analytics

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
        super(NFM, self).__init__()
        self.name = "NFM"
        self.dense_feature_dims = dense_feature_dims
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.is_interact = is_interact
        self.interact_feature_nums = interact_feature_nums

        # first order layers
        if not is_interact and interact_feature_nums:
            feature_dims = feature_dims[:-interact_feature_nums]
        self.dense_layer = nn.Linear(dense_feature_dims, 1)
        self.discrete_layer = nn.ModuleList([nn.Embedding(dim, 1) for dim in feature_dims])
        # embedding layers
        self.dnn_embeddings = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in feature_dims])
        self.batch_norm = nn.BatchNorm1d(embed_dim + dense_feature_dims)
        # dnn layers
        layers = []
        in_features = embed_dim + dense_feature_dims
        for j in range(dnn_layers):
            out_features = self.hidden_size // (2**j)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.BatchNorm1d(out_features))
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
        # NFM first order Calculation | dense_embed shape's [batch_size, 1]
        dense_out = self.dense_layer(dense_x)
        # discrete_out shape's [batch_size, 1]
        discrete_out = torch.stack(
            [emb(discrete_x[:, i]) for i, emb in enumerate(self.discrete_layer)], dim=1
        ).sum(dim=1)
        linear_logits = dense_out + discrete_out
        # NFM second order Calculation | embeds shape's [batch_size, num_discrete_features, embed_dim]
        embeds = torch.stack([emb(discrete_x[:, i]) for i, emb in enumerate(self.dnn_embeddings)], dim=1)
        # sum_of_embeds shape's [batch_size, embed_dim]
        sum_of_embeds = torch.sum(embeds, dim=1)
        # square_of_sum shape's [batch_size, embed_dim]
        square_of_sum = torch.square(sum_of_embeds)
        # sum_of_square shape's [batch_size, embed_dim]
        sum_of_square = torch.sum(torch.square(embeds), dim=1)
        # dnn_input shape's [batch_size, embed_dim]
        bi_interact_out = 0.5 * (square_of_sum - sum_of_square)
        bi_interact_out_ = torch.concat([bi_interact_out, dense_x], dim=-1)
        dnn_input = self.batch_norm(bi_interact_out_)
        # dnn_output shape's [batch_size, 1]
        dnn_output = self.dnn(dnn_input)
        dnn_logits = self.output_layer(dnn_output)
        return linear_logits + dnn_logits

    @classmethod
    def from_config(cls, config: NFMConfig) -> "NFM":
        """Create model from config."""
        return build_from_config(config, Registers.model_registry)
