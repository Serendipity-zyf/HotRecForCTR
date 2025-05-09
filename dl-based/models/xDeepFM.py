import torch
import torch.nn as nn

from typing import List
from typing import Tuple
from config import xDeepFMConfig
from utils.logger import ColorLogger
from utils.register import Registers
from utils.register import build_from_config

logger = ColorLogger(name="xDeepFM")


@Registers.model_registry.register
class xDeepFM(nn.Module):
    """
    xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems

    Args:
        feature_dims (List[int]): List of feature dimensions.
        dense_feature_dims (int): Dimension of dense features.
        embed_dim (int): Dimension of embedding.
        hidden_size (int): Dimension of hidden layer.
        num_dnn_layers (int): Number of DNN layers.
        dnn_dropout (float): Dropout rate for DNN layers.
        num_cin_layers (int): Number of CIN layers.
        feature_maps (int): Number of feature maps.
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
        dnn_dropout: float,
        num_cin_layers: int,
        feature_maps: int,
        interact_feature_nums: int,
        is_interact: bool = False,
    ):
        super(xDeepFM, self).__init__()
        self.name = "xDeepFM"
        self.dense_feature_dims = dense_feature_dims
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.is_interact = is_interact
        self.num_dnn_layers = num_dnn_layers
        self.num_cin_layers = num_cin_layers
        self.feature_maps = feature_maps
        self.interact_feature_nums = interact_feature_nums

        if not self.is_interact and self.interact_feature_nums:
            feature_dims = feature_dims[:-interact_feature_nums]

        self.num_features = len(feature_dims) + dense_feature_dims

        # Linear layers
        self.dense_layer = nn.Linear(dense_feature_dims, 1)
        self.discrete_layer = nn.ModuleList([nn.Embedding(dim, 1) for dim in feature_dims])

        # embedding layers
        self.embeddings = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in feature_dims])
        # dnn layers
        layers = []
        input_dim = len(feature_dims) * embed_dim + dense_feature_dims
        for j in range(num_dnn_layers):
            output_dim = hidden_size // (2**j)
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.BatchNorm1d(output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dnn_dropout))
            input_dim = output_dim

        self.dnn = nn.Sequential(*layers)
        self.dnn_output_layer = nn.Linear(input_dim, 1)
        # CIN layers
        self.dense_proj = nn.Parameter(torch.empty(size=(dense_feature_dims, embed_dim)))

        self.CIN = nn.ModuleList()
        for j in range(num_cin_layers):
            cin_hidden_size = self.num_features if j == 0 else self.feature_maps
            self.CIN.append(CINLayer(feature_maps, cin_hidden_size, self.num_features))
        self.cin_output_layer = nn.Linear(self.feature_maps * num_cin_layers, 1)

        # initialize weights
        def init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        self.apply(init_weights)
        nn.init.xavier_uniform_(self.dense_proj)

    def forward(self, dense_x: torch.Tensor, discrete_x: torch.Tensor) -> torch.Tensor:
        # dense_x shape's [batch_size, dense_feature_dims] | discrete_x shape's [batch_size, num_features]
        if not self.is_interact and self.interact_feature_nums > 0:
            discrete_x = discrete_x[:, : -self.interact_feature_nums]

        # Linear logits
        dense_out = self.dense_layer(dense_x)
        discrete_out = torch.stack(
            [emb(discrete_x[:, i]) for i, emb in enumerate(self.discrete_layer)], dim=1
        ).sum(dim=1)
        linear_logits = dense_out + discrete_out

        # Embedding layers | x_embeds shape's [batch_size, num_features, embed_dim]
        x_embeds = torch.stack([emb(discrete_x[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
        dnn_input = x_embeds.reshape(x_embeds.size(0), -1)
        dnn_input = torch.cat([dnn_input, dense_x], dim=1)
        # dnn_output shape's [batch_size, hidden_size // (2**num_dnn_layers)]
        dnn_output = self.dnn(dnn_input)
        dnn_logits = self.dnn_output_layer(dnn_output)

        # dense_x shape's [batch_size, dense_feature_dims, embed_dim]
        dense_x = dense_x.unsqueeze(-1) * self.dense_proj
        # cin_out shape's [batch_size, self.x_embed_dims]
        x_embeds = torch.cat([x_embeds, dense_x], dim=1)
        p_out_list = []
        cin_out = x_embeds.clone()
        for j in range(self.num_cin_layers):
            cin_out, cin_p_out = self.CIN[j](cin_out, x_embeds)
            p_out_list.append(cin_p_out)
        p_out = torch.cat(p_out_list, dim=1)
        cin_logits = self.cin_output_layer(p_out)
        return linear_logits + dnn_logits + cin_logits

    @classmethod
    def from_config(cls, config: xDeepFMConfig) -> "xDeepFM":
        """Create model from config."""
        return build_from_config(config, Registers.model_registry)


class CINLayer(nn.Module):
    """
    Compressed Interaction Network Layer for xDeepFM model.
    """

    def __init__(self, feature_maps: int, cin_hidden_size: int, num_features: int) -> None:
        super().__init__()
        weight = torch.empty(feature_maps, cin_hidden_size, num_features)
        nn.init.xavier_uniform_(weight)
        self.W = nn.Parameter(weight)

    def forward(self, xj: torch.Tensor, x0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # xj shape's [batch_size, cin_hidden_size, 1, embed_dim]
        xj_extend = xj.unsqueeze(2)
        # x0 shape's [batch_size, 1, num_features, embed_dim]
        x0_extend = x0.unsqueeze(1)
        # hadama_out shape's [batch_size, cin_hidden_size, num_features, embed_dim]
        hadamard = x0_extend * xj_extend
        # hadamard: [B, C, N, D] | W: [F, C, N]
        # out shape's [batch_size, feature_maps, cin_hidden_size, embed_dim]
        out = torch.einsum("b c n d, f c n -> b f c d", hadamard, self.W)
        # cin_out shape's [batch_size, feature_maps, embed_dim]
        cin_out = torch.sum(out, dim=2)
        # cin_p_out shape's [batch_size, feature_maps]
        cin_p_out = torch.sum(cin_out, dim=-1)
        return cin_out, cin_p_out
