import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from config import AutoIntConfig
from utils.logger import ColorLogger
from utils.register import Registers
from utils.register import build_from_config

logger = ColorLogger(name="AutoInt")


@Registers.model_registry.register
class AutoInt(nn.Module):
    """
    AutoInt: Automatic Feature Interaction Learning
    via Self-Attentive Neural Networks for CTR Prediction.

    Args:
        feature_dims (List[int]): List of feature dimensions.
        dense_feature_dims (int): Dimension of dense features.
        embed_dim (int): Dimension of embedding.
        hidden_size (int): Dimension of hidden layer.
        num_heads (int): Number of attention heads.
        num_atten_layers (int): Number of attention layers.
        interact_feature_nums (int): Number of interactive features.
        is_interact (bool): Whether to use interactive features.
    """

    def __init__(
        self,
        feature_dims: List[int],
        dense_feature_dims: int,
        embed_dim: int,
        attn_hidden_size: int,
        num_heads: int,
        num_atten_layers: int,
        attn_dropout: float,
        use_dnn: bool,
        dnn_layers: int,
        dnn_hidden_size: int,
        dnn_dropout: float,
        embedding_dropout: float,
        interact_feature_nums: int,
        is_interact: bool = False,
    ):
        super(AutoInt, self).__init__()
        self.name = "AutoInt"
        self.dense_feature_dims = dense_feature_dims
        self.embed_dim = embed_dim
        self.attn_hidden_size = attn_hidden_size
        self.num_heads = num_heads
        self.use_dnn = use_dnn
        self.dnn_layers = dnn_layers
        self.dnn_hidden_size = dnn_hidden_size
        self.is_interact = is_interact
        self.interact_feature_nums = interact_feature_nums
        self.embed_nums: List[int] = feature_dims if is_interact else feature_dims[:-interact_feature_nums]
        # Linear Layers
        self.dense_layer = nn.Linear(dense_feature_dims, 1)
        self.discrete_layer = nn.ModuleList([nn.Embedding(dim, 1) for dim in self.embed_nums])
        # Dnn Layers
        self.dense_to_embed = nn.Linear(1, embed_dim)
        if use_dnn:
            layers = []
            in_features = len(self.embed_nums) * embed_dim + dense_feature_dims
            for j in range(dnn_layers):
                out_features = self.dnn_hidden_size // (2**j)
                layers.append(nn.Linear(in_features, out_features))
                layers.append(nn.BatchNorm1d(out_features))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dnn_dropout))
                in_features = out_features
            self.dnn = nn.Sequential(*layers)

        self.discrete_embeddings = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in self.embed_nums])
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.atten_layers = nn.ModuleList()
        for j in range(num_atten_layers):
            query_dim = embed_dim if j == 0 else attn_hidden_size
            self.atten_layers.append(AttentionBlock(query_dim, num_heads, attn_hidden_size, attn_dropout))

        num_features = len(self.embed_nums) + dense_feature_dims
        if use_dnn:
            self.output_layer = nn.Linear(attn_hidden_size * num_features + in_features, 1)
        else:
            self.output_layer = nn.Linear(attn_hidden_size * num_features, 1)

        # initialization
        def init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        self.apply(init_weights)

    def forward(self, dense_x: torch.Tensor, discrete_x: torch.Tensor) -> torch.Tensor:
        # Linear Layer Calculation | dense_embed shape's [batch_size, 1]
        dense_logits = self.dense_layer(dense_x)
        discrete_logits = torch.stack(
            [emb(discrete_x[:, i]) for i, emb in enumerate(self.discrete_layer)], dim=1
        ).sum(dim=1)
        linear_logits = dense_logits + discrete_logits

        if not self.is_interact and self.interact_feature_nums:
            discrete_x = discrete_x[:, : -self.interact_feature_nums]
        # discrete_embed: [batch_size, discrete_feature_dims, embed_dim]
        discret_embed = torch.stack(
            [emb(discrete_x[:, i]) for i, emb in enumerate(self.discrete_embeddings)], dim=1
        )
        discret_embed = self.embedding_dropout(discret_embed)

        # DNN Layer Calculation | dnn_input: [batch_size, embed_nums * embed_dim + dense_feature_dims]
        if self.use_dnn:
            batch = discret_embed.size(0)
            dnn_input = torch.concat([discret_embed.reshape(batch, -1), dense_x], dim=1)
            dnn_output = self.dnn(dnn_input)

        # Attention layers Calculation | atten_x: [batch_size, embed_nums + dense_feature_nums, hidden_size]
        dense_embed = self.dense_to_embed(dense_x.unsqueeze(-1))
        atten_x = torch.cat([discret_embed, dense_embed], dim=1)
        for layer in self.atten_layers:
            atten_x = layer(atten_x, mask=None)
        # atten_x: [batch_size, (embed_nums + dense_feature_nums) * hidden_size]
        atten_x = atten_x.view(atten_x.size(0), -1)

        # logits Calculation
        if self.use_dnn:
            dnn_output = dnn_output.view(dnn_output.size(0), -1)
            # out: [batch_size, (embed_nums + dense_feature_nums) * hidden_size + hidden_size // (2**num_dnn_layers)]
            out = torch.cat([atten_x, dnn_output], dim=1)
            logits = self.output_layer(out)
        else:
            logits = self.output_layer(atten_x)
        return linear_logits + logits

    @classmethod
    def from_config(cls, config: AutoIntConfig) -> "AutoInt":
        """Create model from config."""
        return build_from_config(config, Registers.model_registry)


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer for AutoInt Network."""

    def __init__(self, query_dim: int, num_heads: int, hidden_size: int, attn_dropout: float = 0.0) -> None:
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.hidden_size = hidden_size
        assert (
            hidden_size % num_heads == 0
        ), f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        self.W_q = nn.Linear(query_dim, hidden_size)
        self.W_k = nn.Linear(query_dim, hidden_size)
        self.W_v = nn.Linear(query_dim, hidden_size)
        self.W_o = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, _ = x.size()
        # q, k, v: [batch_size, num_heads, seq_len, hidden_size // num_heads]
        q = self.W_q(x).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        # attn_weights: [batch_size, num_heads, seq_len, seq_len]
        head_dim = self.hidden_size // self.num_heads
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
        if mask is not None:
            # mask: [batch_size, 1, 1, seq_len]
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        # attn_output: [batch_size, num_heads, seq_len, hidden_size // num_heads]
        attn_output = torch.matmul(attn_weights, v)
        # attn_output: [batch_size, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.W_o(attn_output)


class AttentionBlock(nn.Module):
    """Attention layer for AutoInt Network."""

    def __init__(
        self,
        query_dim: int,
        num_heads: int,
        hidden_size: int,
        attn_dropout: float,
    ) -> None:
        super(AttentionBlock, self).__init__()
        self.query_dim = query_dim
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.multi_head_attention = MultiHeadAttention(query_dim, num_heads, hidden_size, attn_dropout)
        self.res_layer = nn.Linear(query_dim, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attention_output = self.multi_head_attention(x, mask)
        res_output = self.res_layer(x)
        out_x = attention_output + res_output
        return F.relu(self.norm(out_x))
