import torch
import torch.nn as nn

from typing import List
from config import DINConfig
from utils.logger import ColorLogger
from utils.register import Registers
from utils.register import build_from_config

logger = ColorLogger(name="DIN")


@Registers.model_registry.register
class DIN(nn.Module):
    """
    DIN: Deep Interest Network for Click-Through Rate Prediction

    Args:
        user_num (int): Number of users.
        item_num (int): Number of items.
        cate_num (int): Number of categories.
        embed_dim (int): Dimension of the embedding layer.
        hidden_size (int): Size of the hidden layer.
        unit_hidden_size (int): Size of the unit hidden layer.
        dense_nums (int, optional): Number of dense features. Default is 2.
        use_attn (bool, optional): Whether to use attention mechanism. Default is True.

    """

    def __init__(
        self,
        user_num: int,
        item_num: int,
        cate_num: int,
        embed_dim: int,
        hidden_size: int,
        unit_hidden_size: int,
        dense_nums: int = 2,
        use_attn: bool = True,
    ):
        super(DIN, self).__init__()
        self.name = "DIN"
        self.unit_hidden_size = unit_hidden_size
        self.dense_nums = dense_nums
        self.use_attn = use_attn

        self.user_embedding = nn.Embedding(user_num, embed_dim * 2, padding_idx=0)
        self.item_embedding = nn.Embedding(item_num, embed_dim, padding_idx=0)
        self.cate_embedding = nn.Embedding(cate_num, embed_dim, padding_idx=0)

        self.dnn_input_size = embed_dim * (2 + 2 + 2) + dense_nums
        self.dnn = nn.Sequential(
            nn.Linear(self.dnn_input_size, hidden_size),
            Dice(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            Dice(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1),
        )

        self.unit_input_size = embed_dim * (2 + 2 + 2)
        self.activation_unit = ActivationUnit(self.unit_input_size, self.unit_hidden_size)

        # initialization
        def init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_weights)

    def forward(
        self,
        uid_idx: torch.Tensor,
        iid_idx: torch.Tensor,
        cate_idx: torch.Tensor,
        seq_idx: torch.Tensor,
        seq_cat_idx: torch.Tensor,
        mask: torch.Tensor,
        dense: torch.Tensor,
    ) -> torch.Tensor:
        # user_embed | item_embed | cate_embed: [batch_size, embed_dim]
        user_embed = self.user_embedding(uid_idx)
        item_embed = self.item_embedding(iid_idx)
        cate_embed = self.cate_embedding(cate_idx)
        # item_embed: [batch_size, embed_dim * 2]
        item_embed = torch.cat([item_embed, cate_embed], dim=-1)

        # seq_embed | seq_cate_embed: [batch_size, seq_len, embed_dim]
        seq_embed = self.item_embedding(seq_idx)
        seq_cate_embed = self.cate_embedding(seq_cat_idx)
        # seq_embed: [batch_size, seq_len, embed_dim * 2]
        seq_embed = torch.cat([seq_embed, seq_cate_embed], dim=-1)

        # masked_seq_embed: [batch_size, seq_len, embed_dim * 2] | mask: [batch_size, seq_len]
        masked_seq_embed = seq_embed * mask.unsqueeze(-1).float()
        if self.use_attn:
            # activation_out: [batch_size, seq_len, embed_dim * 2]
            activation_out = self.activation_unit(user_embed, masked_seq_embed)
            # pooled_embed: [batch_size, embed_dim * 2]
            pooled_embed = activation_out.sum(dim=1)
        else:
            # pooled_embed: [batch_size, embed_dim * 2]
            pooled_embed = masked_seq_embed.sum(dim=1)

        dnn_input = torch.cat([user_embed, item_embed, pooled_embed, dense], dim=-1)
        return self.dnn(dnn_input)

    @classmethod
    def from_config(cls, config: DINConfig) -> "DIN":
        """Create model from config."""
        return build_from_config(config, Registers.model_registry)


class Dice(nn.Module):
    def __init__(self, input_dim, epsilon=1e-8):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim, eps=epsilon, affine=False)
        self.alpha = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() == 3:
            x = x.view(-1, x.size(-1))  # [B*T, D]

        norm_x = self.bn(x)
        p = torch.sigmoid(norm_x)
        out = p * x + (1 - p) * self.alpha * x

        if len(orig_shape) == 3:
            out = out.view(orig_shape)  # reshape back to [B, T, D]

        return out


class ActivationUnit(nn.Module):
    def __init__(self, input_dim: int, unit_hidden_size: str):
        super(ActivationUnit, self).__init__()
        self.input_dim = input_dim
        self.unit_hidden_size = unit_hidden_size

        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, unit_hidden_size),
            Dice(unit_hidden_size),
            nn.Linear(unit_hidden_size, 1),
        )

    def forward(self, user_embed: torch.Tensor, seq_embed: torch.Tensor) -> torch.Tensor:
        # user_embed: [batch_size, embed_dim * 2] | seq_embed: [batch_size, seq_len, embed_dim * 2]
        # user_embed: [batch_size, seq_len, embed_dim * 2]
        user_embed = user_embed.unsqueeze(1).expand(-1, seq_embed.size(1), -1)
        # out_product: [batch_size, seq_len, embed_dim * 2]
        out_product = user_embed * seq_embed
        # out: [batch_size, seq_len, input_dim(embed_dim * 6)]
        out = torch.cat([user_embed, out_product, seq_embed], dim=-1)
        # weights: [batch_size, seq_len, 1]
        atten_weights = self.output_layer(out)
        return seq_embed * atten_weights
