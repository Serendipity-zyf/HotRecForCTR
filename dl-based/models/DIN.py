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
        dense_nums: int = 2,
        use_attn: bool = True,
    ):
        super(DIN, self).__init__()
        self.name = "DIN"
        self.dense_nums = dense_nums
        self.use_attn = use_attn
        self.dnn_input_size = embed_dim * (1 + 2 + 2) + dense_nums

        self.user_embedding = nn.Embedding(user_num, embed_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(item_num, embed_dim, padding_idx=0)
        self.cate_embedding = nn.Embedding(cate_num, embed_dim, padding_idx=0)

        self.dnn = nn.Sequential(
            nn.Linear(self.dnn_input_size, hidden_size),
            Dice(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            Dice(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1),
        )

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

        if not self.use_attn:
            # masked_seq_embed: [batch_size, seq_len, embed_dim * 2] | mask: [batch_size, seq_len]
            masked_seq_embed = seq_embed * mask.unsqueeze(-1).float()
            # pooled_embed: [batch_size, embed_dim * 2] | mask: [batch_size, seq_len]
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

    def forward(self, x):
        # x: (batch_size, input_dim)
        norm_x = self.bn(x)
        p = torch.sigmoid(norm_x)
        return p * x + (1 - p) * self.alpha * x
