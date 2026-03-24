import torch
import torch.nn as nn
from typing import Optional


class FusionBlock(nn.Module):
    """
    Decoder-style fusion block:
    1) self-attention over trajectory tokens
    2) cross-attention where trajectory queries image tokens
    3) feed-forward network
    """

    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 8,
        mlp_hidden_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.self_ln = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_drop = nn.Dropout(dropout)

        self.cross_ln = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_drop = nn.Dropout(dropout)

        self.ffn_ln = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, K, D): trajectory tokens
        memory: torch.Tensor,  # (B, N, D): image tokens
    ) -> torch.Tensor:
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected x last dim {self.d_model}, got {x.shape[-1]}")
        if memory.shape[-1] != self.d_model:
            raise ValueError(f"Expected memory last dim {self.d_model}, got {memory.shape[-1]}")

        # 1) Trajectory self-attention
        h = self.self_ln(x)
        self_out, _ = self.self_attn(
            query=h,
            key=h,
            value=h,
            need_weights=False,
        )
        x = x + self.self_drop(self_out)

        # 2) Cross-attention (trajectory queries -> image keys/values)
        h = self.cross_ln(x)
        cross_out, _ = self.cross_attn(
            query=h,
            key=memory,
            value=memory,
            need_weights=False,
        )
        x = x + self.cross_drop(cross_out)

        # 3) Feed-forward
        x = x + self.ffn(self.ffn_ln(x))
        return x
