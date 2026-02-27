import torch
import torch.nn as nn
from typing import Optional

class TrajectoryAttentionBlock(nn.Module):
    """
    Input:  pts  (B, K, 2)  normalized or raw 2D points
    Output: x    (B, K, D)  point tokens after self-attention (Transformer-style)

    Compatible with future cross-attention because it returns standard token embeddings.
    """
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        mlp_hidden_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model

        # Transformer encoder block over point tokens
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,                          # (B, K, d_model)
        padding_mask: Optional[torch.Tensor] = None # (B, K) True = pad (ignored by attention)
    ) -> torch.Tensor:
        B, K, C = x.shape
        assert C == self.d_model, f"Expected (B,K,{self.d_model}) features, got {x.shape}"

        # Self-attention (pre-norm)
        h = self.ln1(x)
        y, _ = self.self_attn(h, h, h, key_padding_mask=padding_mask, need_weights=False)
        x = x + self.drop1(y)

        # FFN
        x = x + self.ffn(self.ln2(x))
        return x
