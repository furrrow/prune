import torch
import torch.nn as nn
from typing import Optional
from models.trajectory_transformer_block import TrajectoryAttentionBlock


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, K, _ = x.shape
        if K > self.pe.shape[1]:
            raise ValueError(f"Sequence length {K} exceeds max_len {self.pe.shape[1]} for sinusoidal encoding.")
        return x + self.pe[:, :K, :].to(dtype=x.dtype)


class PointProjector(nn.Module):
    def __init__(self, d_model:int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        B, K, C = pts.shape
        assert C == 2, f"Expected (B,K,2) points, got {pts.shape}"

        return self.proj(pts)  # (B, K, D)
        
class TrajectoryTransformer(nn.Module):
    """
    A simple stack of PointAttentionBlocks to process point trajectories.
    """
    def __init__(
        self,
        num_blocks: int = 4,
        d_model: int = 256,
        n_heads: int = 8,
        mlp_hidden_mult: int = 4,
        dropout: float = 0.0,
        num_points: int = 512,
        use_sinusoidal_pos: bool = True,
        use_cls_token: bool = True,
    ):
        super().__init__()

        self.point_proj = PointProjector(d_model=d_model)
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)

        self.use_sinusoidal_pos = use_sinusoidal_pos
        if use_sinusoidal_pos:
            pos_len = num_points + (1 if use_cls_token else 0)
            self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=pos_len)
        self.attention = nn.ModuleList([
            TrajectoryAttentionBlock(
                d_model=d_model,
                n_heads=n_heads,
                mlp_hidden_mult=mlp_hidden_mult,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        pts: torch.Tensor,                          # (B, K, 2)
        padding_mask: Optional[torch.Tensor] = None # (B, K) True = pad
    ) -> torch.Tensor:
        x = self.point_proj(pts)
        if self.use_cls_token:
            B = x.shape[0]
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            if padding_mask is not None:
                cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=padding_mask.device)
                padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        if self.use_sinusoidal_pos:
            x = self.pos_enc(x)
        for block in self.attention:
            x = block(x, padding_mask=padding_mask)  # (B, K, D)
        return x
