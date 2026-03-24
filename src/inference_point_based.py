from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoImageProcessor, AutoModel
import yaml
from PIL import Image

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

class RewardModelPointBased(nn.Module):
    """
    Reward model that lets trajectory tokens query image patch tokens via cross-attention,
    then predicts a scalar reward.
    """

    def __init__(
            self,
            d_model: int = 384,
            n_heads: int = 8,
            dropout: float = 0.1,
            verbose: bool = True,
            freeze_image_encoder: bool = True,
            keep_cls_token: bool = False,
            fusion_blocks: int = 4,
            num_blocks: int = 4,
            traj_per_image: int = 4
    ):
        super().__init__()

        self.keep_cls_token = keep_cls_token
        self.image_feature_extractor_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        self.freeze_image_encoder = freeze_image_encoder
        self.traj_per_image = traj_per_image

        # Image feature extractor (DINOv3)
        if verbose:
            print("loading image feature extractor", self.image_feature_extractor_name)
        self.processor = AutoImageProcessor.from_pretrained(self.image_feature_extractor_name)
        self.image_feature_extractor = AutoModel.from_pretrained(self.image_feature_extractor_name)
        if self.freeze_image_encoder:
            # Important for DDP: frozen params must not require grad.
            for p in self.image_feature_extractor.parameters():
                p.requires_grad = False

        image_dim = self.image_feature_extractor.config.hidden_size
        if verbose:
            print(self.image_feature_extractor)
            print("Num register tokens:", self.image_feature_extractor.config.num_register_tokens)  # 4
            print("Image hidden dim:", image_dim)

        self.d_model = d_model
        self.num_register_tokens = self.image_feature_extractor.config.num_register_tokens

        self.trajectory_transformer = TrajectoryTransformer(d_model=d_model,
                                                            num_blocks=num_blocks)

        self.image_proj = nn.Identity() if image_dim == d_model else nn.Linear(image_dim, d_model)

        self.fusion = nn.ModuleList([
            FusionBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(fusion_blocks)
        ])

        self.reward_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def _extract_image_tokens(self, image_inputs, keep_cls=False) -> torch.Tensor:
        """
        Returns patch tokens with shape (B, N_img, D_model), excluding register tokens and cls token (if not keep_cls).
        """
        if self.freeze_image_encoder:
            with torch.no_grad():
                img_output = self.image_feature_extractor(**image_inputs)
        else:
            img_output = self.image_feature_extractor(**image_inputs)

        img_tokens = img_output.last_hidden_state
        img_tokens = img_tokens[:, 1 + self.num_register_tokens:, :]

        if keep_cls:
            cls_token = img_output.last_hidden_state[:, :1, :]
            img_tokens = torch.cat([cls_token, img_tokens], dim=1)
        return self.image_proj(img_tokens)

    def forward(self, pts: torch.Tensor, image_inputs) -> torch.Tensor:
        """
        Args:
            pts: (B, M, K, 2) tensor of point trajectories
            image_inputs: dict-like input for DINOv3 (must include pixel_values)
            padding_mask: (B, K) boolean tensor where True indicates padding points to ignore

        Returns:
            rewards: (B,) tensor of scalar rewards for each trajectory
        """

        B, M, K, _ = pts.shape
        pts_flat = pts.reshape(B * M, K, 2)
        x = self.trajectory_transformer(pts_flat)  # (B, K+1, D_model) with CLS at index 0

        img_tokens = self._extract_image_tokens(image_inputs, keep_cls=self.keep_cls_token)  # (B, N_img, D_model)
        B, N_img, _ = img_tokens.shape

        img_tokens_exp = img_tokens[:, None, :, :].expand(B, M, N_img, self.d_model)
        img_tokens_flat = img_tokens_exp.reshape(B * M, N_img, self.d_model)

        # Extend mask for prepended CLS token (never masked).
        B = x.shape[0]

        # if padding_mask is not None:
        #     cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=x.device)
        #     x_padding_mask = torch.cat([cls_mask, padding_mask.bool()], dim=1)

        # Trajectory queries attend to image patch keys/values.
        for block in self.fusion:
            x = block(x, img_tokens_flat)

        # CLS readout for reward prediction.
        cls_feat = x[:, 0, :]
        rewards = self.reward_head(cls_feat).squeeze(-1)
        return rewards


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config_point_based.yaml"


def build_image_inputs(
    model: RewardModelPointBased, image_tensor: torch.Tensor, device: torch.device
) -> dict[str, torch.Tensor]:
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    if image_tensor.ndim != 4:
        raise ValueError(f"Expected image shaped (3,H,W), (H,W,3), (B,3,H,W), or (B,H,W,3); got {tuple(image_tensor.shape)}.")

    if image_tensor.shape[1] == 3:
        image_batch = image_tensor
    elif image_tensor.shape[-1] == 3:
        image_batch = image_tensor.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Expected 3 image channels, got shape {tuple(image_tensor.shape)}.")

    image_batch = image_batch.to(dtype=torch.float32, device="cpu")
    if image_batch.numel() > 0 and image_batch.max().item() > 1.0:
        image_batch = image_batch / 255.0

    image_inputs = model.processor(images=image_batch, return_tensors="pt", do_rescale=False)
    return {k: v.to(device, non_blocking=True) for k, v in image_inputs.items()}


def build_points_tensor(points_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if points_tensor.ndim == 3:
        points_tensor = points_tensor.unsqueeze(0)
    if points_tensor.ndim != 4:
        raise ValueError(
            f"Expected points shaped (M,K,2) or (B,M,K,2); got {tuple(points_tensor.shape)}."
        )
    if points_tensor.shape[-1] != 2:
        raise ValueError(f"Expected points with last dimension 2, got shape {tuple(points_tensor.shape)}.")
    return points_tensor.to(device=device, dtype=torch.float32, non_blocking=True)


def build_demo_inputs(
    num_paths: int = 4,
    num_points: int = 8,
    max_x: float = 8.0,
    max_y: float = 2.0,
    image_size: tuple[int, int] = (224, 224),
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = image_size
    image_tensor = torch.rand(3, height, width, dtype=torch.float32)

    x_steps = torch.rand(num_paths, num_points, dtype=torch.float32).cumsum(dim=1)
    y_steps = torch.rand(num_paths, num_points, dtype=torch.float32).cumsum(dim=1)
    x_coords = max_x * x_steps / x_steps[:, -1:].clamp_min(1e-6)
    y_coords = max_y * y_steps / y_steps[:, -1:].clamp_min(1e-6)
    points_tensor = torch.stack([x_coords, y_coords], dim=-1)
    return image_tensor, points_tensor


class RewardInferenceRunner:
    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path | None = None,
        verbose: bool = False,
    ) -> None:
    
        self.model = self.load_model(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            verbose=verbose
        )
        

    def _load_config(self, config_path: str) -> dict[str, Any]:
        path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
        with path.open("r") as handle:
            return yaml.load(handle, Loader=yaml.SafeLoader)
        
    def _select_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, checkpoint_path: str,  config_path: str, verbose: bool = False) -> RewardModelPointBased:
        config = self._load_config(config_path)
        device = self._select_device()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

        model = RewardModelPointBased(
            d_model=config["d_model"],
            n_heads=config["num_heads"],
            dropout=config["dropout"],
            verbose=verbose,
            fusion_blocks=config["fusion_blocks"],
            num_blocks=config["num_blocks"],
            traj_per_image=4
        )
        model.to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        return model
    
    @torch.inference_mode()
    def predict_rewards(self, image_tensor: torch.Tensor, points_tensor: torch.Tensor) -> torch.Tensor:
        device = next(self.model.parameters()).device
        image_inputs = build_image_inputs(self.model, image_tensor, device)
        points_batch = build_points_tensor(points_tensor, device)
        return self.model(points_batch, image_inputs)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Inference for point-based reward model")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a saved training checkpoint.")
    parser.add_argument("--points", type=Path, required=True, help="Path to a .npy file shaped (M,K,2) or (B,M,K,2).")
    parser.add_argument("--image", type=Path, required=True, help="Path to an RGB image.")
    parser.add_argument("--config", type=Path, help="Path to config_point_based.yaml.")
    parser.add_argument("--verbose", action="store_true", help="Print model initialization details.")
    return parser.parse_args()

def main() -> None:
    terminal = False

    if terminal:
        args = parse_args()
        DEFAULT_CONFIG_PATH = "/home/jim/Projects/prune/config/config_point_based.yaml"
        config_path = args.config if args.config is not None else DEFAULT_CONFIG_PATH
        checkpoint_path = args.checkpoint
        verbose = args.verbose
        image_tensor = torch.from_numpy(np.array(Image.open(args.image).convert("RGB"))).permute(2, 0, 1).contiguous()  # (3, H, W)
        points_tensor = torch.from_numpy(np.load(args.points))  # (M, K, 2)
    else:
        # For notebook testing, use synthetic inputs so the forward pass can be smoke-tested quickly.
        checkpoint_path = "./weights/epoch_029.pt"
        config_path = "./config/config_point_based.yaml"
        verbose = True
        image_tensor, points_tensor = build_demo_inputs()

    runner = RewardInferenceRunner(checkpoint_path=checkpoint_path, config_path=config_path, verbose=verbose)
    rewards = runner.predict_rewards(image_tensor=image_tensor, points_tensor=points_tensor)

    print("Predicted rewards:", rewards)

if __name__ == "__main__":
    main()
