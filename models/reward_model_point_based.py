import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from typing import Optional

from models.trajectory_transformer import TrajectoryTransformer
from models.fusion_block import FusionBlock

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
        img_tokens = img_tokens[:, 1 + self.num_register_tokens :, :]

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
        
        img_tokens_exp  = img_tokens[:, None, :, :].expand(B, M, N_img, self.d_model)
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
