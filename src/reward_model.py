"""
reward_model.py
reward model using the trajectory preferences
"""
import torch
import torch.nn as nn
from transformers import TorchAoConfig, AutoImageProcessor, AutoModel
from models.trajectory_transformer import TrajectoryTransformer
from models.fusion_block import FusionBlock
from torchvision.transforms import v2

"""
DINOv3 related transforms, etc
see: https://github.com/facebookresearch/dinov3
"""
class RewardModel(nn.Module):
    def __init__(self,
                 d_model: int = 384,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 fusion_blocks: int = 4,
                 num_blocks: int = 4,
                 verbose: bool = True):
        super().__init__()
        # self.model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
        self.image_feature_extractor_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        # self.model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        # self.model_name = "facebook/dinov3-vit7b16-pretrain-lvd1689m"

        # Load DINOv3
        if verbose:
            print("loading model", self.image_feature_extractor_name)
        self.processor = AutoImageProcessor.from_pretrained(self.image_feature_extractor_name)
        self.image_feature_extractor = AutoModel.from_pretrained(self.image_feature_extractor_name)
        self.patch_size = self.image_feature_extractor.config.patch_size
        self.image_dim = self.image_feature_extractor.config.hidden_size
        # Important for DDP: frozen params must not require grad.
        for p in self.image_feature_extractor.parameters():
            p.requires_grad = False
        if verbose:
            print(self.image_feature_extractor)
            print("Patch size:", self.patch_size)  # 16
            print("Image hidden dim:", self.image_dim)
            print("Num register tokens:", self.image_feature_extractor.config.num_register_tokens)  # 4
        self.d_model = d_model
        self.num_register_tokens = self.image_feature_extractor.config.num_register_tokens

        self.trajectory_transformer = TrajectoryTransformer(d_model=d_model,
                                                            num_blocks=num_blocks)

        self.image_proj = nn.Identity() if self.image_dim == d_model else nn.Linear(self.image_dim, d_model)

        self.fusion = nn.ModuleList([
            FusionBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(fusion_blocks)
        ])

        # Reward Prediction Head
        self.reward_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, pts: torch.Tensor, image_inputs) -> torch.Tensor:
        """
        Args:
            pts: (B, M, K, 2) tensor of point trajectories
            B: batch
            M: number of trajectories, 4?
            K: number of points in each trajectory, 10
            2: (x, y) of trajectory coordinates
            image_inputs: dict-like input for DINOv3 (must include pixel_values)
                (batch_size, 3, 224, 224)
        Returns:
            rewards: (BxM) tensor of scalar rewards for each trajectory
        """
        B, M, K, _ = pts.shape
        pts_flat = pts.reshape(B * M, K, 2).float()
        x = self.trajectory_transformer(pts_flat)  # (B, K+1, D_model) with CLS at index 0]

        with torch.no_grad():
            img_output = self.image_feature_extractor(**image_inputs)
        # original_patch_features = orig_output.last_hidden_state[:, 0, :] # same as: img_output.pooler_output

        img_tokens = img_output.last_hidden_state
        img_tokens = img_tokens[:, 1 + self.num_register_tokens :, :] # [batch, 196, 768]

        B, n_patches, embed = img_tokens.shape
        assert (embed == self.d_model) , f"embedding size {embed} does not match d_model {self.d_model}"

        img_tokens_exp  = img_tokens[:, None, :, :].expand(B, M, n_patches, embed)
        img_tokens_flat = img_tokens_exp.reshape(B * M, n_patches, embed)

        # Trajectory queries attend to image patch keys/values.
        for block in self.fusion:
            x = block(x, img_tokens_flat)

        # CLS readout for reward prediction.
        cls_feat = x[:, 0, :]
        rewards = self.reward_head(cls_feat).squeeze(-1)
        return rewards # [batch * M (number of trajectories)]