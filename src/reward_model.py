"""
reward_model.py
reward model using the trajectory preferences
"""
import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel
from models.trajectory_transformer import TrajectoryTransformer
from models.fusion_block import FusionBlock

"""
I-JEPA
see: https://huggingface.co/docs/transformers/en/model_doc/ijepa#transformers.IJepaForImageClassification
"""
class ImageRewardModel(nn.Module):
    def __init__(self,
                 n_heads: int = 8,
                 hidden_dim: int = 1024,
                 dropout: float = 0.1,
                 verbose: bool = True,
                 image_feature_extractor_name: str = "jmtzt/ijepa_vitg16_22k",
                 freeze_image_encoder: bool = True,
                 image_feature_extractor: nn.Module | None = None,
                 processor=None):
        super().__init__()
        self.image_feature_extractor_name = image_feature_extractor_name
        self.freeze_image_encoder = freeze_image_encoder

        # Load DINOv3
        if verbose:
            print("loading model", self.image_feature_extractor_name)
        self.processor = processor
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.image_feature_extractor_name)
        self.image_feature_extractor = image_feature_extractor
        if self.image_feature_extractor is None:
            self.image_feature_extractor = AutoModel.from_pretrained(self.image_feature_extractor_name)
        self.patch_size = self.image_feature_extractor.config.patch_size
        self.image_dim = self.image_feature_extractor.config.hidden_size
        if self.freeze_image_encoder:
            # Important for DDP: frozen params must not require grad.
            for p in self.image_feature_extractor.parameters():
                p.requires_grad = False
            self.image_feature_extractor.eval()
        if verbose:
            print(self.image_feature_extractor)
            print("Patch size:", self.patch_size)  # 16
            print("Image hidden dim:", self.image_dim) # 1408 for jepa
        self.num_heads = n_heads
        self.hidden_dim = hidden_dim
        # Self-Attention Over Vision Features
        self.multihead_attn1 = nn.MultiheadAttention(embed_dim=self.image_dim, num_heads=self.num_heads,
                                                     batch_first=True, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(embed_dim=self.image_dim, num_heads=self.num_heads,
                                                     batch_first=True, dropout=dropout)
        self.multihead_attn3 = nn.MultiheadAttention(embed_dim=self.image_dim, num_heads=self.num_heads,
                                                     batch_first=True, dropout=dropout)
        self.attn_norm = nn.LayerNorm(self.image_dim)
        self.image_proj = nn.Identity() if self.image_dim == self.hidden_dim else nn.Linear(self.image_dim, self.hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

        # patch feature distillation
        self.patch_conv1 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=5, stride=2)
        self.patch_conv2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=2)
        self.patch_conv3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=2)

        # Reward Prediction Head
        self.reward_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1), )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_image_encoder:
            self.image_feature_extractor.eval()
        return self

    def forward(self, orig_input, annotated_input, input_type="image") -> torch.Tensor:
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
            rewards: (B, M) tensor for 4D trajectory input, or (B*M,) tensor for already-flat 3D input
        """
        if input_type=="image":
            if self.freeze_image_encoder:
                with torch.no_grad():
                    orig_output = self.image_feature_extractor(**orig_input)
                    annotated_output = self.image_feature_extractor(**annotated_input)
            else:
                orig_output = self.image_feature_extractor(**orig_input)
                annotated_output = self.image_feature_extractor(**annotated_input)
            orig_features = orig_output.last_hidden_state  # [B, 196, 1408])
            annotated_features = annotated_output.last_hidden_state  # [B, 196, 1408])
        elif input_type=="features": # the image_feature_extractor is already done.
            orig_features = orig_input
            annotated_features = annotated_input
        batch, k_sq = annotated_features.shape[0:2]
        k = torch.sqrt(torch.tensor(k_sq)).item()
        # Self-Attention on Vision Features
        attn_output, _ = self.multihead_attn1(annotated_features, annotated_features,
                                              annotated_features)  # Shape: (batch_size, num_patches, hidden_dim)
        attn_output = self.attn_norm(self.dropout(attn_output))  # Normalize After Self-Attention
        residual1 = attn_output

        attn_output, _ = self.multihead_attn2(attn_output, orig_features,
                                              orig_features)
        attn_output = self.attn_norm(residual1 + self.dropout(attn_output))
        # residual2 = attn_output

        # attn_output, _ = self.multihead_attn3(attn_output, attn_output,
        #                                       attn_output)
        # attn_output = self.attn_norm(residual2 + self.dropout(attn_output))
        proj_output = self.image_proj(attn_output)
        output_img = proj_output.reshape(batch, int(k), int(k), self.hidden_dim)
        output_img = torch.movedim(output_img, 3, 1)
        shrink_img = self.patch_conv1(output_img)
        shrink_img = self.patch_conv2(shrink_img)
        shrink_img = self.patch_conv3(shrink_img) # [Batch, 1024, 1, 1]
        shrink_img = shrink_img.squeeze(-1).squeeze(-1)
        rewards = self.reward_head(shrink_img).squeeze(-1)  # (batch_size)

        return rewards

class TrajectoryRewardModel(nn.Module):
    def __init__(self,
                 d_model: int = 384,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 fusion_blocks: int = 4,
                 num_blocks: int = 4,
                 verbose: bool = True,
                 image_feature_extractor_name: str = "jmtzt/ijepa_vitg16_22k",
                 freeze_image_encoder: bool = True,
                 image_feature_extractor: nn.Module | None = None):
        super().__init__()
        # self.model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
        self.image_feature_extractor_name = image_feature_extractor_name
        self.freeze_image_encoder = freeze_image_encoder
        # self.model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        # self.model_name = "facebook/dinov3-vit7b16-pretrain-lvd1689m"

        # Load DINOv3
        if verbose:
            print("loading model", self.image_feature_extractor_name)
        self.image_feature_extractor = image_feature_extractor
        if self.image_feature_extractor is None:
            self.image_feature_extractor = AutoModel.from_pretrained(self.image_feature_extractor_name)
        self.patch_size = self.image_feature_extractor.config.patch_size
        self.image_dim = self.image_feature_extractor.config.hidden_size
        if self.freeze_image_encoder:
            # Important for DDP: frozen params must not require grad.
            for p in self.image_feature_extractor.parameters():
                p.requires_grad = False
            self.image_feature_extractor.eval()
        if verbose:
            print(self.image_feature_extractor)
            print("Patch size:", self.patch_size)  # 16
            print("Image hidden dim:", self.image_dim)
        self.d_model = d_model

        self.trajectory_transformer = TrajectoryTransformer(d_model=d_model,
                                                            num_blocks=num_blocks,
                                                            dropout=dropout)

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

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_image_encoder:
            self.image_feature_extractor.eval()
        return self

    def forward(self, pts: torch.Tensor, image_inputs, B=None, M=None) -> torch.Tensor:
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
            rewards: (B, M) tensor for 4D trajectory input, or (B*M,) tensor for already-flat 3D input
        """
        return_flat = False
        if len(pts.shape) == 4:
            B, M, K, _ = pts.shape
            pts_flat = pts.reshape(B * M, K, 2).float()
        elif len(pts.shape) == 3: # assuming already flat, we need B and M from outside
            if (B is None) or (M is None):
                raise ValueError("batch size B and trajectory count M are required when pts is already flat")
            pts_flat = pts.float()
            return_flat = True
        else:
            raise ValueError(f"reward model pts shape {pts.shape} mismatch; expected (B,M,K,2) or (B*M,K,2)")
        if pts_flat.shape[0] != B * M:
            raise ValueError(
                f"flat trajectory count {pts_flat.shape[0]} does not match B*M ({B}*{M}={B * M})"
            )
        x = self.trajectory_transformer(pts_flat)  # (B, K+1, D_model) with CLS at index 0]

        if self.freeze_image_encoder:
            with torch.no_grad():
                img_output = self.image_feature_extractor(**image_inputs)
        else:
            img_output = self.image_feature_extractor(**image_inputs)
        # original_patch_features = orig_output.last_hidden_state[:, 0, :] # same as: img_output.pooler_output

        img_tokens = img_output.last_hidden_state # [batch, 196, 1408]
        # img_tokens = img_tokens[:, 1 + self.num_register_tokens :, :] # [batch, 196, 768]

        image_batch_size, n_patches, embed = img_tokens.shape
        if image_batch_size != B:
            raise ValueError(
                f"image batch size {image_batch_size} does not match trajectory batch size {B}"
            )
        assert (embed == self.d_model) , f"embedding size {embed} does not match d_model {self.d_model}"

        img_tokens_exp  = img_tokens[:, None, :, :].expand(image_batch_size, M, n_patches, embed)
        img_tokens_flat = img_tokens_exp.reshape(image_batch_size * M, n_patches, embed)

        # Trajectory queries attend to image patch keys/values.
        for block in self.fusion:
            x = block(x, img_tokens_flat)

        # CLS readout for reward prediction.
        cls_feat = x[:, 0, :]
        rewards = self.reward_head(cls_feat).squeeze(-1)
        if return_flat:
            return rewards # [batch * M (number of trajectories)]
        return rewards.reshape(B, M)