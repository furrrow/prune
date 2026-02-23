"""
reward_model.py
reward model using the trajectory preferences
"""
import torch
import torch.nn as nn
from transformers import TorchAoConfig, AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision.transforms import v2

"""
DINOv3 related transforms, etc
see: https://github.com/facebookresearch/dinov3
"""
# only for LVD-1689M weights (pretrained on web images)
def make_transform(resize_size: int = 256):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


# examples of available DINOv3 models:
MODEL_DINOV3_VITS = "dinov3_vits16"
MODEL_DINOV3_VITSP = "dinov3_vits16plus"
MODEL_DINOV3_VITB = "dinov3_vitb16"
MODEL_DINOV3_VITL = "dinov3_vitl16"
MODEL_DINOV3_VITHP = "dinov3_vith16plus"
MODEL_DINOV3_VIT7B = "dinov3_vit7b16"
MODEL_TO_NUM_LAYERS = {
    MODEL_DINOV3_VITS: 12,
    MODEL_DINOV3_VITSP: 12,
    MODEL_DINOV3_VITB: 12,
    MODEL_DINOV3_VITL: 24,
    MODEL_DINOV3_VITHP: 32,
    MODEL_DINOV3_VIT7B: 40,
}


class PairwiseRewardModel(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1, use_cls=True, verbose=True):
        super().__init__()
        self.model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_cls = use_cls

        # Load DINOv3
        if verbose:
            print("loading model", self.model_name)
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.patch_size = self.model.config.patch_size
        if verbose:
            print(self.model)
            print("Patch size:", self.patch_size)  # 16
            print("Num register tokens:", self.model.config.num_register_tokens)  # 4
        self.hidden_dim = 384 # fixed for DINOv3? check this

        # Self-Attention Over Vision Features
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.num_heads,
                                                    batch_first=True, dropout=dropout)
        self.attn_norm = nn.LayerNorm(self.hidden_dim)

        # patch feature distillation
        self.patch_conv1 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=5, stride=2)
        self.patch_conv2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=2)
        self.patch_conv3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=2)

        # Reward Prediction Head
        self.reward_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1), )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights using Xavier uniform distribution."""
        for name, module in self.named_modules():
            if "vision_model" in name:
                continue
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)  # Zero-bias initialization

    def forward(self, orig_input, annotated_input):
        """
        original_img: (batch_size, 3, 224, 224)
        annotated_img: (batch_size, 3, 224, 224)
        """
        # handling these at the training script
        # orig_input = self.processor(images=original_img, return_tensors="pt")
        # annotated_input = self.processor(images=annotated_img, return_tensors="pt")
        # orig_input.data['pixel_values'].shape = torch.Size([batch_size, 3, 224, 224])
        with torch.no_grad():
            orig_output = self.model(**orig_input)
            annotated_output = self.model(**annotated_input)
        if self.use_cls:
            original_patch_features = orig_output.last_hidden_state[:, 0, :]
            annotated_patch_features = annotated_output.last_hidden_state[:, 0, :]
        else:
            # [batch_size, 196, 384]
            original_patch_features = orig_output.last_hidden_state[:, 1 + self.model.config.num_register_tokens:, :]
            annotated_patch_features = annotated_output.last_hidden_state[:, 1 + self.model.config.num_register_tokens:, :]
            batch_size, _, img_height, img_width = orig_input.pixel_values.shape
            patch_size = self.model.config.patch_size
            num_patches_height, num_patches_width = img_height // patch_size, img_width // patch_size
            # num_patches_flat = num_patches_height * num_patches_width
            # to unflatten: patch_features_flat.unflatten(1, (num_patches_height, num_patches_width))
            original_patch_features = original_patch_features.unflatten(1, (num_patches_height, num_patches_width))
            annotated_patch_features = annotated_patch_features.unflatten(1, (num_patches_height, num_patches_width))
            # channel first
            original_patch_features = torch.movedim(original_patch_features, 3, 1)
            annotated_patch_features = torch.movedim(annotated_patch_features, 3, 1)

            original_patch_features = self.patch_conv1(original_patch_features)
            original_patch_features = self.patch_conv2(original_patch_features)
            original_patch_features = self.patch_conv3(original_patch_features)
            original_patch_features = original_patch_features.squeeze(-1).squeeze(-1)

            annotated_patch_features = self.patch_conv1(annotated_patch_features)
            annotated_patch_features = self.patch_conv2(annotated_patch_features)
            annotated_patch_features = self.patch_conv3(annotated_patch_features)
            annotated_patch_features = annotated_patch_features.squeeze(-1).squeeze(-1)

        # Self-Attention on Vision Features
        attn_output, _ = self.multihead_attn(original_patch_features, annotated_patch_features, original_patch_features)  # Shape: (batch_size, num_patches, hidden_dim)
        attn_output = self.attn_norm(attn_output)  # Normalize After Self-Attention

        # Predict rewards for all 25 actions
        rewards = self.reward_head(attn_output).squeeze(-1)  # (batch_size)
        return rewards