from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from PIL import Image

from models.reward_model_point_based import RewardModelPointBased


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
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config_point_based.yaml.")
    parser.add_argument("--verbose", action="store_true", help="Print model initialization details.")
    return parser.parse_args()

def main() -> None:
    terminal = False

    if terminal:
        args = parse_args()
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
