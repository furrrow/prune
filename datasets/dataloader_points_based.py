from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _load_json_array(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON array in {path}, got {type(data).__name__}.")
    return data


def _parse_pair_key(key: str) -> Tuple[int, int]:
    try:
        first, second = key.split("_")
        i, j = int(first), int(second)
    except Exception as exc:  # pragma: no cover - defensive path
        raise ValueError(f"Invalid pairwise key '{key}', expected format 'i_j'.") from exc
    if i >= j:
        raise ValueError(f"Invalid pairwise key '{key}', expected i < j.")
    return i, j

def collate_points_based(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for PointBasedPreferenceDataset.
    Assumes fixed M, K and fixed number of pairwise comparisons per sample.
    """
    out: Dict[str, Any] = {
        "id": [b["id"] for b in batch],
        "bag": [b["bag"] for b in batch],
        "timestamp": [b["timestamp"] for b in batch],
        "image": torch.stack([b["image"] for b in batch], dim=0),  # (B, 3, H, W)
        "points": torch.stack([b["points"] for b in batch], dim=0),  # (B, M, K, C)
        "ranking": torch.stack([b["ranking"] for b in batch], dim=0),  # (B, M)
        "pair_i": torch.stack([b["pair_i"] for b in batch], dim=0),  # (B, P)
        "pair_j": torch.stack([b["pair_j"] for b in batch], dim=0),  # (B, P)
        "pair_target": torch.stack([b["pair_target"] for b in batch], dim=0),  # (B, P)
    }
    return out

class CHOPDatasetFull(Dataset):
    """
    Dataset for flat CHOP-preprocessed JSON produced by datasets/preprocess_scand_a_chop.py.

    One item corresponds to one image and all ranked trajectories for that image.
    """

    def __init__(
        self,
        annotations_path: str | Path,
        images_root: str | Path,
        image_size: Optional[Tuple[int, int]] = None,
        use_xy_only: bool = True,
    ) -> None:
        """
        Args:
            annotations_path: Path to train.json or test.json (flat array format).
            images_root: Root folder that contains bag subfolders referenced by image_path.
            image_size: Optional (height, width) resize target.
            use_xy_only: If True, use only x/y coordinates from 3D points.
        """
        self.annotations_path = Path(annotations_path)
        self.images_root = Path(images_root)
        self.image_size = image_size
        self.use_xy_only = use_xy_only

        self.samples = _load_json_array(self.annotations_path)
        self._validate_schema()

    def _validate_schema(self) -> None:
        if not self.samples:
            return
        required_fields = ("image_path", "paths", "ranking", "pairwise_map")
        sample = self.samples[0]
        missing = [field for field in required_fields if field not in sample]
        if missing:
            raise ValueError(
                f"Missing required fields in {self.annotations_path}: {missing}. "
                "Make sure preprocess_scand_a_chop.py output is being used."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, rel_image_path: str) -> torch.Tensor:
        img_path = self.images_root / rel_image_path
        try:
            image_pil = Image.open(img_path).convert("RGB")
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Image not found: {img_path}") from exc
        except Exception as exc:
            raise RuntimeError(f"Image unreadable: {img_path}") from exc

        if self.image_size is not None:
            h, w = self.image_size
            image_pil = image_pil.resize((w, h), resample=Image.BILINEAR)

        # Ensure writable contiguous memory for torch.from_numpy (avoids non-writable array warning).
        image_np = np.array(image_pil, dtype=np.uint8, copy=True)
        image = torch.from_numpy(image_np).permute(2, 0, 1).contiguous().float() / 255.0
        if image.ndim != 3 or image.shape[0] != 3:
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
        return image

    def _extract_points(self, paths: Dict[str, Dict[str, Any]], ranking: Sequence[int]) -> torch.Tensor:
        traj_list: List[torch.Tensor] = []
        for path_id in ranking:
            entry = paths[str(path_id)]
            pts = torch.tensor(entry["points"], dtype=torch.float32)
            if pts.ndim != 2 or pts.shape[1] < 2:
                raise ValueError(
                    f"Path '{path_id}' points must be shaped (K, >=2), got {pts.shape}."
                )
            pts = pts[:, :2] if self.use_xy_only else pts[:, :3]
            traj_list.append(pts)

        return torch.stack(traj_list, dim=0)  # (M, K, C)

    def _extract_pairwise(self, pairwise_map: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keys = sorted(pairwise_map.keys(), key=lambda s: tuple(int(x) for x in s.split("_")))
        pair_i: List[int] = []
        pair_j: List[int] = []
        pair_target: List[float] = []

        for key in keys:
            i, j = _parse_pair_key(key)
            winner = int(pairwise_map[key])
            if winner not in (i, j):
                raise ValueError(
                    f"Pair '{key}' has winner {winner}, expected one of ({i}, {j})."
                )
            pair_i.append(i)
            pair_j.append(j)
            pair_target.append(1.0 if winner == i else 0.0)

        return (
            torch.tensor(pair_i, dtype=torch.long),
            torch.tensor(pair_j, dtype=torch.long),
            torch.tensor(pair_target, dtype=torch.float32),
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        ranking = [int(x) for x in sample["ranking"]]

        item = {
            "id": sample.get("id"),
            "bag": sample.get("bag"),
            "timestamp": sample.get("timestamp"),
            "image": self._load_image(sample["image_path"]),
            "points": self._extract_points(sample["paths"], ranking),  # (M, K, 2|3)
            "ranking": torch.tensor(ranking, dtype=torch.long),  # (M,)
        }

        pair_i, pair_j, pair_target = self._extract_pairwise(sample["pairwise_map"])
        item["pair_i"] = pair_i
        item["pair_j"] = pair_j
        item["pair_target"] = pair_target
        return item
