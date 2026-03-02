from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lmdb
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


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
        use_lmdb: bool = False,
        lmdb_path: Optional[str | Path] = None,
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
        self.use_lmdb = use_lmdb
        self.lmdb_path = Path(lmdb_path) if lmdb_path is not None else None
        self._lmdb_env: Optional[lmdb.Environment] = None

        self.samples = _load_json_array(self.annotations_path)
        self._validate_schema()
        self._init_lmdb()

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

    def _init_lmdb(self) -> None:
        if not self.use_lmdb:
            return
        if self.lmdb_path is None:
            raise ValueError("use_lmdb=True requires lmdb_path.")
        if not self.lmdb_path.exists():
            raise FileNotFoundError(
                f"LMDB cache not found at {self.lmdb_path}. Build it first using CHOPDatasetFull.build_lmdb_cache(...)."
            )

    def _get_lmdb_env(self) -> lmdb.Environment:
        if self._lmdb_env is None:
            assert self.lmdb_path is not None
            self._lmdb_env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=2048,
                subdir=self.lmdb_path.is_dir(),
            )
        return self._lmdb_env

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

    @staticmethod
    def _encode_cache_record(
        sample: Dict[str, Any],
        images_root: Path,
        image_size: Optional[Tuple[int, int]],
        use_xy_only: bool,
    ) -> bytes:
        ranking = [int(x) for x in sample["ranking"]]
        paths = sample["paths"]
        points_list: List[np.ndarray] = []
        for path_id in ranking:
            pts = np.asarray(paths[str(path_id)]["points"], dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] < 2:
                raise ValueError(f"Path '{path_id}' points must be shaped (K, >=2), got {pts.shape}.")
            pts = pts[:, :2] if use_xy_only else pts[:, :3]
            points_list.append(pts)
        points = np.stack(points_list, axis=0)

        keys = sorted(sample["pairwise_map"].keys(), key=lambda s: tuple(int(x) for x in s.split("_")))
        pair_i: List[int] = []
        pair_j: List[int] = []
        pair_target: List[float] = []
        for key in keys:
            i, j = _parse_pair_key(key)
            winner = int(sample["pairwise_map"][key])
            if winner not in (i, j):
                raise ValueError(f"Pair '{key}' has winner {winner}, expected one of ({i}, {j}).")
            pair_i.append(i)
            pair_j.append(j)
            pair_target.append(1.0 if winner == i else 0.0)

        img_path = images_root / sample["image_path"]
        image_pil = Image.open(img_path).convert("RGB")
        if image_size is not None:
            h, w = image_size
            image_pil = image_pil.resize((w, h), resample=Image.BILINEAR)
        image = np.array(image_pil, dtype=np.uint8, copy=True)  # H, W, 3

        record = {
            "id": sample.get("id"),
            "bag": sample.get("bag"),
            "timestamp": sample.get("timestamp"),
            "ranking": np.asarray(ranking, dtype=np.int64),
            "points": points,
            "pair_i": np.asarray(pair_i, dtype=np.int64),
            "pair_j": np.asarray(pair_j, dtype=np.int64),
            "pair_target": np.asarray(pair_target, dtype=np.float32),
            "image": image,
        }
        return pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def build_lmdb_cache(
        cls,
        annotations_path: str | Path,
        images_root: str | Path,
        lmdb_path: str | Path,
        image_size: Optional[Tuple[int, int]] = None,
        use_xy_only: bool = True,
        map_size_bytes: int = 128 * 1024 * 1024 * 1024,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        annotations_path = Path(annotations_path)
        images_root = Path(images_root)
        lmdb_path = Path(lmdb_path)
        if lmdb_path.exists() and not overwrite:
            return

        lmdb_path.parent.mkdir(parents=True, exist_ok=True)
        samples = _load_json_array(annotations_path)
        env = lmdb.open(
            str(lmdb_path),
            map_size=map_size_bytes,
            subdir=lmdb_path.is_dir(),
            lock=True,
            readonly=False,
            meminit=False,
            map_async=True,
        )
        try:
            commit_every = 1000
            txn = env.begin(write=True)
            txn.put(b"length", str(len(samples)).encode("utf-8"))
            iterator = tqdm(
                enumerate(samples),
                total=len(samples),
                desc=f"Building LMDB ({annotations_path.name})",
                disable=not verbose,
            )
            for idx, sample in iterator:
                key = f"{idx:09d}".encode("ascii")
                val = cls._encode_cache_record(
                    sample=sample,
                    images_root=images_root,
                    image_size=image_size,
                    use_xy_only=use_xy_only,
                )
                txn.put(key, val)
                if (idx + 1) % commit_every == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                    txn.put(b"length", str(len(samples)).encode("utf-8"))
            txn.commit()
            env.sync()
            if verbose:
                print(f"LMDB cache ready: {lmdb_path} ({len(samples)} samples)")
        finally:
            env.close()

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
        if self.use_lmdb:
            env = self._get_lmdb_env()
            key = f"{idx:09d}".encode("ascii")
            with env.begin(write=False) as txn:
                blob = txn.get(key)
            if blob is None:
                raise KeyError(f"LMDB entry not found for index {idx}")
            rec = pickle.loads(blob)
            image = torch.from_numpy(rec["image"]).permute(2, 0, 1).contiguous().float() / 255.0
            return {
                "id": rec["id"],
                "bag": rec["bag"],
                "timestamp": rec["timestamp"],
                "image": image,
                "points": torch.from_numpy(rec["points"]),
                "ranking": torch.from_numpy(rec["ranking"]),
                "pair_i": torch.from_numpy(rec["pair_i"]),
                "pair_j": torch.from_numpy(rec["pair_j"]),
                "pair_target": torch.from_numpy(rec["pair_target"]),
            }

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
