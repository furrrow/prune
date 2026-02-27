from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple
import numpy as np

Annotation = MutableMapping[str, Any]
PathDict = Dict[str, Any]


def _get_rankings(annotation: Annotation) -> Optional[List[int]]:
    """Return ordered path rankings as integers."""
    preference = annotation.get("preference", [])
    if not isinstance(preference, Iterable) or isinstance(preference, (str, bytes)):
        return None
    try:
        pref_list = [int(p) for p in preference]
    except (TypeError, ValueError):
        return None
    return pref_list if pref_list else None


def _build_pairwise_map(ranking: List[int]) -> Dict[str, int]:
    """
    Build pairwise winners keyed as "i_j" where i < j.
    Winner is the path id (i or j) that appears earlier in ranking.
    """
    rank_pos = {path_id: idx for idx, path_id in enumerate(ranking)}
    pairwise_map: Dict[str, int] = {}
    ids = sorted(ranking)
    for i, first_id in enumerate(ids):
        for second_id in ids[i + 1 :]:
            winner = first_id if rank_pos[first_id] < rank_pos[second_id] else second_id
            pairwise_map[f"{first_id}_{second_id}"] = winner
    return pairwise_map


def _resample_path(path: np.ndarray, k: int) -> np.ndarray:
    """
    Evenly resample a sequence of 3D points to length k using linear interpolation.
    Expects ``path`` shape (n, 3); returns float32 array shape (k, 3).
    """
    path = np.asarray(path, dtype=np.float32).reshape(-1, 3)
    if path.size == 0:
        return np.zeros((k, 3), dtype=np.float32)
    if len(path) == 1:
        return np.repeat(path, k, axis=0)

    deltas = path[1:] - path[:-1]
    seg_len = np.linalg.norm(deltas, axis=1)
    cum = np.concatenate([np.array([0.0], dtype=np.float32), np.cumsum(seg_len, dtype=np.float32)])
    total = cum[-1]
    if total == 0:
        return np.repeat(path[:1], k, axis=0)

    target = np.linspace(0.0, float(total), num=k, dtype=np.float32)
    out = np.empty((k, path.shape[1]), dtype=np.float32)
    for i, t in enumerate(target):
        j = np.searchsorted(cum, t, side="right") - 1
        j = int(np.clip(j, 0, len(seg_len) - 1))
        t0, t1 = cum[j], cum[j + 1]
        alpha = 0.0 if t1 == t0 else float((t - t0) / (t1 - t0))
        out[i] = path[j] * (1 - alpha) + path[j + 1] * alpha
    return out

def _get_yaws(points: np.ndarray) -> np.ndarray:
    """Compute yaw angles (in radians) for a sequence of 3D points."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    deltas = points[1:, :2] - points[:-1, :2]
    if deltas.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.arctan2(deltas[:, 1], deltas[:, 0])

def _extract_path(data: Dict[str, Any], num_points: int) -> Dict[str, Any]:

    # Resample to k+1, then drop the first (origin) so the stored points align with actions
    points_full = _resample_path(data.get("points", []), k=num_points+1)
    yaws = _get_yaws(points_full)
    points = points_full[1:]
    return {
        "points": points.tolist(),
        "left_boundary": _resample_path(data.get("left_boundary", []), k=num_points+1)[1:].tolist(),
        "right_boundary": _resample_path(data.get("right_boundary", []), k=num_points+1)[1:].tolist(),
        "yaws": yaws.tolist(),
    }

def _process_annotation_file(
    json_path: Path, images_root: Path, image_ext: str, num_points: int
) -> List[Dict[str, Any]]:
    with json_path.open("r") as f:
        raw = json.load(f)

    bag_name = Path(raw.get("bag", json_path.stem)).stem
    annotations = raw.get("annotations_by_stamp") or {}

    bag_dir = images_root / bag_name
    bag_prefix = Path(bag_name)
    processed: List[Dict[str, Any]] = []
    for stamp, annotation in annotations.items():
        rankings = _get_rankings(annotation)
        if rankings is None or len(rankings) != 4:
            continue

        paths: Dict[str, PathDict] = annotation.get("paths") or {}
        path_map: Dict[str, Dict[str, Any]] = {}
        missing_ranked_path = False
        for path_id in rankings:
            raw_path = paths.get(str(path_id))
            if not isinstance(raw_path, MutableMapping):
                missing_ranked_path = True
                break
            path_map[str(path_id)] = _extract_path(raw_path, num_points=num_points)
        if missing_ranked_path:
            continue

        image_filename = f"img_{stamp}.{image_ext}"
        if not (bag_dir / image_filename).is_file():
            print(f"Warning: missing image file {bag_dir / image_filename}, skipping sample.")
            continue
        image_path = bag_prefix / image_filename
        processed.append(
            {
                "id": f"{bag_name}_{stamp}",
                "bag": bag_name,
                "timestamp": stamp,
                "frame_idx": annotation.get("frame_idx"),
                "robot_width": annotation.get("robot_width"),
                "image_path": str(image_path),
                "paths": path_map,
                "ranking": rankings,
                "pairwise_map": _build_pairwise_map(rankings),
                "position": annotation.get("position"),
                "yaw": annotation.get("yaw"),
                "stop": annotation.get("stop", False),
            }
        )

    return processed


class _JsonArrayWriter:
    """Minimal streaming JSON array writer to avoid holding all data in memory."""

    def __init__(self, path: Path, pretty: bool = False):
        self.path = path
        self.pretty = pretty
        self._fh = path.open("w")
        self._empty = True

    def write(self, obj: Any) -> None:
        if self._empty:
            self._fh.write("[\n" if self.pretty else "[")
            json.dump(obj, self._fh, separators=(",", ":"), indent=2 if self.pretty else None)
            self._empty = False
        else:
            self._fh.write(",\n" if self.pretty else ",")
            json.dump(obj, self._fh, separators=(",", ":"), indent=2 if self.pretty else None)

    def close(self) -> None:
        if self._fh is None:
            return
        if self._empty:
            self._fh.write("[]\n")
        else:
            if self.pretty:
                self._fh.write("\n")
            self._fh.write("]\n")
        self._fh.close()
        self._fh = None

    def __enter__(self) -> "_JsonArrayWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def preprocess_scand(
    scand_dir: Path,
    images_root: Path,
    output_dir: Path,
    test_train_split_json: Path,
    image_ext: str = "jpg",
    default_split: str = "train",
    num_points: int = 8,
) -> Tuple[int, int]:
    """Generate per-split SCAND-A indices as flat sample lists."""
    split_map = json.load(test_train_split_json.open("r")) if test_train_split_json.exists() else {}
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.json"
    test_path = output_dir / "test.json"
    train_count = 0
    test_count = 0

    with _JsonArrayWriter(train_path, pretty=True) as train_writer, _JsonArrayWriter(test_path, pretty=True) as test_writer:
        for json_file in sorted(scand_dir.glob("*.json")):
            entries = _process_annotation_file(json_file, images_root, image_ext, num_points)
            if not entries:
                continue

            bag_name = Path(json_file).stem
            split = split_map.get(bag_name, default_split)
            writer = train_writer if split == "train" else test_writer

            for entry in entries:
                writer.write(entry)
                if writer is train_writer:
                    train_count += 1
                else:
                    test_count += 1

    return train_count, test_count

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess SCAND-A annotations into a flat index of image/trajectory pairs."
    )
    parser.add_argument(
        "--scand-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "annotations" / "preferences",
        help="Directory containing SCAND annotation JSON files.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "images",
        help="Root directory containing extracted SCAND images (organized by bag name).",
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "lora-data",
        help="Directory to write the train/test annotation indices.",
    )
    parser.add_argument(
        "--image-ext",
        type=str,
        default="png",
        help="Image extension to use when constructing image paths (e.g., jpg or png).",
    )
    parser.add_argument(
        "--test-train-split-json",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "annotations" / "test-train-split.json",
        help="Path to the JSON file defining the train/test split.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=8,
        help="Number of points to sample from each trajectory.",
    )
    args = parser.parse_args()

    train_count, test_count = preprocess_scand(
        scand_dir=args.scand_dir,
        images_root=args.images_root,
        output_dir=args.output_dir,
        test_train_split_json=args.test_train_split_json,
        image_ext=args.image_ext,
        num_points=args.num_points,
    )

    print(f"Wrote {train_count} train samples to {args.output_dir / 'train.json'} (flat)")
    print(f"Wrote {test_count} test samples to {args.output_dir / 'test.json'} (flat)")


if __name__ == "__main__":
    main()
