"""
chop_dataloader.py
dataloader class to load both the SCAND image and the preferred trajectory
"""
import os
import torch
import json
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import argparse
from pathlib import Path
import yaml
import glob
from tqdm import tqdm
import cv2
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple

# Ignore warnings
import warnings
# warnings.filterwarnings("ignore")

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


class ChopPreferenceDataset(Dataset):
    """CHOP preference dataset"""

    def __init__(self, preference_root, image_root, img_extension, split_json, mode, transform=None):
        """
        Arguments:
            preference_root (string): Path to the preference dataset.
            image_root (string): Directory of all SCAND images (not rosbags).\
            img_extension (string): Extension of image files, e.g. png
            split_json (string): Path to the JSON file defining the train/test split.
            mode (string): 'train' or 'test'
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.preference_root = preference_root
        self.image_root = image_root
        self.img_extension = img_extension
        self.split_json = split_json
        self.mode = mode
        self.transform = transform
        # if os.path.exists(self.split_json):
        #     with open(self.split_json, 'r') as f:
        #         self.bag_test_train_lookup = json.load(f)
        #         print(f"{self.split_json} loaded, {len(self.bag_test_train_lookup)} entries.")
        self.json_paths = Path(self.preference_root) / self.mode
        self.glob_list = sorted(glob.glob(f"{self.preference_root}/**/*.json", recursive=True))

    def __len__(self):
        return len(self.glob_list)

    def __getitem__(self, idx):
        """
        pref_dict keys:
        dict_keys(['frame_idx', 'robot_width', 'paths', 'preference', 'pairwise', 'position', 'yaw', 'stop'])
        pref_dict['paths']['0'].keys():
        dict_keys(['points', 'left_boundary', 'right_boundary', 'timestamp'])
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # preferences
        json_path = self.glob_list[idx]
        with open(json_path, 'r') as f:
            pref_dict = json.load(f)
        ranking_list = list(pref_dict['preference'])
        points_list = []
        left_boundaries = []
        right_boundaries = []
        for rank in ranking_list:
            path_data = _extract_path(pref_dict['paths'][str(rank)], num_points=10)
            points_list.append(path_data['points'])
            left_boundaries.append(path_data['left_boundary'])
            right_boundaries.append(path_data['right_boundary'])
        # images
        stem, json_file = os.path.split(json_path)
        stem, bag_name = os.path.split(stem)
        img_path = os.path.join(self.image_root, bag_name)
        img_name = f"img_{Path(json_file).stem}.{self.img_extension}"
        img_path = os.path.join(img_path, img_name)
        image = cv2.imread(img_path)

        sample = {
            'image': np.expand_dims(image, 0),
            'points': np.expand_dims(np.array(points_list), 0),
            'left_boundaries': np.expand_dims(np.array(left_boundaries), 0),
            'right_boundaries': np.expand_dims(np.array(right_boundaries), 0),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

def main():
    with open('../config/setting.yaml', 'r') as f:
        settings = yaml.load(f, Loader=yaml.SafeLoader)
    project_home_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Preprocess SCAND-A annotations into a flat index of image/trajectory pairs."
    )
    parser.add_argument(
        "--preference-root",
        type=Path,
        default=settings['scand_preference_root'],
        help="Directory containing SCAND annotation JSON files.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=settings['scand_img_root'],
        help="Root directory containing extracted SCAND images (organized by bag name)",
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
        default=project_home_dir / "data" / "annotations" / "test-train-split.json",
        help="Path to the JSON file defining the train/test split.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="train or test",
    )
    args = parser.parse_args()

    my_dataset = ChopPreferenceDataset(preference_root=args.preference_root,
                                      image_root=args.image_root,
                                      img_extension=args.image_ext,
                                      split_json=args.test_train_split_json,
                                      mode=args.mode
                                      )
    for i, sample in enumerate(my_dataset):
        print(i, sample['image'].shape, sample['points'].shape)


if __name__ == "__main__":
    main()