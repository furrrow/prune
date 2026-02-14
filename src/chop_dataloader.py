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
# Ignore warnings
import warnings
# warnings.filterwarnings("ignore")

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

    def process_annotation_file(self, json_path: Path, num_points):
        with json_path.open("r") as f:
            raw = json.load(f)

        bag_name = Path(raw.get("bag", json_path.stem)).stem
        annotations = raw.get("annotations_by_stamp") or {}

        bag_dir = self.images_root / bag_name
        bag_prefix = Path(bag_name)
        processed: List[Dict[str, Any]] = []
        for stamp, annotation in annotations.items():
            rankings = _get_rankings(annotation)

        paths: Dict[str, PathDict] = annotation.get("paths") or {}
        path_0_data = _extract_path(paths.get(rankings[0]), num_points=num_points)
        path_1_data = _extract_path(paths.get(rankings[1]), num_points=num_points)

        image_filename = f"img_{stamp}.{self.image_ext}"
        if not (bag_dir / image_filename).is_file():
            print(f"Warning: missing image file {bag_dir / image_filename}, skipping sample.")
            return None
        image_path = bag_prefix / image_filename
        processed.append(
            {
                "timestamp": stamp,
                "frame_idx": annotation.get("frame_idx"),
                "robot_width": annotation.get("robot_width"),
                "image_path": str(image_path),
                "path_0": path_0_data,
                "path_1": path_1_data,
                "position": annotation.get("position"),
                "yaw": annotation.get("yaw"),
                "stop": annotation.get("stop", False),
            }
        )

        return processed

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
            points_list.append(pref_dict['paths'][str(rank)]['points'])
            left_boundaries.append(pref_dict['paths'][str(rank)]['left_boundary'])
            right_boundaries.append(pref_dict['paths'][str(rank)]['right_boundary'])
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