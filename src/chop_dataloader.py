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
from tqdm import tqdm
# Ignore warnings
import warnings
# warnings.filterwarnings("ignore")

class ChopPreferenceDataset(Dataset):
    """CHOP preference dataset"""

    def __init__(self, preference_dir, image_root, img_extension, split_json, mode, transform=None):
        """
        Arguments:
            preference_dir (string): Path to the preference dataset.
            image_root (string): Directory of all SCAND images (not rosbags).\
            img_extension (string): Extension of image files, e.g. png
            split_json (string): Path to the JSON file defining the train/test split.
            mode (string): 'train' or 'test'
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.preference_dir = preference_dir
        self.image_root = image_root
        self.img_extension = img_extension
        self.split_json = split_json
        self.mode = mode
        self.transform = transform
        if os.path.exists(self.split_json):
            with open(self.split_json, 'r') as f:
                self.bag_test_train_lookup = json.load(f)
                print(f"{self.split_json} loaded, {len(self.bag_test_train_lookup)} entries.")

        self.entry_save_json = f"../data/annotations/{mode}_dict.json"
        self.entry_dict = {}
        if os.path.exists(self.entry_save_json):
            with open(self.entry_save_json, 'r') as f:
                self.entry_dict = json.load(f)
                print(f"{self.entry_save_json} loaded, {len(self.entry_dict)} entries.")
        else:
            self.generate_dataset_mapping()
        self.timestamp_list = list(self.entry_dict.keys())

    def generate_dataset_mapping(self):
        """
        generates a complete mapping between each annotation entry (by timestamp) and its matching bag file
        example: self.entry_dict[''1635452200233731230''] = 'A_Jackal_AHG_Library_Thu_Oct_28_1.bag'

        :return: None
        """
        print(f"generating entry mapping for {self.entry_save_json}...")
        for json_file in tqdm(sorted(self.preference_dir.glob("*.json"))):
            with json_file.open("r") as f:
                raw = json.load(f)
                bag_name = Path(raw.get("bag", json_file.stem)).stem
                if self.bag_test_train_lookup[bag_name] != self.mode:
                    continue
                timestamp_list = list(raw['annotations_by_stamp'].keys())
                for timestamp in timestamp_list:
                    self.entry_dict[timestamp] = bag_name
        with open(self.entry_save_json, "w") as f:
            json.dump(self.entry_dict, f)
            print(f"entry mapping saved to {self.entry_save_json}")

    def __len__(self):
        return len(self.entry_save_json)

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
        if torch.is_tensor(idx):
            idx = idx.tolist()

        bag_name = self.timestamp_list[idx]

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

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
        "--preference-dir",
        type=Path,
        default= project_home_dir / "data" / "annotations" / "preferences",
        help="Directory containing SCAND annotation JSON files.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        # default=project_home_dir / "data" / "images",
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

    my_dataset = ChopPreferenceDataset(preference_dir=args.preference_dir,
                                      image_root=args.image_root,
                                      img_extension=args.image_ext,
                                      split_json=args.test_train_split_json,
                                      mode=args.mode
                                      )
    for i, sample in enumerate(my_dataset):
        print(i, sample['image'].shape, sample['landmarks'].shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_landmarks(**sample)


if __name__ == "__main__":
    main()