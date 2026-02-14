"""
split_preferences.py
splitting each annotation into its own json file rather than havinga single json file per bag
this matches how the images are organized and maximizes utilities of dataloaders
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

def main():
    annotation_preferences_dir = "./annotations/preferences"
    annotation_save_root = "/media/jim/Ironwolf/datasets/scand_data/chop_annotations"
    split_json = "./annotations/test-train-split.json"
    if os.path.exists(split_json):
        with open(split_json, 'r') as f:
            test_train_split = json.load(f)
            print(f"{split_json} loaded, {len(test_train_split)} entries.")
    for bag_name in tqdm(test_train_split):
        split = test_train_split[bag_name]
        json_name = f"{bag_name}.json"
        annotation_file = os.path.join(annotation_preferences_dir, json_name)
        with open(annotation_file, 'r') as f:
            annotation_dict = json.load(f)
        assert annotation_dict['bag'] == f"{bag_name}.bag"
        for timestamp in annotation_dict['annotations_by_stamp']:
            folder_name = os.path.join(annotation_save_root, bag_name)
            if not os.path.exists(folder_name):
                os.makedirs(folder_name, exist_ok=True)
            with open(f"{folder_name}/{timestamp}.json", "w") as f:
                json.dump(annotation_dict['annotations_by_stamp'][timestamp], f)


if __name__ == "__main__":
    main()