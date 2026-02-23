"""
chop_dataloader.py
dataloader class to load both the SCAND image and the preferred trajectory
"""
import os
import torch
import json

from cv2 import Mat
from numpy import dtype, floating, integer, ndarray
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
from utils.vis_utils import draw_corridor, load_calibration, clean_2d, project_clip, make_corridor_polygon_from_cam_lines
from utils.vis_utils import color_dict
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

    def __init__(self, preference_root, image_root, img_extension, calib_file,
                 split_json, mode, num_points, verbose, plot_imgs, transform=None):
        """
        Arguments:
            preference_root (string): Path to the preference dataset.
            image_root (string): Directory of all SCAND images (not rosbags).\
            img_extension (string): Extension of image files, e.g. png
            calib_file (string): location of calibration file for intrinsics & extrinsics
            split_json (string): Path to the JSON file defining the train/test split.
            mode (string): 'train' or 'test'
            num_points (int): number of points per trajectory to resample to
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.preference_root = preference_root
        self.image_root = image_root
        self.img_extension = img_extension
        self.calib_file = calib_file
        self.split_json = split_json
        self.mode = mode
        self.verbose = verbose
        self.plot_imgs = plot_imgs
        self.transform = transform
        # if os.path.exists(self.split_json):
        #     with open(self.split_json, 'r') as f:
        #         self.bag_test_train_lookup = json.load(f)
        #         print(f"{self.split_json} loaded, {len(self.bag_test_train_lookup)} entries.")
        self.json_paths = Path(self.preference_root) / self.mode
        self.glob_list = sorted(glob.glob(f"{self.json_paths}/**/*.json", recursive=True))
        self.num_points = num_points
        with open(self.calib_file, "r") as f:
            calib_data = json.load(f)
        fx, fy, cx, cy = (calib_data['scand_kinect_intrinsics']['fx'], calib_data['scand_kinect_intrinsics']['fy'],
                          calib_data['scand_kinect_intrinsics']['cx'], calib_data['scand_kinect_intrinsics']['cy'])
        self.T_base_from_cam = {}
        self.T_cam_from_base = {}

        self.K, self.dist, self.T_base_from_cam["jackal"] = load_calibration(self.calib_file, fx, fy, cx, cy,
                                                                             mode="jackal")
        self.K, self.dist, self.T_base_from_cam["spot"] = load_calibration(self.calib_file, fx, fy, cx, cy, mode="spot")
        self.T_cam_from_base["jackal"] = np.linalg.inv(self.T_base_from_cam["jackal"])
        self.T_cam_from_base["spot"] = np.linalg.inv(self.T_base_from_cam["spot"])

    def __len__(self):
        return len(self.glob_list)

    def __getitem__(self, idx, pick_mode="two"):
        """
        pick_mode:
            two: randomly pick two where first trajectory is ranked higher than the next
            all: return all four
        pref_dict keys:
        dict_keys(['frame_idx', 'robot_width', 'paths', 'preference', 'pairwise', 'position', 'yaw', 'stop'])
        pref_dict['paths']['0'].keys():
        dict_keys(['points', 'left_boundary', 'right_boundary', 'timestamp'])
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # preferences
        json_path = self.glob_list[idx]
        try:
            with open(json_path, 'r') as f:
                pref_dict = json.load(f)
                # print(f"opened {json_path}")
        except FileNotFoundError:
            print(f"File not found {json_path}")

        except json.JSONDecodeError as e:
            print(f"Invalid JSON format {json_path}")
            print(f"Line {e.lineno}, Column {e.colno}")
            print(e)

        except Exception as e:
            print(f"Unexpected error in loading {json_path}", e)
        ranking_list = list(pref_dict['preference'])
        points_list = []
        left_boundaries = []
        right_boundaries = []

        # pick two random rankings where the first one is better ranked than the next,
        if pick_mode == "two":
            first_trajectory = np.random.randint(0, 3)
            second_trajectory = first_trajectory + 1
            ranking_list = [str(first_trajectory), str(second_trajectory)]

        for rank in ranking_list:
            path_data = _extract_path(pref_dict['paths'][str(rank)], num_points=self.num_points)
            points_list.append(path_data['points'])
            left_boundaries.append(path_data['left_boundary'])
            right_boundaries.append(path_data['right_boundary'])
        # images
        stem, json_file = os.path.split(json_path)
        stem, bag_name = os.path.split(stem)
        if "Jackal" in bag_name:
            robot_name="jackal"
        elif "Spot" in bag_name:
            robot_name="spot"
        else:
            raise ValueError('Error, robot type unclear.')
        img_path = os.path.join(self.image_root, bag_name)
        img_name = f"img_{Path(json_file).stem}.{self.img_extension}"
        img_path = os.path.join(img_path, img_name)
        if os.path.exists(img_path):
            image = cv2.imread(img_path)
        else:
            print(f"warning, idx {idx} img not found: {img_path}")
            # self.glob_list.pop(idx)
            return None
        # draw overlay of preferred trajectory
        if self.verbose:
            print("ranking", ranking_list, "points len:", len(pref_dict['paths'][str(ranking_list[0])]['points']))
        # path_data = _extract_path(pref_dict['paths'][str(ranking_list[0])], num_points=self.num_points)
        path_data = pref_dict['paths'][ranking_list[0]]
        stop_pref = pref_dict['stop']
        pref_img = self.overlay_trajectory(image, path_data, color=color_dict['GREEN'], robot_name=robot_name, bypass=stop_pref)
        # draw overlay of bad trajectory
        # path_data = _extract_path(pref_dict['paths'][str(ranking_list[1])], num_points=self.num_points)
        path_data = pref_dict['paths'][ranking_list[1]]
        rej_img = self.overlay_trajectory(image, path_data, color=color_dict['RED'], robot_name=robot_name, bypass=stop_pref)
        if self.plot_imgs:
            fig, ax = plt.subplots(2, 1)
            pref_view = cv2.cvtColor(pref_img, cv2.COLOR_BGR2RGB)
            rej_view = cv2.cvtColor(rej_img, cv2.COLOR_BGR2RGB)
            ax[0].imshow(pref_view)
            ax[1].imshow(rej_view)
            plt.show(block=True)

        sample = {
            'image': image,
            'preferred': pref_img,
            'rejected': rej_img,
            'points': np.array(points_list),
            'left_boundaries': np.array(left_boundaries),
            'right_boundaries': np.array(right_boundaries),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def overlay_trajectory(self, image, path_data, color, robot_name, bypass):

        img = image.copy()
        if bypass:
            return img
        img_h, img_w = img.shape[:2]
        left_boundary = np.array(path_data['left_boundary'])
        right_boundary = np.array(path_data['right_boundary'])
        if (len(left_boundary.shape) < 2) or (len(right_boundary.shape) < 2):
            print("insufficient boundary in")
            print(path_data)
            return img
        left_2d = clean_2d(
            project_clip(left_boundary, self.T_cam_from_base[robot_name], self.K, self.dist, img_h, img_w),
            img_w, img_h)
        right_2d = clean_2d(
            project_clip(right_boundary, self.T_cam_from_base[robot_name], self.K, self.dist, img_h, img_w),
            img_w, img_h)

        poly_2d = make_corridor_polygon_from_cam_lines(left_2d, right_2d)
        draw_corridor(img, poly_2d, left_2d, right_2d, fill_alpha=0.35, fill_color=color, edge_thickness=2)
        return img

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
        "--calibration-file",
        type=Path,
        default=settings['calibration_file'],
        help="Calibration file for camera intrinsics & extrinsics",
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
        default="test",
        help="train or test",
    )
    parser.add_argument(
        "--num-points",
        type=str,
        default=10,
        help="number of points to resample for each trajectory",
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default=True,
        help="show print statements",
    )
    parser.add_argument(
        "--plot-imgs",
        type=str,
        default=True,
        help="plot dataloader graphs, set to false unless debug",
    )
    args = parser.parse_args()

    my_dataset = ChopPreferenceDataset(preference_root=args.preference_root,
                                       image_root=args.image_root,
                                       calib_file=args.calibration_file,
                                       img_extension=args.image_ext,
                                       split_json=args.test_train_split_json,
                                       mode=args.mode,
                                       verbose=args.verbose,
                                       plot_imgs=args.plot_imgs,
                                       num_points=args.num_points,
                                      )
    for i, sample in enumerate(my_dataset):
        print(i, "image shape:", sample['image'].shape,
              "points shape:", sample['points'].shape,
              )
        break

if __name__ == "__main__":
    main()