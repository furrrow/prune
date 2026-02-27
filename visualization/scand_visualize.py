"""
scand_visualize.py
used to visualize different trajectories from the CHOP + SCAND dataset.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple
import numpy as np
from preprocess_scand_a_chop import _process_annotation_file, _JsonArrayWriter

import os
import cv2
from dataclasses import dataclass
import time

from utils.vis_utils import point_to_traj, make_corridor_polygon, draw_polyline, draw_corridor, transform_points, \
    project_points_cam, load_calibration, clean_2d, project_clip, make_corridor_polygon_from_cam_lines
from utils.traj_utils import solve_arc_from_point


Annotation = MutableMapping[str, Any]
PathDict = Dict[str, Any]
# Colors (BGR)
RED = (0, 0, 255)    # RED
GREEN = (0, 255, 0)    # GREEN
YELLOW = (0, 255, 255)    # YELLOW

@dataclass
class FrameItem:
    idx: int
    img: np.ndarray
    position: np.ndarray
    velocity: float
    omega: float
    rotation: np.ndarray
    yaw: float

@dataclass
class PathItem:
    path_points: np.ndarray
    left_boundary: np.ndarray
    right_boundary: np.ndarray

def draw(frame_item: FrameItem, path_0: PathItem, path_1: PathItem, K: np.ndarray, dist, T_cam_from_base: np.ndarray,
         window_name: str = "SCAND preference vis"):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    current_img = frame_item.img
    if current_img is None:
        return
    img = current_img.copy()
    img_h, img_w = img.shape[:2]
    # points_2d_0 = clean_2d(
    #     project_clip(path_0.path_points, T_cam_from_base, K, dist, img_h, img_w, smooth_first=True),
    #     img_w, img_h)
    left_2d_0 = clean_2d(
        project_clip(path_0.left_boundary, T_cam_from_base, K, dist, img_h, img_w, smooth_first=True),
        img_w, img_h)
    right_2d_0 = clean_2d(
        project_clip(path_0.right_boundary, T_cam_from_base, K, dist, img_h, img_w, smooth_first=True),
        img_w, img_h)

    poly_2d_0 = make_corridor_polygon_from_cam_lines(left_2d_0, right_2d_0)
    # draw_polyline(img, points_2d, 2, color)
    # draw_corridor(img, poly_2d, left_2d, right_2d, fill_alpha=0.35, fill_color=RED, edge_thickness=2)
    draw_corridor(img, poly_2d_0, left_2d_0, right_2d_0, fill_alpha=0.35, fill_color=GREEN, edge_thickness=2)

    left_2d_1 = clean_2d(
        project_clip(path_1.left_boundary, T_cam_from_base, K, dist, img_h, img_w, smooth_first=True),
        img_w, img_h)
    right_2d_1 = clean_2d(
        project_clip(path_1.right_boundary, T_cam_from_base, K, dist, img_h, img_w, smooth_first=True),
        img_w, img_h)
    poly_2d_1 = make_corridor_polygon_from_cam_lines(left_2d_1, right_2d_1)
    # draw_polyline(img, points_2d, 2, color)
    draw_corridor(img, poly_2d_1, left_2d_1, right_2d_1, fill_alpha=0.35, fill_color=RED, edge_thickness=2)


    cv2.imshow(window_name, img)
    cv2.waitKey()

def visualize_scand(
    scand_dir: Path,
    images_root: Path,
    output_dir: Path,
    test_train_split_json: Path,
    image_ext: str = "jpg",
    default_split: str = "train",
    num_points: int = 8,
):
    """Generate per-split SCAND-A indices, grouping samples by bag within train/test files."""
    split_map = json.load(test_train_split_json.open("r")) if test_train_split_json.exists() else {}
    output_dir.mkdir(parents=True, exist_ok=True)

    """constants for visualization"""
    fx, fy, cx, cy = 640.0, 637.0, 640.0, 360.0  # SCAND Kinect intrinsics ### DO NOT CHANGE
    calib_path = "../config/tf.json"

    K, dist, T_base_from_cam = load_calibration(calib_path, fx, fy, cx, cy, mode="spot")
    T_cam_from_base = np.linalg.inv(T_base_from_cam)

    for json_file in tqdm(sorted(scand_dir.glob("*.json"))):
        entries = _process_annotation_file(json_file, images_root, image_ext, num_points)
        for idx, i_entry in enumerate(entries):
            image_full_path = os.path.join(images_root, i_entry['image_path'])
            frame_item = FrameItem(idx=idx, img=cv2.imread(image_full_path),
                                   position=np.array([0, 0]), velocity=1.0, omega=0.0,
                                   rotation=np.array([0, 0]), yaw=0.0)
            path_0 = PathItem(path_points=np.array(i_entry['path_0']['points']),
                                 left_boundary=np.array(i_entry['path_0']['left_boundary']),
                                 right_boundary=np.array(i_entry['path_0']['right_boundary']))
            path_1 = PathItem(path_points=np.array(i_entry['path_1']['points']),
                              left_boundary=np.array(i_entry['path_1']['left_boundary']),
                              right_boundary=np.array(i_entry['path_1']['right_boundary']))
            draw(frame_item, path_0, path_1, K , dist, T_cam_from_base)


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
        # default=Path(__file__).resolve().parent.parent / "data" / "images",
        default=Path("/media/jim/Ironwolf/datasets/scand_data/images"),
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

    train_count, test_count = visualize_scand(
        args.scand_dir,
        args.images_root,
        args.output_dir,
        args.test_train_split_json,
        args.image_ext,
        args.num_points,
    )

    print(f"Wrote {train_count} train samples to {args.output_dir / 'train.json'} (bag-grouped)")
    print(f"Wrote {test_count} test samples to {args.output_dir / 'test.json'} (bag-grouped)")


if __name__ == "__main__":
    main()
