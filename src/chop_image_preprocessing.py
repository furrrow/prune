"""
chop_image_preprocessing.py
dataloader class to load both the SCAND image and the preferred trajectory
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import glob
from tqdm import tqdm
import cv2
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple
from utils.vis_utils import draw_corridor, load_calibration, clean_2d, project_clip, make_corridor_polygon_from_cam_lines
from utils.vis_utils import BGR_color_dict as color_dict
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


def horizontal_flip_image(image):
    image = cv2.flip(image, 1)
    return image


def horizontal_flip_path(trajectory):
    trajectory = np.array(trajectory).copy()
    if len(trajectory.shape) == 2:
        trajectory[:, 1] *= -1
    elif len(trajectory.shape) == 3:
        trajectory[:, :, 1] *= -1
    else:
        raise RuntimeError(f"Error, horizontal_flip_path trajectory shape {trajectory.shape}")
    return trajectory


def overlay_trajectory(image, path_data, color,
                       T_cam_from_base, robot_name, K, dist, bypass):
    img = image.copy()
    if bypass:
        return img
    img_h, img_w = img.shape[:2]
    left_boundary = np.array(path_data['left_boundary'])
    right_boundary = np.array(path_data['right_boundary'])
    if (len(left_boundary.shape) < 2) or (len(right_boundary.shape) < 2):
        print(f"insufficient boundary at {path_data['timestamp']}, left_boundary: {left_boundary.shape} right_boundary: {right_boundary.shape}")
        return img
    left_2d = clean_2d(
        project_clip(left_boundary, T_cam_from_base[robot_name], K, dist, img_h, img_w),
        img_w, img_h)
    right_2d = clean_2d(
        project_clip(right_boundary, T_cam_from_base[robot_name], K, dist, img_h, img_w),
        img_w, img_h)

    poly_2d = make_corridor_polygon_from_cam_lines(left_2d, right_2d)
    draw_corridor(img, poly_2d, left_2d, right_2d, fill_alpha=0.5, fill_color=color, edge_thickness=2)
    return img


def main(mode="train"):
    with open('../config/setting.yaml', 'r') as f:
        settings = yaml.load(f, Loader=yaml.SafeLoader)
    project_home_dir = Path(__file__).resolve().parent.parent
    preference_root = settings['preference_root']
    image_root = settings['image_root']
    img_resize = settings["image_size"]
    overlay_img_save_root = "/media/jim/Ironwolf/datasets/scand_data/overlay_images"
    img_extension = 'png'
    calib_file = project_home_dir / settings['calibration_file']
    show_img = True
    if not os.path.exists(preference_root):
        print(f"ERROR, preference root not found in {preference_root}")
        exit()
    json_paths = Path(preference_root) / mode
    glob_list = sorted(glob.glob(f"{json_paths}/**/*.json", recursive=True))
    num_points = 8
    re_index = False
    with open(calib_file, "r") as f:
        calib_data = json.load(f)
    fx, fy, cx, cy = (calib_data['scand_kinect_intrinsics']['fx'], calib_data['scand_kinect_intrinsics']['fy'],
                      calib_data['scand_kinect_intrinsics']['cx'], calib_data['scand_kinect_intrinsics']['cy'])
    T_base_from_cam = {}
    T_cam_from_base = {}
    K, dist, T_base_from_cam["jackal"] = load_calibration(calib_file, fx, fy, cx, cy, mode="jackal")
    K, dist, T_base_from_cam["spot"] = load_calibration(calib_file, fx, fy, cx, cy, mode="spot")
    T_cam_from_base["jackal"] = np.linalg.inv(T_base_from_cam["jackal"])
    T_cam_from_base["spot"] = np.linalg.inv(T_base_from_cam["spot"])
    pair_scratch_file = f"{mode}_scratch.json"
    verified_pairs = {}
    if os.path.exists(pair_scratch_file) and (re_index is False):
        with open(pair_scratch_file, "r") as file:
            verified_pairs = json.load(file)
        print(f"{pair_scratch_file} loaded")
    else:
        for json_path in tqdm(glob_list, desc="verifying preference-image matching"):
            stem, json_file = os.path.split(json_path)
            stem, bag_name = os.path.split(stem)
            img_path = os.path.join(image_root, bag_name)
            img_name = f"img_{Path(json_file).stem}.{img_extension}"
            img_path = os.path.join(img_path, img_name)
            # verify preference exists:
            if os.path.exists(json_path) and os.path.exists(img_path):
                verified_pairs[str(len(verified_pairs))] = (json_path, img_path)
        with open(pair_scratch_file, "w") as f:
            json.dump(verified_pairs, f, indent=4)

    for idx in tqdm(verified_pairs, desc=f"processing {mode}"):
        json_path, img_path = verified_pairs[idx]
        try:
            with open(json_path, 'r') as f:
                pref_dict = json.load(f)
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

        for rank in ranking_list:
            path_data = _extract_path(pref_dict['paths'][str(rank)], num_points=num_points)
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

        if os.path.exists(img_path):
            image = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
        else:
            print(f"warning, idx {idx} img not found: {img_path}")
            # glob_list.pop(idx)
            return None
        # draw overlay of preferred trajectory
        if image is None:
            print("Error! Image is None, returning...", img_path)
            return None

        stop_pref = pref_dict['stop']
        color_key = "GREEN"
        img_stem, img_name = os.path.split(img_path)
        for rank in range(4):
            if stop_pref:
                skip = True if rank < 2 else False # return bare img if rank=0 or 1.
                if path is not None:
                    path = path  # just use the last path
                else:
                    Exception("Catch Error in chop_image_processing")
            else:
                path = pref_dict['paths'][str(ranking_list[rank])]
                skip = False
            overlay_img = overlay_trajectory(image, path, color=color_dict[color_key],
                                             T_cam_from_base=T_cam_from_base,
                                             robot_name=robot_name,
                                             K=K, dist=dist,
                                             bypass=skip)
            save_path = f"{overlay_img_save_root}/{str(rank)}/{bag_name}"
            cv2.resize(overlay_img, img_resize, interpolation=cv2.INTER_AREA)
            os.makedirs(save_path, exist_ok=True)
            # cv2.imwrite(f"{save_path}/{img_name}", overlay_img)
            if show_img:
                fig, ax = plt.subplots(1, 1)
                ax.imshow(overlay_img)
                plt.show(block=True)

if __name__ == "__main__":
    for mode in ["train", "test"]:
        main(mode)