import math
from typing import Optional, Tuple
import numpy as np
"""
traj_utils.py
borrowed from the visual annotations script
https://github.com/gershom96/VisualFrontiers-Annotation/blob/main/SCAND/traj_utils.py
"""
def solve_arc_from_point(x: float, y: float) -> Optional[Tuple[float, float]]:
    """
    Given target (x,y) in base_link, solve for (r, theta):
      x = r sinθ,  y = r (1 - cosθ)
      => r = (x^2 + y^2)/(2y),  θ = 2 atan2(y, x)
    """
    if abs(y) < 1e-6:
        # Straight line ahead
        r = 1e9
        theta = 0.0
        return r, theta
    r = (x*x + y*y) / (2.0*y)
    theta = 2.0 * math.atan2(y, x) 
    return r, theta

def arc_to_traj(r: float,
                theta: float,
                T_horizon: float,
                num: int,
                x_base: float,
                y_base: float) -> Tuple[np.ndarray, float, float]:

    T_end = T_horizon
    w = theta / T_end if T_end > 1e-9 else 0.0
    v = r*w if theta !=0.0 else x_base / T_end

    t = np.linspace(0.0, T_end, num)
    x = r * np.sin(w * t) if theta != 0.0 else (x_base / T_end) * t
    y = r * (1.0 - np.cos(w * t)) 
    z = np.zeros_like(x)
    pts_b = np.stack([x, y, z], axis=1)
    theta_samples = w * t
    return pts_b, v, w, t, theta_samples

def make_offset_path_to_point(traj_b: np.ndarray,
                              theta_samples: np.ndarray,
                              point: np.ndarray,
                              cum_dists: np.ndarray) -> np.ndarray:
    
    "Generate path offset with respect to user defined goal"

    # Compute offset at goal point
    closest_idx = np.argmin(np.linalg.norm(traj_b[:,:2] - point[:2], axis=1))

    extrapolate = False
    if closest_idx < 0.7 * len(traj_b):
        extrapolate = True
        
    yaw_goal = theta_samples[closest_idx]

    n_goal = np.array([-math.sin(yaw_goal), math.cos(yaw_goal)], dtype=float)
    offset_signed = float(np.dot(point[:2] - traj_b[closest_idx, :2], n_goal))

    if extrapolate:
        # print("Extrapolating offset path...")
        closest_idx = int(0.7*len(traj_b))

    cum_dists_goal = cum_dists[:closest_idx+1]
    
    x = traj_b[:closest_idx+1, 0]
    y = traj_b[:closest_idx+1, 1]

    offsets = offset_signed * cum_dists_goal / cum_dists_goal[-1]  # Linear ramp to goal

    theta_samples_1 = theta_samples[:closest_idx+1]
    n_x = -np.sin(theta_samples_1)
    n_y =  np.cos(theta_samples_1)

    x = x + offsets * n_x
    y = y + offsets * n_y

    z = np.zeros_like(x)

    offset_path = np.stack([x, y, z], axis=1)

    return offset_path

def check_left_right(traj_b: np.ndarray,
                     theta_samples: np.ndarray,
                     point: np.ndarray,
                     closest_idx: int,
                     offset: float) -> int:
    """Check if goal is to left or right of trajectory and adjust sign of offset accordingly."""
    
    x_traj, y_traj = traj_b[closest_idx,0], traj_b[closest_idx,1]
    yaw_traj = theta_samples[closest_idx]

    n_x = -np.sin(yaw_traj)
    n_y =  np.cos(yaw_traj)

    vec_to_goal = np.array([point[0]-x_traj, point[1]-y_traj])
    normal_vec = np.array([n_x, n_y])
    dot_product = np.dot(vec_to_goal, normal_vec)
    if dot_product < 0:
        offset = -offset  # Goal is to the right
    return offset

def make_offset_paths(traj_b: np.ndarray,
                      theta_samples: np.ndarray, 
                      offsets: np.ndarray):
    """Generate left/right offset paths for trajectory following."""

    x = traj_b[:, 0]
    y = traj_b[:, 1]
    # normal to heading (x-forward, y-left):
    n_x = -np.sin(theta_samples)
    n_y =  np.cos(theta_samples)

    xL = x + offsets * n_x
    yL = y + offsets * n_y
    xR = x - offsets * n_x
    yR = y - offsets * n_y

    z = np.zeros_like(x)
    left_o  = np.stack([xL, yL, z], axis=1)
    right_o = np.stack([xR, yR, z], axis=1)

    return left_o, right_o

def create_yaws_from_path(path_b: np.ndarray) -> np.ndarray:
    """Create yaw angles (radians) from a base_link path."""
    deltas = np.diff(path_b[:, :2], axis=0)  # (N-1, 2)
    yaws = np.arctan2(deltas[:, 1], deltas[:, 0])  # (N-1,)
    # Append last yaw to maintain same length
    if len(yaws) > 0:
        yaws = np.concatenate([yaws, yaws[-1:]], axis=0)
    else:
        yaws = np.array([0.0])
    return yaws