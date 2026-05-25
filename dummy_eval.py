"""
eval.py

"""
import torch
import torch.optim as optim
import os
import time
import datetime
import wandb
import json
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import yaml
import os

from src.chop_dataloader import ChopPreferenceDataset
from src.reward_model import RewardModel
from src.loss_fn import bradley_terry_loss
from utils.vis_utils import clean_2d, project_clip


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _prepare_image_for_plot(image):
    image = _to_numpy(image)
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
    return image


def infer_robot_name(dataset, sample_idx):
    pair = dataset.verified_pairs.get(str(sample_idx))
    if pair is None:
        return "spot"

    path_text = " ".join(pair)
    if "Jackal" in path_text:
        return "jackal"
    if "Spot" in path_text:
        return "spot"
    return "spot"

dummy_points = torch.tensor(
    [[[2.0970e-01, -3.6237e-03, 0.0000e+00],
      [4.1943e-01, -1.4998e-03, 0.0000e+00],
      [6.2915e-01, 9.0028e-04, 0.0000e+00],
      [8.3888e-01, 3.5398e-03, 0.0000e+00],
      [1.0486e+00, 6.0044e-03, 0.0000e+00],
      [1.2583e+00, 8.3039e-03, 0.0000e+00],
      [1.4681e+00, 1.0401e-02, 0.0000e+00],
      [1.6778e+00, 1.2498e-02, 0.0000e+00]],
     [[2.0945e-01, 5.2138e-03, 0.0000e+00],
      [4.1870e-01, 1.6165e-02, 0.0000e+00],
      [6.2793e-01, 2.7396e-02, 0.0000e+00],
      [8.3715e-01, 3.8866e-02, 0.0000e+00],
      [1.0464e+00, 5.0163e-02, 0.0000e+00],
      [1.2556e+00, 6.1295e-02, 0.0000e+00],
      [1.4649e+00, 7.2225e-02, 0.0000e+00],
      [1.6741e+00, 8.3154e-02, 0.0000e+00]],
     [[2.1058e-01, -3.5210e-02, 0.0000e+00],
      [4.2205e-01, -6.4667e-02, 0.0000e+00],
      [6.3356e-01, -9.3901e-02, 0.0000e+00],
      [8.4509e-01, -1.2294e-01, 0.0000e+00],
      [1.0566e+00, -1.5217e-01, 0.0000e+00],
      [1.2681e+00, -1.8156e-01, 0.0000e+00],
      [1.4795e+00, -2.1114e-01, 0.0000e+00],
      [1.6910e+00, -2.4072e-01, 0.0000e+00]],
     [[2.0832e-01, 4.5858e-02, 0.0000e+00],
      [4.1533e-01, 9.7360e-02, 0.0000e+00],
      [6.2229e-01, 1.4908e-01, 0.0000e+00],
      [8.2919e-01, 2.0099e-01, 0.0000e+00],
      [1.0361e+00, 2.5271e-01, 0.0000e+00],
      [1.2431e+00, 3.0428e-01, 0.0000e+00],
      [1.4502e+00, 3.5565e-01, 0.0000e+00],
      [1.6572e+00, 4.0703e-01, 0.0000e+00]],
     [[2.4595e-01, -3.9302e-03, 0.0000e+00],
      [4.9186e-01, -9.9386e-03, 0.0000e+00],
      [7.3777e-01, -1.6369e-02, 0.0000e+00],
      [9.8368e-01, -2.2585e-02, 0.0000e+00],
      [1.2296e+00, -2.8527e-02, 0.0000e+00],
      [1.4755e+00, -3.4321e-02, 0.0000e+00],
      [1.7214e+00, -3.9956e-02, 0.0000e+00],
      [1.9674e+00, -4.5483e-02, 0.0000e+00]],
     [[2.4593e-01, -4.7674e-03, 0.0000e+00],
      [4.9183e-01, -1.1613e-02, 0.0000e+00],
      [7.3771e-01, -1.8880e-02, 0.0000e+00],
      [9.8360e-01, -2.5933e-02, 0.0000e+00],
      [1.2295e+00, -3.2712e-02, 0.0000e+00],
      [1.4754e+00, -3.9343e-02, 0.0000e+00],
      [1.7213e+00, -4.5816e-02, 0.0000e+00],
      [1.9672e+00, -5.2179e-02, 0.0000e+00]],
     [[2.4542e-01, -3.2770e-02, 0.0000e+00],
      [4.9057e-01, -6.7604e-02, 0.0000e+00],
      [7.3566e-01, -1.0284e-01, 0.0000e+00],
      [9.8079e-01, -1.3785e-01, 0.0000e+00],
      [1.2259e+00, -1.7259e-01, 0.0000e+00],
      [1.4711e+00, -2.0719e-01, 0.0000e+00],
      [1.7163e+00, -2.4163e-01, 0.0000e+00],
      [1.9616e+00, -2.7594e-01, 0.0000e+00]],
     [[2.4644e-01, 2.3181e-02, 0.0000e+00],
      [4.9308e-01, 4.4287e-02, 0.0000e+00],
      [7.3975e-01, 6.4990e-02, 0.0000e+00],
      [9.8641e-01, 8.5914e-02, 0.0000e+00],
      [1.2330e+00, 1.0711e-01, 0.0000e+00],
      [1.4797e+00, 1.2845e-01, 0.0000e+00],
      [1.7263e+00, 1.4996e-01, 0.0000e+00],
      [1.9729e+00, 1.7158e-01, 0.0000e+00]],
     [[2.3334e-01, 1.6606e-02, 0.0000e+00],
      [4.6672e-01, 3.2838e-02, 0.0000e+00],
      [7.0010e-01, 4.8887e-02, 0.0000e+00],
      [9.3347e-01, 6.5180e-02, 0.0000e+00],
      [1.1668e+00, 8.1746e-02, 0.0000e+00],
      [1.4002e+00, 9.8384e-02, 0.0000e+00],
      [1.6335e+00, 1.1520e-01, 0.0000e+00],
      [1.8668e+00, 1.3210e-01, 0.0000e+00]]]
)

def overlay_paths_on_image(image, trajectories, rewards, K, dist, T_cam_from_base):
    """
    Project base-frame trajectory centerlines into image pixels, following the
    projection flow used by src/scand_visualize.py.
    """
    overlay = image.copy()
    img_h, img_w = overlay.shape[:2]
    best_idx = int(np.argmax(rewards))
    reward_labels = []

    for idx, trajectory in enumerate(trajectories):
        trajectory_xyz = np.concatenate(
            [trajectory, np.zeros((*trajectory.shape[:-1], 1), dtype=trajectory.dtype)],
            axis=-1,
        )
        points_2d = clean_2d(
            project_clip(trajectory_xyz.copy(), T_cam_from_base, K, dist, img_h, img_w, smooth_first=True),
            img_w,
            img_h,
        )
        if len(points_2d) < 2:
            continue

        color = (255, 0, 0) if idx == best_idx else (0, 255, 0)
        thickness = 4 if idx == best_idx else 2
        alpha = 0.9 if idx == best_idx else 0.35
        layer = overlay.copy()
        pts = np.round(points_2d).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(layer, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        overlay = cv2.addWeighted(layer, alpha, overlay, 1.0 - alpha, 0)
        reward_labels.append({
            "label_lines": [f"{idx}", f"rew {rewards[idx]:.2f}"],
            "anchor": pts[-1, 0],
            "color": color,
        })

    reward_labels.sort(key=lambda item: item["anchor"][0])
    if reward_labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        text_thickness = 1
        pad = 4
        gap = 6
        bg_alpha = 0.55
        line_gap = 4

        text_sizes = [
            cv2.getTextSize(line, font, font_scale, text_thickness)
            for item in reward_labels
            for line in item["label_lines"]
        ]
        text_w = max((size[0][0] for size in text_sizes), default=80)
        text_h = max((size[0][1] for size in text_sizes), default=15)
        baseline = max((size[1] for size in text_sizes), default=5)
        max_lines = max((len(item["label_lines"]) for item in reward_labels), default=1)
        box_w = text_w + 2 * pad
        box_h = max_lines * text_h + (max_lines - 1) * line_gap + baseline + 2 * pad

        top_y = min(int(item["anchor"][1]) for item in reward_labels)
        box_top = int(np.clip(top_y - box_h - 90, 0, max(0, img_h - box_h - 1)))
        total_row_w = len(reward_labels) * box_w + max(0, len(reward_labels) - 1) * gap
        leftmost_anchor_x = min(int(item["anchor"][0]) for item in reward_labels)
        leftmost_anchor_x = min(leftmost_anchor_x, img_w // 3)
        max_start_x = max(0, img_w - total_row_w - 1)
        current_x = int(np.clip(leftmost_anchor_x - box_w // 2, 0, max_start_x))

        for item in reward_labels:
            label_lines = item["label_lines"]
            anchor = item["anchor"]
            color = item["color"]
            x1 = current_x
            y1 = box_top
            x2 = x1 + box_w
            y2 = y1 + box_h
            current_x = x2 + gap

            anchor_xy = (int(anchor[0]), int(anchor[1]))
            label_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.line(overlay, anchor_xy, label_center, color=color, thickness=1, lineType=cv2.LINE_AA)
            cv2.circle(overlay, anchor_xy, radius=2, color=color, thickness=-1)

            box_roi = overlay[y1:y2, x1:x2]
            black_fill = np.zeros_like(box_roi)
            cv2.addWeighted(black_fill, bg_alpha, box_roi, 1 - bg_alpha, 0, dst=box_roi)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=1)
            for line_idx, line in enumerate(label_lines):
                line_y = y1 + pad + text_h + line_idx * (text_h + line_gap)
                cv2.putText(
                    overlay,
                    line,
                    (x1 + pad, line_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    text_thickness,
                    cv2.LINE_AA,
                )

    return overlay


def plot_trajectory(
        image,
        trajectories,
        rewards,
        K=None,
        dist=None,
        T_cam_from_base=None,
        save_path=None,
        show=True,
        title="trajectory visualization"):
    """
    Plot candidate trajectories and the corresponding observation image.

    Args:
        image: Observation image for one sample.
        trajectories: Tensor/array with shape (num_trajectories, trajectory_length, 2).
        rewards: Tensor/array with shape (num_trajectories,).
        K: Camera intrinsic matrix. Required for image overlays.
        dist: Camera distortion coefficients. May be None.
        T_cam_from_base: Base-to-camera transform. Required for image overlays.
        save_path: Optional path to save the figure. If omitted, the figure is shown.
        show: Whether to display the figure interactively.
        title: Figure title.
    """
    trajectories = _to_numpy(trajectories)
    rewards = _to_numpy(rewards).reshape(-1)
    image = _prepare_image_for_plot(image)

    if trajectories.ndim != 3 or trajectories.shape[-1] != 2:
        raise ValueError(f"Expected trajectories with shape (N, T, 2), got {trajectories.shape}")

    best_idx = int(np.argmax(rewards))
    traj_list = trajectories[:, :, ::-1].copy()
    traj_list[:, :, 0] = -traj_list[:, :, 0]

    fig = plt.figure(num="trajectory visualization", figsize=(12, 5), clear=True)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 2])
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_image = fig.add_subplot(gs[0, 1])
    fig.suptitle(title)

    for idx, traj in enumerate(traj_list):
        color = "red" if idx == best_idx else "green"
        alpha = 0.9 if idx == best_idx else 0.25
        linewidth = 2.0 if idx == best_idx else 1.0
        label = f"{idx}: {rewards[idx]:.3f}"
        ax_traj.plot(traj[:, 0], traj[:, 1], color=color, alpha=alpha, linewidth=linewidth, label=label)
        ax_traj.scatter(traj[-1, 0], traj[-1, 1], color=color, alpha=alpha, s=16)

    ax_traj.scatter([0], [0], color="blue", s=24)
    ax_traj.set_title("trajectory rewards")
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.grid(True, alpha=0.25)
    ax_traj.legend(loc="best", fontsize=8)

    if K is not None and T_cam_from_base is not None:
        image = overlay_paths_on_image(image, trajectories, rewards, K, dist, T_cam_from_base)

    ax_image.imshow(image)
    ax_image.set_title(f"observation, best trajectory {best_idx}")
    ax_image.axis("off")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show(block=False)
        plt.pause(1)
    else:
        plt.close(fig)


def load_model_from_checkpoint(model, optimizer, load_checkpoint_path, device):
    if load_checkpoint_path != "":
        checkpoint = torch.load(load_checkpoint_path, map_location=device)

        print(f"\nTotal Layers in Checkpoint: {len(checkpoint['model_state_dict'])}")

        total_layers = len(model.state_dict().keys())
        missing_layers = [key for key in model.state_dict().keys() if key not in checkpoint['model_state_dict']]
        print(f"\n Missing Layers (Expected in Model, but NOT in Checkpoint): {len(missing_layers)}")
        missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        print("Missing Layers (not in checkpoint):", len(missing_layers), total_layers)
        # print(checkpoint['optimizer_state_dict'].keys())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from {load_checkpoint_path} at epoch {start_epoch}")
    else:
        start_epoch = 0
        print("No previous checkpoint found. Starting fresh.")

    return start_epoch


def main():
    config_file_path = "config/setting.yaml"
    # config_file_path = "config/config_point_based.yaml"
    load_checkpoint_path = "./weights/model_150_epoch_34.pth"
    # load_checkpoint_path = "./weights/model_151_epoch_22.pth"
    # load_checkpoint_path = "./weights/epoch_029.pt"

    with open(config_file_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    device = "cuda" if (torch.cuda.is_available() and config['device'] == "cuda") else "cpu"
    device = torch.device(device)
    # print(f"Using device: {device}")

    # checkpoint_dir = "/fs/nexus-scratch/jianyu34/Projects/HALO/checkpoints/"
    checkpoint_dir = config['checkpoint_dir']
    batch_size = 16 # config['batch_size']
    n_epochs = config['epochs']
    use_wandb = False
    verbose = config['verbose']
    # Get the current time
    now = datetime.datetime.now()

    # Format the time as a string
    timestamp = now.strftime("%y-%m-%d_%H-%M-%S")
    project_name = config['project_name']
    entity_name = config['entity']
    lr = config['learning_rate']
    exp_name = f"{project_name}_{timestamp}"
    checkpoint_dir = os.path.join(checkpoint_dir, exp_name)
    save_name = "run"

    if use_wandb:
        run = wandb.init(entity=entity_name, project=project_name, dir=checkpoint_dir,
                         config=config)
        config['wandb_run_name'] = run.name
        save_name = run.name

    print("model config:")
    print(json.dumps(config, indent=4))
    # Define Model, Loss, Optimizer
    model = RewardModel(d_model=config["d_model"],
                        n_heads=config["num_heads"],
                        dropout=config["dropout"],
                        fusion_blocks=config["fusion_blocks"],
                        num_blocks=config["num_blocks"],
                        verbose=config['verbose']).to(device)
    criterion = bradley_terry_loss
    optimizer = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-3)
    if use_wandb:
        wandb.watch(model, log_freq=config['gradient_log_freq'])

    train_dataset = ChopPreferenceDataset(preference_root=config['preference_root'],
                                          image_root=config['image_root'],
                                          calib_file=config['calibration_file'],
                                          img_extension=config['image_ext'],
                                          mode="train",
                                          verbose=False,
                                          plot_imgs=config['plot_imgs'],
                                          dataset_len_limit=None,
                                          )
    val_dataset = ChopPreferenceDataset(preference_root=config['preference_root'],
                                        image_root=config['image_root'],
                                        calib_file=config['calibration_file'],
                                        img_extension=config['image_ext'],
                                        mode="test",
                                        verbose=False,
                                        plot_imgs=config['plot_imgs'],
                                        dataset_len_limit=None,
                                        )


    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=3)

    os.makedirs(checkpoint_dir, exist_ok=True)

    if checkpoint_dir is not None:
        print(f"checkpoint_dir: {checkpoint_dir}")
        plot_dir = os.path.join(checkpoint_dir, "trajectory_plots")
        print(f"plot_dir: {plot_dir}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        with open(os.path.join(checkpoint_dir, "config.yaml"), "w") as f:
            yaml.dump(run_config, f)
            model.eval()

    global_step = 0
    start_time = time.time()
    # Eval Loop
    dataloader = train_loader
    # dataloader = val_loader
    for epoch in range(0, 1):  # Start from checkpointed epoch
        train_loss = 0.0
        batch_count = 0

        for batch in tqdm(dataloader, desc="eval loop..."):
            image = batch["image"].to(device) # [Batch, 720, 1280, 3]
            points = batch["points"].to(device) # [Batch, n_points, 10, 3]
            points = points[:, :, :, :2] # [Batch, n_points, 10, 2], only get x and y coords
            B, n_points, k, d = points.shape

            sample_idx = batch_count * batch_size
            robot_name = infer_robot_name(dataloader.dataset, sample_idx)

            display_image = image[0]
            image = display_image.unsqueeze(0).repeat(B, 1, 1, 1)
            flat_points = points.reshape(-1, k, 2)
            flat_points[:len(dummy_points)] = dummy_points[:, :, :2]
            points = points.reshape(B, n_points, k, 2)
            # optimizer.zero_grad()
            # Forward pass
            image = model.processor(image, return_tensors="pt") # pixel_values: # [B, 3, 224, 224]
            with torch.inference_mode():
                reward_prediction = model(points, image) # [batch * n_points]

            # shape reward back into pairwise setting
            reshaped_rwd = reward_prediction.reshape((B, n_points))
            display_points = points.reshape(-1, k, 2)
            rand_indices = np.arange(len(dummy_points))
            display_reward = reshaped_rwd.reshape(-1)[rand_indices]
            print(f"best reward idx {torch.argmax(display_reward).item()} out of reward {display_reward}")
            plot_trajectory(
                display_image,
                display_points[rand_indices],
                display_reward,
                K=dataloader.dataset.K,
                dist=dataloader.dataset.dist,
                T_cam_from_base=dataloader.dataset.T_cam_from_base[robot_name],
                save_path=os.path.join(plot_dir, f"trajectory_{global_step:06d}.png"),
                show=True,
                title=f"trajectory visualization | step {global_step}",
            )
            batch_count += 1
            global_step += 1

            if batch_count % config['batch_print_freq'] == 0:
                SPS = global_step / (time.time() - start_time)
                if use_wandb:
                    run.log({"charts/SPS": SPS, "epoch": epoch}, global_step)
        avg_loss = train_loss / len(train_loader)
        if use_wandb:
            run.log({"charts/avg_loss": avg_loss, "epoch": epoch}, global_step)

        # Print Epoch Results
        print(f"! End of epoch ({epoch + 1}/{n_epochs}) | Avg Loss: {avg_loss:.4f}")
        print({"charts/avg_loss": avg_loss, "charts/learning_rate": optimizer.param_groups[0]['lr']})
        # scheduler.step()  # Adjust learning rate
        # scheduler.step(avg_val_loss)  # Adjust learning rate

    print("Eval Complete!")
    if use_wandb:
        run.finish()


if __name__ == "__main__":
    with open('config/setting.yaml', 'r') as f:
        run_config = yaml.load(f, Loader=yaml.SafeLoader)
        main()
