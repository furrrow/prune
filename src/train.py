"""
train.py

"""
import torch
import torch.optim as optim
import os
import time
import datetime
import wandb
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler
import argparse
from pathlib import Path
import yaml
import os

from src.chop_dataloader import ChopPreferenceDataset
from src.reward_model import PairwiseRewardModel
from src.loss_fn import bradley_terry_loss

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
        default="train",
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
        default=False,
        help="plot dataloader graphs, set to false unless debug",
    )
    args = parser.parse_args()

    # checkpoint_dir = "/fs/nexus-scratch/jianyu34/Projects/HALO/checkpoints/"
    checkpoint_dir = "../models/checkpoints/"
    load_checkpoint_path = ""
    BATCH_SIZE = 8  # 288=48.6GB 256=43GB
    LEARNING_RATE = 2.5e-4
    HIDDEN_DIM = 768
    N_EPOCHS = 1
    train_val_split = 0.8
    num_workers = 1
    num_queries = 1
    num_heads = 4
    # num_attn_stacks = 2
    batch_print_freq = 5
    checkpoint_freq = 10
    gradient_log_freq = 50
    dropout = 0.1
    use_wandb = False
    save_model = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_config = {
        "device": str(device),
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": N_EPOCHS,
        "num_queries": num_queries,
        "hidden_dim": HIDDEN_DIM,
        "train_val_split": train_val_split,
        "num_workers": num_workers,
        "save_model": save_model,
        "batch_print_freq": batch_print_freq,
        "dropout": dropout,
        "warmup_epochs": 5,
        "cosine_LR_T": 10,
        "cosine_LR_mult": 2,
    }
    # Load Dataset and Split
    train_dataset = ChopPreferenceDataset(preference_root=args.preference_root,
                                          image_root=args.image_root,
                                          calib_file=args.calibration_file,
                                          img_extension=args.image_ext,
                                          split_json=args.test_train_split_json,
                                          mode="train",
                                          verbose=False,
                                          plot_imgs=args.plot_imgs,
                                          num_points=args.num_points,
                                          )
    val_dataset = ChopPreferenceDataset(preference_root=args.preference_root,
                                        image_root=args.image_root,
                                        calib_file=args.calibration_file,
                                        img_extension=args.image_ext,
                                        split_json=args.test_train_split_json,
                                        mode="test",
                                        verbose=False,
                                        plot_imgs=args.plot_imgs,
                                        num_points=args.num_points,
                                        )

    # train_sampler = WeightedRandomSampler(weights=train_dataset.sample_weights, num_samples=len(train_dataset),
    #                                       replacement=True)
    # val_sampler = WeightedRandomSampler(weights=val_dataset.sample_weights, num_samples=len(val_dataset),
    #                                     replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=True)

    # Get the current time
    now = datetime.datetime.now()

    # Format the time as a string
    timestamp = now.strftime("%y-%m-%d_%H-%M-%S")
    project_name = "Prune"
    exp_name = f"{project_name}_{timestamp}"
    run_name = f"{exp_name}_lr_{LEARNING_RATE}"
    checkpoint_dir = os.path.join(checkpoint_dir, run_name)
    # Define Model, Loss, Optimizer
    model = PairwiseRewardModel(num_heads=num_heads, dropout=dropout).to(device)
    criterion = bradley_terry_loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    # Define warmup scheduler
    warmup_epochs = run_config['warmup_epochs']
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                                   total_iters=warmup_epochs)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=run_config['cosine_LR_T'],
                                                                      T_mult=run_config['cosine_LR_mult'], eta_min=5e-6)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    arch_path = f"{checkpoint_dir}/reward_model_architecture.txt"

    with open(arch_path, "w") as f:
        f.write(str(model))

    if checkpoint_dir is not None:
        print(f"checkpoint_dir: {checkpoint_dir}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(checkpoint_dir, "config.yaml"), "w") as f:
            yaml.dump(run_config, f)

            # Load from latest checkpoint (if available)
            latest_checkpoint = None
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
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch']
                print(f"Loaded checkpoint from {load_checkpoint_path} at epoch {start_epoch}")
                wandb_id = checkpoint['wandb_id']
            else:
                start_epoch = 0
                print("No previous checkpoint found. Starting fresh.")
                wandb_id=None

            global_step = 0

            start_time = time.time()
    if use_wandb:
        run = wandb.init(project=project_name, config=run_config, dir=checkpoint_dir, name=run_name, fork_from=wandb_id)
        wandb.watch(model, log_freq=gradient_log_freq)

    # Training Loop
    for epoch in range(start_epoch, N_EPOCHS):  # Start from checkpointed epoch
        model.train()
        train_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            original = batch["image"].to(device)
            preferred = batch["preferred"].to(device)
            rejected = batch["rejected"].to(device)
            optimizer.zero_grad()

            # Forward pass
            original = model.processor(original, return_tensors="pt")
            preferred = model.processor(preferred, return_tensors="pt")
            rejected = model.processor(rejected, return_tensors="pt")
            preferred_reward = model(original, preferred)
            rejected_reward = model(original, rejected)

            # Compute Loss
            loss = criterion(preferred_reward, rejected_reward)
            # reward_reg_loss = lambda_reward * torch.mean(predicted_rewards ** 2)
            # loss = loss + reward_reg_loss
            # Backpropagation
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if use_wandb:
                run.log({"charts/train_loss": loss.item(), "charts/learning_rate": optimizer.param_groups[0]['lr'], "charts/scheduler_lr": scheduler.get_last_lr()[0]}
                    , global_step)
            batch_count += 1
            global_step += 1

            if batch_count % batch_print_freq == 0:
                SPS = global_step / (time.time() - start_time)
                print(
                    f"Epoch [{epoch + 1}/{N_EPOCHS}] | Batch {batch_count} | Train Loss: {loss.item():.4f}, steps per second: {SPS:.3f} | LR: {optimizer.param_groups[0]['lr']}")
                if use_wandb:
                    run.log({"charts/SPS": SPS, "epoch": epoch}, global_step)
        avg_train_loss = train_loss / len(train_loader)
        if use_wandb:
            run.log({"charts/avg_train_loss": avg_train_loss, "epoch": epoch}, global_step)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                original = batch["image"].to(device)
                preferred = batch["preferred"].to(device)
                rejected = batch["rejected"].to(device)
                original = model.processor(original, return_tensors="pt")
                preferred = model.processor(preferred, return_tensors="pt")
                rejected = model.processor(rejected, return_tensors="pt")
                preferred_reward = model(original, preferred)
                rejected_reward = model(original, rejected)
                loss = criterion(preferred_reward, rejected_reward)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        if use_wandb:
            run.log(
            {"charts/avg_val_loss": avg_val_loss, "charts/learning_rate": optimizer.param_groups[0]['lr'], "charts/scheduler_lr": scheduler.get_last_lr()[0]}
            , global_step)
        # Print Epoch Results
        print(f"Epoch [{epoch + 1}/{N_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        scheduler.step()  # Adjust learning rate

        # scheduler.step(avg_val_loss)  # Adjust learning rate

        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")

            # Save only trainable parameters (excluding frozen ones)
            trainable_state_dict = {k: v for k, v in model.state_dict().items() if "vision_model" not in k}

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': trainable_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, checkpoint_path)

            print(f"Checkpoint saved: {checkpoint_path}")

    print("Training Complete!")
    if use_wandb:
        run.finish()

if __name__ == "__main__":
    main()