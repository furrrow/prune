"""
train.py

"""
import torch
import torch.optim as optim
import os
import time
import datetime
import wandb
import json
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import yaml
import os

from src.chop_dataloader import ChopImageDataset
from src.reward_model import ImageRewardModel
from src.loss_fn import bradley_terry_loss


def scheduled_probability(epoch, n_epochs, start_prob=0.0, end_prob=1.0):
    if n_epochs <= 1:
        return float(end_prob)
    progress = epoch / (n_epochs - 1)
    progress = max(0.0, min(1.0, progress))
    probability = start_prob + progress * (end_prob - start_prob)
    return float(max(0.0, min(1.0, probability)))

def main():
    with open('config/setting.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # checkpoint_dir = "/fs/nexus-scratch/jianyu34/Projects/HALO/checkpoints/"
    checkpoint_dir = config['checkpoint_dir']
    load_checkpoint_path = config['load_checkpoint_path']
    device = "cuda" if (torch.cuda.is_available() and config['device'] == "cuda") else "cpu"
    device = torch.device(device)
    # print(f"Using device: {device}")

    batch_size = config['batch_size']
    n_epochs = config['epochs']
    use_wandb = config['use_wandb']
    verbose = config['verbose']
    lambda_reward = float(config.get("reward_l2", 1e-3))
    # Get the current time
    now = datetime.datetime.now()

    # Format the time as a string
    timestamp = now.strftime("%y-%m-%d_%H-%M-%S")
    project_name = config['project_name']
    entity_name = config['entity']
    lr = config['learning_rate']
    exp_name = f"{project_name}_img_{timestamp}"
    checkpoint_dir = os.path.join(checkpoint_dir, exp_name)
    save_name = "run"
    if config['sweep']:
        use_wandb = True

    if use_wandb:
        run = wandb.init(entity=entity_name, project=project_name, dir=checkpoint_dir,
                         config=config)
        config['wandb_run_name'] = run.name
        save_name = run.name
        # update hyperparams from the wandb sweep if there is one:
        if config['sweep']:
            for key, value in dict(run.config).items():
                if key == "lr":
                    config["learning_rate"] = float(value)
                elif key in config:
                    config[key] = value
            lr = config["learning_rate"]
            batch_size = config["batch_size"]
            n_epochs = config["epochs"]

    print("model config:")
    print(json.dumps(config, indent=4))
    # Define Model, Loss, Optimizer
    model = ImageRewardModel(hidden_dim=config['hidden_dim'],
                             n_heads=config["num_heads"],
                             dropout=config["dropout"],
                             verbose=config['verbose'],
                             quantize=True).to(device)
    criterion = bradley_terry_loss
    optimizer = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(config["weight_decay"]))
    if use_wandb:
        wandb.watch(model, log_freq=config['gradient_log_freq'])

    train_dataset = ChopImageDataset(preference_root=config['preference_root'],
                                     image_root=config['image_root'],
                                     image_overlay_root=config['image_overlay_root'],
                                      calib_file=config['calibration_file'],
                                      img_extension=config['image_ext'],
                                      mode="train",
                                      verbose=False,
                                      plot_imgs=config['plot_imgs'],
                                      dataset_len_limit=None,
                                      )
    val_dataset = ChopImageDataset(preference_root=config['preference_root'],
                                   image_root=config['image_root'],
                                   image_overlay_root=config['image_overlay_root'],
                                    calib_file=config['calibration_file'],
                                    img_extension=config['image_ext'],
                                    mode="test",
                                    verbose=False,
                                    plot_imgs=config['plot_imgs'],
                                    dataset_len_limit=None,
                                    )


    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=config['num_workers'],
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=3)

    use_scheduler = bool(config.get("use_scheduler", True))
    if config["sweep"] and config.get("disable_scheduler_during_sweep", True):
        use_scheduler = False

    if use_scheduler:
        warmup_epochs = config['warmup_epochs']
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                                       total_iters=warmup_epochs)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config['cosine_LR_T'],
                                                                          T_mult=config['cosine_LR_mult'], eta_min=5e-6)
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
    else:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    os.makedirs(checkpoint_dir, exist_ok=True)
    arch_path = f"{checkpoint_dir}/{save_name}_model_architecture.txt"

    with open(arch_path, "w") as f:
        f.write(str(model))

    if checkpoint_dir is not None:
        print(f"checkpoint_dir: {checkpoint_dir}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(checkpoint_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f)

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
            else:
                start_epoch = 0
                print("No previous checkpoint found. Starting fresh.")

            global_step = 0

            start_time = time.time()


    # Training Loop
    for epoch in range(start_epoch, n_epochs):  # Start from checkpointed epoch
        model.train()
        train_loss = 0.0
        batch_count = 0
        hard_pair_tally = 0
        for batch in tqdm(train_loader, desc="training loop..."):
            images = batch["image"].to(device)
            B, n_img, h, w, c = images.shape
            images = images.reshape(-1, h, w, c)
            optimizer.zero_grad()

            # Forward pass
            with torch.no_grad():
                processed_imgs = model.processor(images, return_tensors="pt")
                img_features = model.image_feature_extractor(**processed_imgs).last_hidden_state
            _, c, d = img_features.shape
            img_features = img_features.reshape(B, n_img, c, d)
            original = img_features[:, 0]
            rank0_img = img_features[:, 1]
            rank1_img = img_features[:, 2]
            rank2_img = img_features[:, 3]
            rank3_img = img_features[:, 4]

            # as we get later in the epochs we should use rank2 vs rank3 more often as this comparison should be harder
            hard_pair_prob = scheduled_probability(epoch, n_epochs)
            use_hard_pair = torch.rand(1).item() < hard_pair_prob
            if use_hard_pair:
                preferred_reward = model(original, rank2_img, input_type="features")
                rejected_reward = model(original, rank3_img, input_type="features")
                hard_pair_tally += 1
            else:
                rank0 = model(original, rank0_img, input_type="features")
                rank1 = model(original, rank1_img, input_type="features")
                rank2 = model(original, rank2_img, input_type="features")
                rank3 = model(original, rank3_img, input_type="features")

                # shape reward back into pairwise setting
                coin = torch.randint(0, 2, (1,)).item()
                if coin == 0:
                    preferred_reward = torch.concat((rank0, rank1))
                    rejected_reward = torch.concat((rank2, rank3))
                else:
                    preferred_reward = torch.concat((rank0, rank2))
                    rejected_reward = torch.concat((rank1, rank3))

            # Compute Loss
            # loss = criterion(preferred_reward, rejected_reward)
            bt_loss = criterion(preferred_reward, rejected_reward)
            reward_l2 = torch.mean(preferred_reward ** 2) + torch.mean(rejected_reward ** 2)
            loss = bt_loss + lambda_reward * reward_l2
            # Backpropagation
            loss.backward()
            optimizer.step()
            if verbose:
                print(f"global_step {global_step} batch_count {batch_count} charts/train_loss {loss.item():.4f}")
            train_loss += loss.item()
            if use_wandb:
                run.log({"charts/train_loss": loss.item(), "charts/learning_rate": optimizer.param_groups[0]['lr'],
                         "charts/scheduler_lr": scheduler.get_last_lr()[0]}
                    , global_step)
            batch_count += 1
            global_step += 1
            if batch_count % config['batch_print_freq'] == 0:
                SPS = global_step / (time.time() - start_time)
                if use_wandb:
                    run.log({"charts/SPS": SPS, "epoch": epoch}, global_step)
        avg_train_loss = train_loss / len(train_loader)
        if use_wandb:
            run.log({"charts/avg_train_loss": avg_train_loss, "epoch": epoch, "hard_pair_prob": hard_pair_tally/batch_count},
                    global_step)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="validation loop..."):
                images = batch["image"].to(device)
                B, n_img, h, w, c = images.shape
                images = images.reshape(-1, h, w, c)
                with torch.no_grad():
                    processed_imgs = model.processor(images, return_tensors="pt")
                    img_features = model.image_feature_extractor(**processed_imgs).last_hidden_state

                preferred_reward = model(img_features[:, 0], img_features[:, 2], input_type="features")
                rejected_reward = model(img_features[:, 0], img_features[:, 4], input_type="features")
                bt_loss = criterion(preferred_reward, rejected_reward)
                reward_l2 = torch.mean(preferred_reward ** 2) + torch.mean(rejected_reward ** 2)
                loss = bt_loss + lambda_reward * reward_l2
                val_loss += loss.item()
                if verbose:
                    print(f"global_step {global_step} batch_count {batch_count} val_loss {loss.item():.4f}")

        avg_val_loss = val_loss / len(val_loader)

        if use_wandb:
            run.log(
            {"charts/avg_val_loss": avg_val_loss, "charts/learning_rate": optimizer.param_groups[0]['lr'], "charts/scheduler_lr": scheduler.get_last_lr()[0]}
            , global_step)
        # Print Epoch Results
        print(f"! End of epoch ({epoch + 1}/{n_epochs}) | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")
        print({"charts/avg_val_loss": avg_val_loss, "charts/learning_rate": optimizer.param_groups[0]['lr'], "charts/scheduler_lr": scheduler.get_last_lr()[0]})
        scheduler.step()  # Adjust learning rate
        # scheduler.step(avg_val_loss)  # Adjust learning rate

        if (epoch + 1) % config['checkpoint_freq'] == 0:
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
    with open('config/setting.yaml', 'r') as f:
        run_config = yaml.load(f, Loader=yaml.SafeLoader)

    # Define a sweep config dictionary
    sweep_configuration = {
        "method": "bayes",
        "name": "dual_rm_sweep",
        # Metric that you want to optimize
        # For example, if you want to maximize validation
        # accuracy set "goal": "maximize" and the name of the variable
        # you want to optimize for, in this case "val_acc"
        "metric": {
            "goal": "minimize",
            "name": "charts/avg_val_loss"
        },
        "parameters": {
            "lr": {"distribution": "log_uniform_values", "min": 1e-5, "max": 5e-3},
            "dropout": {"values": [0.05, 0.1, 0.15]},
        },
    }
    if run_config['sweep']:
        # Initialize the sweep by passing in the config dictionary
        sweep_id = wandb.sweep(sweep=sweep_configuration, entity=run_config['entity'],
                               project=run_config['project_name'])
        # Start the sweep job
        wandb.agent(sweep_id, function=main, count=10)
    else:
        main()
