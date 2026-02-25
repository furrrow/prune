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
from tqdm import tqdm
import yaml
import os

from src.chop_dataloader import ChopPreferenceDataset
from src.reward_model import PairwiseRewardModel
from src.loss_fn import bradley_terry_loss

def main():
    with open('../config/setting.yaml', 'r') as f:
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
    # Get the current time
    now = datetime.datetime.now()

    # Format the time as a string
    timestamp = now.strftime("%y-%m-%d_%H-%M-%S")
    project_name = config['project_name']
    entity_name = config['entity']
    lr = config['learning_rate']
    use_cls = config['use_cls']
    exp_name = f"{project_name}_{timestamp}"
    checkpoint_dir = os.path.join(checkpoint_dir, exp_name)
    if config['sweep']:
        use_wandb = True

    if use_wandb:
        run = wandb.init(entity=entity_name, project=project_name, dir=checkpoint_dir,
                         config=config)
        # update hyperparams from the wandb sweep if there is one:
        if config['sweep']:
            lr = run.config["lr"]
            use_cls = run.config['use_cls']


    # Define Model, Loss, Optimizer
    model = PairwiseRewardModel(hidden_dim=config['hidden_dim'], num_heads=config['num_heads'],
                                dropout=config['dropout'],
                                use_cls=use_cls, verbose=config['verbose']).to(device)
    criterion = bradley_terry_loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    if use_wandb:
        wandb.watch(model, log_freq=config['gradient_log_freq'])

    train_dataset = ChopPreferenceDataset(preference_root=config['preference_root'],
                                          image_root=config['image_root'],
                                          calib_file=config['calibration_file'],
                                          img_extension=config['image_ext'],
                                          mode="train",
                                          verbose=False,
                                          plot_imgs=config['plot_imgs'],
                                          )
    val_dataset = ChopPreferenceDataset(preference_root=config['preference_root'],
                                          image_root=config['image_root'],
                                          calib_file=config['calibration_file'],
                                          img_extension=config['image_ext'],
                                          mode="test",
                                          verbose=False,
                                          plot_imgs=config['plot_imgs'],
                                        )

    # train_sampler = WeightedRandomSampler(weights=train_dataset.sample_weights, num_samples=len(train_dataset),
    #                                       replacement=True)
    # val_sampler = WeightedRandomSampler(weights=val_dataset.sample_weights, num_samples=len(val_dataset),
    #                                     replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

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
                # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
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

        for batch in tqdm(train_loader, desc="training loop..."):
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
            if verbose:
                print(f"global_step {global_step} batch_count {batch_count} charts/train_loss {loss.item():.4f}")
            train_loss += loss.item()
            if use_wandb:
                run.log({"charts/train_loss": loss.item(), "charts/learning_rate": optimizer.param_groups[0]['lr'], "charts/scheduler_lr": scheduler.get_last_lr()[0]}
                    , global_step)
            batch_count += 1
            global_step += 1

            if batch_count % config['batch_print_freq'] == 0:
                SPS = global_step / (time.time() - start_time)
                print(
                    f"Epoch [{epoch + 1}/{n_epochs}] | Batch {batch_count} | Train Loss: {loss.item():.4f}, steps per second: {SPS:.3f} | LR: {optimizer.param_groups[0]['lr']}")
                if use_wandb:
                    run.log({"charts/SPS": SPS, "epoch": epoch}, global_step)
        avg_train_loss = train_loss / len(train_loader)
        if use_wandb:
            run.log({"charts/avg_train_loss": avg_train_loss, "epoch": epoch}, global_step)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="validation loop..."):
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
                if verbose:
                    print(f"global_step {global_step} batch_count {batch_count} val_loss {loss.item():.4f}")

        avg_val_loss = val_loss / len(val_loader)

        if use_wandb:
            run.log(
            {"charts/avg_val_loss": avg_val_loss, "charts/learning_rate": optimizer.param_groups[0]['lr'], "charts/scheduler_lr": scheduler.get_last_lr()[0]}
            , global_step)
        # Print Epoch Results
        print(f"! End of epoch ({epoch + 1}/{n_epochs}) | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")

        # scheduler.step()  # Adjust learning rate

        # scheduler.step(avg_val_loss)  # Adjust learning rate

        if (epoch + 1) % config['checkpoint_freq'] == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")

            # Save only trainable parameters (excluding frozen ones)
            trainable_state_dict = {k: v for k, v in model.state_dict().items() if "vision_model" not in k}

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': trainable_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                # 'val_loss': avg_val_loss
            }, checkpoint_path)

            print(f"Checkpoint saved: {checkpoint_path}")

    print("Training Complete!")
    if use_wandb:
        run.finish()

if __name__ == "__main__":
    with open('../config/setting.yaml', 'r') as f:
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
            "lr": {"max": 0.005, "min": 0.0001},
            "use_cls": {"values": [True, False]},
        },
    }
    if run_config['sweep']:
        # Initialize the sweep by passing in the config dictionary
        sweep_id = wandb.sweep(sweep=sweep_configuration, entity=run_config['entity'],
                               project=run_config['project_name'])
        # Start the sweep job
        wandb.agent(sweep_id, function=main, count=5)
    else:
        main()