from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Tuple, Union

import yaml
import wandb

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from datasets.dataloader_points_based import CHOPDatasetFull, collate_points_based
from models.reward_model_point_based import RewardModelPointBased


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("DDP training for point-based reward model")
    parser.add_argument("--config", type=Path, help="Path to config.yaml", default=Path("../config/config_point_based.yaml"))
    return parser.parse_args()


def setup_distributed() -> Tuple[int, int, int, torch.device, bool]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        distributed = world_size > 1
    else:
        rank, world_size, local_rank = 0, 1, 0
        distributed = False

    use_cuda = torch.cuda.is_available()
    if distributed:
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device, distributed


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
    return out


def unwrap_model(model: Union[DDP, RewardModelPointBased]) -> RewardModelPointBased:
    return model.module if isinstance(model, DDP) else model


def build_image_inputs(
    model: Union[DDP, RewardModelPointBased], images: torch.Tensor, device: torch.device
) -> Dict[str, torch.Tensor]:
    processor = unwrap_model(model).processor
    image_inputs = processor(images=images, return_tensors="pt", do_rescale=False)
    return {k: v.to(device, non_blocking=True) for k, v in image_inputs.items()}


def maybe_all_reduce(t: torch.Tensor) -> None:
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)


def compute_pairwise_metrics(
    scores: torch.Tensor, pair_i: torch.Tensor, pair_j: torch.Tensor, pair_target: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # scores: (B, M), pair_*: (B, P)
    logits = scores.gather(1, pair_i) - scores.gather(1, pair_j)  # (B, P)
    loss = F.binary_cross_entropy_with_logits(logits, pair_target)
    preds = (logits >= 0).to(pair_target.dtype)
    acc = (preds == pair_target).float().mean()
    return loss, acc


@torch.no_grad()
def run_eval(
    model: Union[DDP, RewardModelPointBased],
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float]:
    model.eval()
    total_loss = torch.zeros(1, device=device)
    total_acc = torch.zeros(1, device=device)
    total_count = torch.zeros(1, device=device)

    amp_enabled = use_amp and device.type == "cuda"
    for batch in loader:
        dev_batch = to_device(batch, device)
        image_inputs = build_image_inputs(model, dev_batch["image"], device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            flat_scores = model(dev_batch["points"], image_inputs)  # (B*M,)
            bsz, num_paths = dev_batch["points"].shape[:2]
            scores = flat_scores.view(bsz, num_paths)
            loss, acc = compute_pairwise_metrics(
                scores, dev_batch["pair_i"], dev_batch["pair_j"], dev_batch["pair_target"]
            )

        total_loss += loss.detach()
        total_acc += acc.detach()
        total_count += 1

    maybe_all_reduce(total_loss)
    maybe_all_reduce(total_acc)
    maybe_all_reduce(total_count)
    mean_loss = (total_loss / total_count).item()
    mean_acc = (total_acc / total_count).item()
    return mean_loss, mean_acc


def main() -> None:

    args = parse_args()

    config_path = args.config.resolve()
    project_root = config_path.parent.parent

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    rank, world_size, local_rank, device, distributed = setup_distributed()
    set_seed(config["seed"] + rank)

    checkpoint_dir = Path(config["checkpoint_dir"])
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = (project_root / checkpoint_dir).resolve()
    resume_checkpoint = Path(config["resume_checkpoint"]) if config.get("resume_checkpoint") else None
    train_json = Path(config["train_json"])
    test_json = Path(config["test_json"])
    image_root = Path(config["image_root"])
    if not train_json.is_absolute():
        train_json = (project_root / train_json).resolve()
    if not test_json.is_absolute():
        test_json = (project_root / test_json).resolve()
    if not image_root.is_absolute():
        image_root = (project_root / image_root).resolve()
    if resume_checkpoint is not None and not resume_checkpoint.is_absolute():
        resume_checkpoint = (project_root / resume_checkpoint).resolve()

    wandb_dir = Path(config.get("wandb_dir", project_root / "wandb"))
    if not wandb_dir.is_absolute():
        wandb_dir = (project_root / wandb_dir).resolve()

    run = None
    run_name = "no_wandb"
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        wandb_dir.mkdir(parents=True, exist_ok=True)
        if config.get("use_wandb", False):
            run = wandb.init(
                project=config.get("project_name", "prune"),
                entity=config.get("entity"),
                config=config,
                dir=str(wandb_dir),
            )
            run_name = run.name or run.id or "wandb_run"

    train_ds = CHOPDatasetFull(
        annotations_path=train_json,
        images_root=image_root,
        image_size=(config["image_size"][0], config["image_size"][1]),
        use_xy_only=True,
    )
    val_ds = CHOPDatasetFull(
        annotations_path=test_json,
        images_root=image_root,
        image_size=(config["image_size"][0], config["image_size"][1]),
        use_xy_only=True,
    )

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        if distributed
        else None
    )
    val_sampler = (
        DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        if distributed
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config["num_workers"],
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_points_based,
        persistent_workers=config["num_workers"] > 0,
        prefetch_factor=int(config.get("prefetch_factor", 2)) if config["num_workers"] > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        sampler=val_sampler,
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_points_based,
        persistent_workers=config["num_workers"] > 0,
        prefetch_factor=int(config.get("prefetch_factor", 2)) if config["num_workers"] > 0 else None,
    )

    base_model = RewardModelPointBased(
        d_model=config["d_model"],
        n_heads=config["num_heads"],
        dropout=config["dropout"],
        verbose=(rank == 0),
        fusion_blocks=config["fusion_blocks"],
        num_blocks=config["num_blocks"],
        traj_per_image=4,
    ).to(device)
    model: Union[DDP, RewardModelPointBased]
    if distributed:
        model = DDP(
            base_model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )
    else:
        model = base_model

    lr = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(enabled=(config["use_amp"] and device.type == "cuda"))
    warmup_epochs = int(config.get("warmup_epochs", 0))
    cosine_t0 = int(config.get("cosine_LR_T", 10))
    cosine_tmult = int(config.get("cosine_LR_mult", 2))
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cosine_t0, T_mult=cosine_tmult, eta_min=5e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cosine_t0, T_mult=cosine_tmult, eta_min=5e-6
        )

    if rank == 0 and run is not None:
        wandb.watch(
            unwrap_model(model),
            log="gradients",
            log_freq=int(config.get("gradient_log_freq", 100)),
        )

    start_epoch = 0
    if config["resume"]:
        map_location = {"cuda:%d" % 0: "cuda:%d" % local_rank} if device.type == "cuda" else device
        ckpt = torch.load(resume_checkpoint, map_location=map_location)
        unwrap_model(model).load_state_dict(ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        if rank == 0:
            print(f"Resumed from {resume_checkpoint} at epoch {start_epoch}")

    amp_enabled = config["use_amp"] and device.type == "cuda"
    global_step = 0
    batch_print_freq = int(config.get("batch_print_freq", 0))

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        total_train_batches = len(train_loader)

        running_loss = torch.zeros(1, device=device)
        running_acc = torch.zeros(1, device=device)
        running_count = torch.zeros(1, device=device)
        epoch_data_time = 0.0
        epoch_compute_time = 0.0
        step_start = time.perf_counter()

        for batch_idx, batch in enumerate(train_loader, start=1):
            data_end = time.perf_counter()
            batch_data_time = data_end - step_start
            epoch_data_time += batch_data_time

            compute_start = time.perf_counter()
            dev_batch = to_device(batch, device)
            image_inputs = build_image_inputs(model, dev_batch["image"], device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                flat_scores = model(dev_batch["points"], image_inputs)  # (B*M,)
                bsz, num_paths = dev_batch["points"].shape[:2]
                scores = flat_scores.view(bsz, num_paths)  # (B, M)
                loss, acc = compute_pairwise_metrics(
                    scores, dev_batch["pair_i"], dev_batch["pair_j"], dev_batch["pair_target"]
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.detach()
            running_acc += acc.detach()
            running_count += 1
            batch_compute_time = time.perf_counter() - compute_start
            epoch_compute_time += batch_compute_time
            global_step += 1

            if rank == 0 and batch_print_freq > 0 and batch_idx % batch_print_freq == 0:
                print(
                    f"[Epoch {epoch:03d} | Batch {batch_idx:05d}/{total_train_batches:05d}] "
                    f"loss={loss.item():.5f} acc={acc.item():.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.6e} "
                    f"data_time={batch_data_time:.4f}s compute_time={batch_compute_time:.4f}s"
                )
                if run is not None:
                    run.log(
                        {
                            "step": global_step,
                            "charts/step_train_loss": float(loss.item()),
                            "charts/step_train_pair_acc": float(acc.item()),
                            "charts/lr_step": optimizer.param_groups[0]["lr"],
                            "timing/step_data_time_s": float(batch_data_time),
                            "timing/step_compute_time_s": float(batch_compute_time),
                            "charts/step_progress": float(batch_idx / max(total_train_batches, 1)),
                        }
                    )
            step_start = time.perf_counter()

        maybe_all_reduce(running_loss)
        maybe_all_reduce(running_acc)
        maybe_all_reduce(running_count)
        train_loss = (running_loss / running_count).item()
        train_acc = (running_acc / running_count).item()
        train_steps = max(1, int(running_count.item()))
        avg_data_time = epoch_data_time / train_steps
        avg_compute_time = epoch_compute_time / train_steps

        val_loss, val_acc = run_eval(model, val_loader, device, config["use_amp"])

        if rank == 0:
            print(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_loss:.5f} train_pair_acc={train_acc:.4f} "
                f"val_loss={val_loss:.5f} val_pair_acc={val_acc:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6e} "
                f"data_time={avg_data_time:.4f}s compute_time={avg_compute_time:.4f}s"
            )
            if run is not None:
                run.log(
                    {
                        "epoch": epoch,
                        "charts/train_loss": train_loss,
                        "charts/train_pair_acc": train_acc,
                        "charts/val_loss": val_loss,
                        "charts/val_pair_acc": val_acc,
                        "charts/lr": optimizer.param_groups[0]["lr"],
                        "charts/scheduler_lr": scheduler.get_last_lr()[0],
                        "timing/avg_data_time_s": avg_data_time,
                        "timing/avg_compute_time_s": avg_compute_time,
                    }
                )

            if (epoch + 1) % config["checkpoint_freq"] == 0:
                run_ckpt_dir = checkpoint_dir / run_name
                run_ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = run_ckpt_dir / f"epoch_{epoch:03d}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": unwrap_model(model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint: {ckpt_path}")
        scheduler.step()

    if rank == 0 and run is not None:
        run.finish()
    cleanup_distributed()

if __name__ == "__main__":
    main()
