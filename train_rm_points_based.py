from __future__ import annotations

import argparse
import os
import time
import datetime
from pathlib import Path
from typing import Dict, Tuple, Union

import yaml
import wandb

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from chop_dataloader_points_based import CHOPDatasetFull, collate_points_based
from models.reward_model_point_based import RewardModelPointBased


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
    scores: torch.Tensor,
    ranking: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    pair_target: torch.Tensor,
    eps: float = 0.05,     # label smoothing
    tau: float = 1.0,      # temperature (>=1 softens)
    lam: float = 0.0,      # margin penalty strength
    use_rank_gap_weight: bool = False,
    rank_gap_power: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # scores: (B, M) in the same column order as `ranking`.
    # pair_i/pair_j are path IDs; map ID -> column position first.
    bsz, num_paths = scores.shape
    pos_ids = torch.arange(num_paths, device=scores.device, dtype=ranking.dtype).unsqueeze(0).expand(bsz, -1)
    id_to_pos = torch.empty_like(ranking)
    id_to_pos.scatter_(1, ranking, pos_ids)  # id_to_pos[b, path_id] = column position in scores

    pos_i = id_to_pos.gather(1, pair_i)
    pos_j = id_to_pos.gather(1, pair_j)
    logits = scores.gather(1, pos_i) - scores.gather(1, pos_j)  # (B, P)

    # label smoothing
    y = pair_target * (1 - 2 * eps) + eps

    # temperature
    logits_t = logits / tau

    # BCE loss (optionally weighted by rank-distance, e.g., best-vs-worst higher weight)
    bce_raw = F.binary_cross_entropy_with_logits(logits_t, y, reduction="none")
    if use_rank_gap_weight:
        rank_gap = (pos_i - pos_j).abs().to(bce_raw.dtype).clamp_min(1.0)
        weights = rank_gap.pow(float(rank_gap_power))
        # Normalize to keep comparable loss magnitude across settings.
        weights = weights / (weights.mean().detach() + 1e-8)
        bce = (bce_raw * weights).mean()
    else:
        bce = bce_raw.mean()

    # margin/logit penalty (apply to *raw* logits, not temperature-scaled)
    pen = (logits ** 2).mean()
    loss = bce + lam * pen

    # accuracy should be computed on raw logits (threshold at 0)
    preds = (logits >= 0).to(pair_target.dtype)
    acc = (preds == pair_target).float().mean()

    return loss, acc

def compute_topk_metrics(
    scores: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Assumption: For each sample, the M input trajectories are already ordered
    best->worst in the input tensor. Therefore:
      - position 0 is the GT-best trajectory
      - position 1 is GT-second best
      - ...
    scores: (B, M) predicted rewards aligned with that GT order.

    Returns:
      topk_set_acc: 1 if model's top-k positions are exactly {0..k-1} (order ignored)
      topk_order_acc: 1 if model's top-k positions are exactly [0,1,...,k-1] (order exact)
    """
    B, M = scores.shape
    k_eff = max(1, min(int(k), M))

    # model's top-k positions (highest score first)
    topk_pos = scores.topk(k=k_eff, dim=1, largest=True, sorted=True).indices  # (B,k)

    # expected GT top-k positions
    gt_pos = torch.arange(k_eff, device=scores.device).unsqueeze(0).expand(B, -1)  # (B,k)

    # order accuracy: exact match [0,1,...,k-1]
    topk_order_acc = (topk_pos == gt_pos).all(dim=1).float().mean()

    # set accuracy: same set {0..k-1} ignoring order
    topk_set_acc = (torch.sort(topk_pos, dim=1).values == gt_pos).all(dim=1).float().mean()

    return topk_set_acc, topk_order_acc

@torch.no_grad()
def run_eval(
    model: Union[DDP, RewardModelPointBased],
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    topk: int,
    use_rank_gap_weight: bool,
    rank_gap_power: float,
) -> Tuple[float, float, float, float]:
    model.eval()
    total_loss = torch.zeros(1, device=device)
    total_acc = torch.zeros(1, device=device)
    total_topk_set_acc = torch.zeros(1, device=device)
    total_topk_order_acc = torch.zeros(1, device=device)
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
                scores,
                dev_batch["ranking"],
                dev_batch["pair_i"],
                dev_batch["pair_j"],
                dev_batch["pair_target"],
                use_rank_gap_weight=use_rank_gap_weight,
                rank_gap_power=rank_gap_power,
            )
            topk_set_acc, topk_order_acc = compute_topk_metrics(scores, k=topk)

        total_loss += loss.detach()
        total_acc += acc.detach()
        total_topk_set_acc += topk_set_acc.detach()
        total_topk_order_acc += topk_order_acc.detach()
        total_count += 1

    maybe_all_reduce(total_loss)
    maybe_all_reduce(total_acc)
    maybe_all_reduce(total_topk_set_acc)
    maybe_all_reduce(total_topk_order_acc)
    maybe_all_reduce(total_count)
    mean_loss = (total_loss / total_count).item()
    mean_acc = (total_acc / total_count).item()
    mean_topk_set_acc = (total_topk_set_acc / total_count).item()
    mean_topk_order_acc = (total_topk_order_acc / total_count).item()
    return mean_loss, mean_acc, mean_topk_set_acc, mean_topk_order_acc


def main() -> None:
    config_path = "config/setting.yaml"

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    rank, world_size, local_rank, device, distributed = setup_distributed()
    set_seed(config["seed"] + rank)

    checkpoint_dir = config["checkpoint_dir"]
    resume_checkpoint = Path(config["resume_checkpoint"]) if config.get("resume_checkpoint") else None
    train_json = Path(config["train_json"])
    test_json = Path(config["test_json"])
    image_root = Path(config["image_root"])

    use_lmdb = config["use_lmdb"]
    lmdb_root = Path(config["lmdb_root"])
    train_lmdb_path = lmdb_root / f"{train_json.stem}.lmdb"
    test_lmdb_path = lmdb_root / f"{test_json.stem}.lmdb"
    lmdb_build_if_missing = bool(config.get("lmdb_build_if_missing", True))
    lmdb_overwrite = bool(config.get("lmdb_overwrite", False))

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
    exp_name = f"{project_name}_{timestamp}"
    checkpoint_dir = os.path.join(checkpoint_dir, exp_name)
    save_name = "run"

    if rank == 0:
        if use_wandb:
            run = wandb.init(entity=entity_name, project=project_name, dir=checkpoint_dir,
                             config=config)
            config['wandb_run_name'] = run.name
            save_name = run.name

    # if use_lmdb and lmdb_build_if_missing:
    #     if rank == 0:
    #         CHOPDatasetFull.build_lmdb_cache(
    #             annotations_path=train_json,
    #             images_root=image_root,
    #             lmdb_path=train_lmdb_path,
    #             image_size=(config["image_size"][0], config["image_size"][1]),
    #             use_xy_only=True,
    #             overwrite=lmdb_overwrite,
    #         )
    #         CHOPDatasetFull.build_lmdb_cache(
    #             annotations_path=test_json,
    #             images_root=image_root,
    #             lmdb_path=test_lmdb_path,
    #             image_size=(config["image_size"][0], config["image_size"][1]),
    #             use_xy_only=True,
    #             overwrite=lmdb_overwrite,
    #         )
    #     if distributed:
    #         dist.barrier()

    train_ds = CHOPDatasetFull(
        annotations_path=train_json,
        images_root=image_root,
        image_size=(config["image_size"][0], config["image_size"][1]),
        use_xy_only=True,
        use_lmdb=use_lmdb,
        lmdb_path=train_lmdb_path if use_lmdb else None,
    )
    val_ds = CHOPDatasetFull(
        annotations_path=test_json,
        images_root=image_root,
        image_size=(config["image_size"][0], config["image_size"][1]),
        use_xy_only=True,
        use_lmdb=use_lmdb,
        lmdb_path=test_lmdb_path if use_lmdb else None,
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
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(enabled=(config["use_amp"] and device.type == "cuda"))
    warmup_epochs = int(config.get("warmup_epochs", 0))
    cosine_t_max = int(max(1, int(config["epochs"]) - warmup_epochs))
    eta_min = float(config.get("min_lr", 5e-6))
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_t_max, eta_min=eta_min
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_t_max, eta_min=eta_min
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
    topk_metric = int(config.get("top_k_metric", 2))
    use_rank_gap_weight = bool(config.get("use_rank_gap_weight", False))
    rank_gap_power = float(config.get("rank_gap_power", 1.0))
    global_step = 0
    batch_print_freq = int(config.get("batch_print_freq", 0))

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        total_train_batches = len(train_loader)

        running_loss = torch.zeros(1, device=device)
        running_acc = torch.zeros(1, device=device)
        running_topk_set_acc = torch.zeros(1, device=device)
        running_topk_order_acc = torch.zeros(1, device=device)
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
                    scores,
                    dev_batch["ranking"],
                    dev_batch["pair_i"],
                    dev_batch["pair_j"],
                    dev_batch["pair_target"],
                    use_rank_gap_weight=use_rank_gap_weight,
                    rank_gap_power=rank_gap_power,
                )
                topk_set_acc, topk_order_acc = compute_topk_metrics(scores, k=topk_metric)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.detach()
            running_acc += acc.detach()
            running_topk_set_acc += topk_set_acc.detach()
            running_topk_order_acc += topk_order_acc.detach()
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
        maybe_all_reduce(running_topk_set_acc)
        maybe_all_reduce(running_topk_order_acc)
        maybe_all_reduce(running_count)
        train_loss = (running_loss / running_count).item()
        train_acc = (running_acc / running_count).item()
        train_topk_set_acc = (running_topk_set_acc / running_count).item()
        train_topk_order_acc = (running_topk_order_acc / running_count).item()
        train_steps = max(1, int(running_count.item()))
        avg_data_time = epoch_data_time / train_steps
        avg_compute_time = epoch_compute_time / train_steps

        val_loss, val_acc, val_topk_set_acc, val_topk_order_acc = run_eval(
            model,
            val_loader,
            device,
            config["use_amp"],
            topk=topk_metric,
            use_rank_gap_weight=use_rank_gap_weight,
            rank_gap_power=rank_gap_power,
        )

        if rank == 0:
            print(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_loss:.5f} train_pair_acc={train_acc:.4f} "
                f"train_top{topk_metric}_set_acc={train_topk_set_acc:.4f} train_top{topk_metric}_order_acc={train_topk_order_acc:.4f} "
                f"val_loss={val_loss:.5f} val_pair_acc={val_acc:.4f} "
                f"val_top{topk_metric}_set_acc={val_topk_set_acc:.4f} val_top{topk_metric}_order_acc={val_topk_order_acc:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6e} "
                f"data_time={avg_data_time:.4f}s compute_time={avg_compute_time:.4f}s"
            )
            if run is not None:
                run.log(
                    {
                        "epoch": epoch,
                        "charts/train_loss": train_loss,
                        "charts/train_pair_acc": train_acc,
                        f"charts/train_top{topk_metric}_set_acc": train_topk_set_acc,
                        f"charts/train_top{topk_metric}_order_acc": train_topk_order_acc,
                        "charts/val_loss": val_loss,
                        "charts/val_pair_acc": val_acc,
                        f"charts/val_top{topk_metric}_set_acc": val_topk_set_acc,
                        f"charts/val_top{topk_metric}_order_acc": val_topk_order_acc,
                        "charts/lr": optimizer.param_groups[0]["lr"],
                        "charts/scheduler_lr": scheduler.get_last_lr()[0],
                        "timing/avg_data_time_s": avg_data_time,
                        "timing/avg_compute_time_s": avg_compute_time,
                    }
                )

            if (epoch + 1) % config["checkpoint_freq"] == 0:
                run_ckpt_dir = checkpoint_dir / save_name
                run_ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = run_ckpt_dir / f"epoch_{epoch:03d}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": unwrap_model(model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        # "args": vars(args),
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
