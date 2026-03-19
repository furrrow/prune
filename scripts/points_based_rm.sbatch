#!/bin/bash
#SBATCH --job-name=prune_rm_points
#SBATCH --output=logs/points_rm/%x_%j.out
#SBATCH --error=logs/points_rm/%x_%j.err
#SBATCH --gres=gpu:rtxa5000:8
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --qos=huge-long
#SBATCH --mem=256G
#SBATCH --time=1-23:00:00
#SBATCH --ntasks=8

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/points_rm

source ~/.bashrc
conda activate prune

# Keep CPU thread contention under control when using many data workers.
export OMP_NUM_THREADS=1

torchrun \
  --standalone \
  --nproc_per_node="${SLURM_NTASKS}" \
  src/train_rm_points_based.py \
  --config config/config_point_based.yaml
