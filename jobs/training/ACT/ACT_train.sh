#!/bin/bash
#SBATCH --job-name=act_train
#SBATCH -A hjx@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.out

set -x
set -e

export XDG_RUNTIME_DIR=$SCRATCH/tmp/runtime-$SLURM_JOBID; mkdir -p $XDG_RUNTIME_DIR; chmod 700 $XDG_RUNTIME_DIR

module purge
module load miniforge

pwd; hostname; date

cd "$WORK/Projects/lerobot"

source "$HOME/.bashrc"
conda activate lerobot
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Set HuggingFace cache directory to $SCRATCH to avoid filling home directory
export HF_HOME="$SCRATCH/data/lerobot/.cache/huggingface"
mkdir -p "$HF_HOME"

# Set LeRobot data directory (using new environment variable name)
export HF_LEROBOT_HOME="$SCRATCH/data/lerobot"
export WANDB_MODE=offline

TRAIN_DATASET="${TRAIN_DATASET:-put_banana_in_plate}"

lerobot-train \
  --dataset.repo_id=local \
  --dataset.root="$SCRATCH/data/lerobot/$TRAIN_DATASET" \
  --policy.type=act \
  --output_dir="$SCRATCH/data/lerobot/outputs/train/act_$TRAIN_DATASET" \
  --job_name="act_$TRAIN_DATASET" \
  --policy.device=cuda \
  --policy.use_amp=true \
  --batch_size=16 \
  --num_workers=12 \
  --wandb.enable=true \
  --wandb.mode=offline \
  --wandb.project=lerobot_hpc \
  --policy.push_to_hub=false