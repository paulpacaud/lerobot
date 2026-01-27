#!/bin/bash
#SBATCH --job-name=pi0_train
#SBATCH -A hjx@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
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

# Enable offline mode for HuggingFace Hub to use cached models only
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Debug: Show cache directory and contents
echo "=========================================="
echo "HuggingFace Cache Configuration:"
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
echo "=========================================="
echo "Checking cached models:"
ls -la "$HF_HOME" 2>/dev/null || echo "Cache directory not found!"
echo ""
echo "Pi0 base model location:"
ls -d "$HF_HOME/models--lerobot--pi0_base" 2>/dev/null || echo "Pi0 model not found!"
echo ""
echo "PaliGemma tokenizer location:"
ls -d "$HF_HOME/models--google--paligemma-3b-pt-224" 2>/dev/null || echo "PaliGemma tokenizer not found!"
echo "=========================================="
echo ""

# Set LeRobot data directory (using new environment variable name)
export HF_LEROBOT_HOME="$SCRATCH/data/lerobot"
export WANDB_MODE=offline

TRAIN_DATASET="${TRAIN_DATASET:-data_v3_3tasks}"

# Generate timestamp for unique output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$SCRATCH/data/lerobot/outputs/train/pi0_$TRAIN_DATASET_${TIMESTAMP}"
echo "Output directory: $OUTPUT_DIR"


python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=local \
    --dataset.root="$SCRATCH/data/lerobot/$TRAIN_DATASET" \
    --policy.type=pi0 \
    --output_dir="$OUTPUT_DIR" \
    --job_name="pi0_training_$TRAIN_DATASET" \
    --policy.use_amp=true \
    --policy.pretrained_path="$SCRATCH/data/lerobot/.cache/huggingface/models--lerobot--pi0_base/snapshots/e3a5218ef7a5903445baf2cb656912fc35dc8712" \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --steps=3000 \
    --policy.device=cuda \
    --batch_size=32 \
    --num_workers=12 \
    --wandb.enable=true \
    --wandb.mode=offline \
    --wandb.project=lerobot_hpc \
    --policy.push_to_hub=false
