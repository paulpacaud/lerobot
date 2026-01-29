#!/bin/bash
#SBATCH --job-name=groot1.5_train
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
echo "GR00T N1.5 base model location:"
ls -d "$HF_HOME/hub/models--nvidia--GR00T-N1.5-3B" 2>/dev/null || ls -d "$HF_HOME/models--nvidia--GR00T-N1.5-3B" 2>/dev/null || echo "GR00T model not found!"
echo ""
echo "Eagle2.5 VL tokenizer location:"
ls -d "$HF_HOME/hub/models--lerobot--eagle2hg-processor-groot-n1p5" 2>/dev/null || ls -d "$HF_HOME/models--lerobot--eagle2hg-processor-groot-n1p5" 2>/dev/null || echo "Eagle tokenizer not found!"
echo "=========================================="
echo ""

# Set LeRobot data directory using new environment variable name
export HF_LEROBOT_HOME="$SCRATCH/data/lerobot"
export WANDB_MODE=offline

TRAIN_DATASET="${TRAIN_DATASET:-data_v3_3tasks}"

# Generate timestamp for unique output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$SCRATCH/data/lerobot/outputs/train/groot1.5_${TRAIN_DATASET}_${TIMESTAMP}"
echo "Output directory: $OUTPUT_DIR"

python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=local \
    --dataset.root="$SCRATCH/data/lerobot/$TRAIN_DATASET" \
    --policy.type=groot \
    --policy.base_model_path="$SCRATCH/data/lerobot/.cache/huggingface/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e" \
    --output_dir="$OUTPUT_DIR" \
    --job_name="groot1.5_training_$TRAIN_DATASET" \
    --policy.tune_llm=false \
    --policy.tune_visual=false \
    --policy.tune_projector=true \
    --policy.tune_diffusion_model=true \
    --policy.use_bf16=true \
    --steps=10000 \
    --save_freq=2500 \
    --batch_size=32 \
    --num_workers=12 \
    --wandb.enable=true \
    --wandb.mode=offline \
    --wandb.project=lerobot_hpc \
    --policy.push_to_hub=false
