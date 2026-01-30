#!/bin/bash
#SBATCH --job-name=poinact_conversion
#SBATCH -A hjx@v100
#SBATCH -C v100
#SBATCH --qos=qos_gpu-dev
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread
#SBATCH --time=00:15:00
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

INPUT_DATASET="${INPUT_DATASET:-put_socks_into_drawer}"

python -m examples.post_process_dataset.create_small_pointact_dataset --dataset_dir="${SCRATCH}/data/lerobot/${INPUT_DATASET}" --num_episodes=2