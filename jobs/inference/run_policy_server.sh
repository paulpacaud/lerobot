#!/bin/bash
#SBATCH --job-name=evRP
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --gres=gpu:a100:1
#SBATCH--cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=40:00:00
#SBATCH --mem=80G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.out
#SBATCH -p willow
#SBATCH -A willow

set -x
set -e

export XDG_RUNTIME_DIR=$SCRATCH/tmp/runtime-$SLURM_JOBID; mkdir -p $XDG_RUNTIME_DIR; chmod 700 $XDG_RUNTIME_DIR

module purge
pwd; hostname; date

############# CONDA  #############
cd $HOME/Projects/lerobot
. $HOME/miniconda3/etc/profile.d/conda.sh

conda activate lerobot
export python_bin=$HOME/miniconda3/envs/lerobot/bin/python


############# SCRIPT  #############
