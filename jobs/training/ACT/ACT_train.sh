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

#  --policy.n_obs_steps int
  #  --policy.input_features [Dict]
  #                        `input_features` can be set to None/null in order to infer those values from the dataset. (default: {})
  #  --policy.output_features [Dict]
  #  --policy.device [str]
  #                        e.g. "cuda", "cuda:0", "cpu", or "mps" (default: None)
  #  --policy.use_amp bool
  #                        `use_amp` determines whether to use Automatic Mixed Precision (AMP) for training and evaluation. With AMP, automatic
  #                        gradient scaling is used. (default: False)
  #  --policy.use_peft bool
  #                        Whether the policy employed PEFT for training. (default: False)
  #  --policy.push_to_hub bool
  #                        type: ignore[assignment] # TODO: use a different name to avoid override (default: True)
  #  --policy.repo_id [str]
  #  --policy.private [bool]
  #                        Upload on private repository on the Hugging Face hub. (default: None)
  #  --policy.tags [List]  Add tags to your policy on the hub. (default: None)
  #  --policy.license [str]
  #                        Add tags to your policy on the hub. (default: None)
  #  --policy.pretrained_path [Path]
  #                        Either the repo ID of a model hosted on the Hub or a path to a directory containing weights saved using
  #                        `Policy.save_pretrained`. If not provided, the policy is initialized from scratch. (default: None)
  #  --policy.chunk_size int
  #  --policy.n_action_steps int
  #  --policy.normalization_mapping Dict
  #  --policy.vision_backbone str
  #  --policy.pretrained_backbone_weights [str]
  #  --policy.replace_final_stride_with_dilation int
  #  --policy.pre_norm bool
  #  --policy.dim_model int
  #  --policy.n_heads int
  #  --policy.dim_feedforward int
  #  --policy.feedforward_activation str
  #  --policy.n_encoder_layers int
  #  --policy.n_decoder_layers int
  #  --policy.use_vae bool
  #  --policy.latent_dim int
  #  --policy.n_vae_encoder_layers int
  #  --policy.temporal_ensemble_coeff [float]
  #  --policy.dropout float
  #  --policy.kl_weight float
  #  --policy.optimizer_lr float
  #                        Training preset (default: 1e-05)
  #  --policy.optimizer_weight_decay float
  #  --policy.optimizer_lr_backbone float

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

lerobot-train \
  --dataset.repo_id=local \
  --dataset.root="$SCRATCH/data/lerobot/put_fruits_in_plate" \
  --policy.type=act \
  --output_dir="$SCRATCH/data/lerobot/outputs/train/act_put_fruits_in_plate" \
  --job_name=act_put_fruits_in_plate \
  --policy.device=cuda \
  --policy.use_amp=true \
  --batch_size=16 \
  --num_workers=12 \
  --wandb.enable=true \
  --wandb.mode=offline \
  --wandb.project=lerobot_hpc \
  --policy.push_to_hub=false