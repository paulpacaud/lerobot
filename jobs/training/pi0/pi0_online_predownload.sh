#!/bin/bash

set -x
set -e

cd "$WORK/Projects/lerobot"

module purge
module load miniforge


eval "$(conda shell.bash hook)"

conda activate lerobot

export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Set HuggingFace cache directory to $SCRATCH to avoid filling home directory
export HF_HOME="$SCRATCH/data/lerobot/.cache/huggingface"

mkdir -p "$HF_HOME"

echo ""
echo "----------------------------------------"
echo "Pre-downloading Pi0 model dependencies"
echo "----------------------------------------"

# Check if user is authenticated with HuggingFace
echo "Checking HuggingFace authentication..."
if ! huggingface-cli whoami &>/dev/null; then
    echo "⚠ Not authenticated with HuggingFace Hub"
    echo "Please run: huggingface-cli login"
    echo "Or set HF_TOKEN environment variable with your token from https://huggingface.co/settings/tokens"
    exit 1
fi
echo "✓ Authenticated with HuggingFace Hub"
echo ""

# Download the pretrained Pi0 base model from HuggingFace Hub
# This includes PaliGemma (with SigLIP vision encoder) and Gemma expert weights
python -c "
from huggingface_hub import snapshot_download
import os

cache_dir = os.environ.get('HF_HOME', None)
print('Downloading Pi0 base model (lerobot/pi0_base)...')
print(f'Cache directory: {cache_dir}')

try:
    # Download the full model repository
    snapshot_download(
        repo_id='lerobot/pi0_base',
        cache_dir=cache_dir,
        resume_download=True,
    )
    print('✓ Pi0 base model downloaded successfully')
except Exception as e:
    print(f'Error downloading Pi0 base model: {e}')
    print('This model is required for training. Please ensure you have internet access.')
    exit(1)

print('')
print('Downloading PaliGemma tokenizer (google/paligemma-3b-pt-224)...')
try:
    # Download the PaliGemma tokenizer
    snapshot_download(
        repo_id='google/paligemma-3b-pt-224',
        cache_dir=cache_dir,
        resume_download=True,
        allow_patterns=['tokenizer*', '*.json', '*.txt', '*.model'],
    )
    print('✓ PaliGemma tokenizer downloaded successfully')
except Exception as e:
    print(f'Error downloading PaliGemma tokenizer: {e}')
    print('This tokenizer is required for training. Please ensure you have internet access.')
    exit(1)
"


echo ""
echo "----------------------------------------"
echo "Verifying Pi0 model components"
echo "----------------------------------------"

# Verify that the model can be loaded (imports transformers components)
python -c "
import torch
from lerobot.utils.import_utils import _transformers_available

if not _transformers_available:
    print('⚠ Transformers is not available, please install it')
    exit(1)

print('✓ Transformers library is available')

# Verify critical transformers components for Pi0
try:
    from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
    from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
    print('✓ PaliGemma and Gemma models are available')
except ImportError as e:
    print(f'⚠ Error importing transformers models: {e}')
    exit(1)

try:
    from transformers.models.siglip import check
    if not check.check_whether_transformers_replace_is_installed_correctly():
        print('⚠ Transformers SigLIP patch is not installed correctly')
        print('Please ensure you have the correct transformers version for Pi0')
        exit(1)
    print('✓ SigLIP vision encoder patches are correctly installed')
except ImportError:
    print('⚠ SigLIP check module not found')
    print('Please ensure you have the correct transformers version for Pi0')
    exit(1)

print('')
print('All Pi0 dependencies are ready!')
"

echo ""
echo "✓ Pre-download complete!"
