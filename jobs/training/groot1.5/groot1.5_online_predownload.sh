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
echo "Pre-downloading GR00T N1.5 model dependencies"
echo "----------------------------------------"

# Check if user is authenticated with HuggingFace
echo "Checking HuggingFace authentication..."
if ! huggingface-cli whoami &>/dev/null; then
    echo "WARNING: Not authenticated with HuggingFace Hub"
    echo "Please run: huggingface-cli login"
    echo "Or set HF_TOKEN environment variable with your token from https://huggingface.co/settings/tokens"
    exit 1
fi
echo "Authenticated with HuggingFace Hub"
echo ""

# Download the pretrained GR00T N1.5 base model from HuggingFace Hub
# This includes Eagle2.5 VL backbone (SigLIP vision + Qwen2 LLM) and Flow Matching Action Head
python -c "
from huggingface_hub import snapshot_download
import os

cache_dir = os.environ.get('HF_HOME', None)
print('Downloading GR00T N1.5 base model (nvidia/GR00T-N1.5-3B)...')
print(f'Cache directory: {cache_dir}')

try:
    # Download the full model repository
    path = snapshot_download(
        repo_id='nvidia/GR00T-N1.5-3B',
        cache_dir=cache_dir,
        resume_download=True,
    )
    print(f'GR00T N1.5 base model downloaded successfully to: {path}')
except Exception as e:
    print(f'Error downloading GR00T N1.5 base model: {e}')
    print('This model is required for training. Please ensure you have internet access.')
    exit(1)

print('')
print('Downloading Eagle2.5 VL tokenizer/processor assets (lerobot/eagle2hg-processor-groot-n1p5)...')
try:
    # Download the Eagle2.5 VL processor/tokenizer assets
    path = snapshot_download(
        repo_id='lerobot/eagle2hg-processor-groot-n1p5',
        cache_dir=cache_dir,
        resume_download=True,
    )
    print(f'Eagle2.5 VL tokenizer/processor downloaded successfully to: {path}')
except Exception as e:
    print(f'Error downloading Eagle2.5 VL tokenizer/processor: {e}')
    print('This tokenizer is required for training. Please ensure you have internet access.')
    exit(1)
"


echo ""
echo "----------------------------------------"
echo "Verifying GR00T N1.5 model components"
echo "----------------------------------------"

# Verify that the model can be loaded (imports transformers components)
python -c "
import torch
import os

print('Verifying GR00T N1.5 dependencies...')
print('')

# Check flash attention availability
try:
    import flash_attn
    print(f'Flash Attention {flash_attn.__version__} is available')
except ImportError:
    print('WARNING: Flash Attention is not installed')
    print('GR00T N1.5 requires flash attention. Please install it:')
    print('  pip install flash-attn>=2.5.9,<3.0.0 --no-build-isolation')
    exit(1)

# Check transformers availability
try:
    import transformers
    print(f'Transformers {transformers.__version__} is available')
except ImportError:
    print('ERROR: Transformers is not available')
    exit(1)

# Check accelerate availability (required for multi-GPU training)
try:
    import accelerate
    print(f'Accelerate {accelerate.__version__} is available')
except ImportError:
    print('WARNING: Accelerate is not installed (required for multi-GPU training)')
    print('  pip install accelerate')

# Verify the model files exist in cache
hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))

# Check GR00T model
groot_model_dir = os.path.join(hf_home, 'hub', 'models--nvidia--GR00T-N1.5-3B')
if os.path.exists(groot_model_dir):
    print(f'GR00T N1.5 model found at: {groot_model_dir}')
else:
    # Try alternative path structure
    groot_model_dir = os.path.join(hf_home, 'models--nvidia--GR00T-N1.5-3B')
    if os.path.exists(groot_model_dir):
        print(f'GR00T N1.5 model found at: {groot_model_dir}')
    else:
        print(f'WARNING: GR00T N1.5 model directory not found')

# Check Eagle tokenizer
eagle_tokenizer_dir = os.path.join(hf_home, 'hub', 'models--lerobot--eagle2hg-processor-groot-n1p5')
if os.path.exists(eagle_tokenizer_dir):
    print(f'Eagle2.5 VL tokenizer found at: {eagle_tokenizer_dir}')
else:
    eagle_tokenizer_dir = os.path.join(hf_home, 'models--lerobot--eagle2hg-processor-groot-n1p5')
    if os.path.exists(eagle_tokenizer_dir):
        print(f'Eagle2.5 VL tokenizer found at: {eagle_tokenizer_dir}')
    else:
        print(f'WARNING: Eagle2.5 VL tokenizer directory not found')

print('')
print('All GR00T N1.5 dependencies are ready!')
"

echo ""
echo "Pre-download complete!"
