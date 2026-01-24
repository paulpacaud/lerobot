#!/bin/bash

set -x
set -e

cd "$WORK/Projects/lerobot"

module purge
module load miniforge

eval "$(conda shell.bash hook)"

conda activate lerobot

export PYTHONPATH="$(pwd):${PYTHONPATH}"

echo ""
echo "----------------------------------------"
echo "Pre-downloading model dependencies"
echo "----------------------------------------"
# Download ResNet18 weights that ACT policy uses for vision encoder
python -c "
import torch
import torchvision.models as models
print('Downloading ResNet18 weights for ACT vision encoder...')
try:
    model = models.resnet18(weights='IMAGENET1K_V1')
    print('✓ ResNet18 weights downloaded successfully')
except Exception as e:
    print(f'Note: {e}')
    print('This may be handled during training')
"