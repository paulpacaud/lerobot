conda create -y -n lerobot_groot python=3.10
conda activate lerobot_groot
conda install ffmpeg=7.1.1 -c conda-forge
pip install -e ".[feetech,intelrealsense]"

# on the DGX we have cuda 12.8
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

pip install "torch>=2.2.1,<2.8.0" "torchvision>=0.21.0,<0.23.0" --index-url https://download.pytorch.org/whl/cu128
pip install ninja "packaging>=24.2,<26.0"
pip install "flash-attn>=2.5.9,<3.0.0" --no-build-isolation

python -c "import flash_attn; print(f'Flash Attention {flash_attn.__version__} imported successfully')"

pip install lerobot[groot]

# for inference
pip install -e ".[async]"