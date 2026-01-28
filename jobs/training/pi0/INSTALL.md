conda create -y -n lerobot_pi0 python=3.10
conda activate lerobot_pi0
conda install ffmpeg=7.1.1 -c conda-forge
pip install --no-cache-dir -e ".[feetech,intelrealsense]" 
pip install --no-cache-dir -e ".[pi]"
