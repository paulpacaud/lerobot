# Post-Process Dataset Tools

### 0. Push/pull to hub
huggingface-cli upload ${HF_USER}/data_v3_3tasks $HOME/lerobot_datasets/data_v3_3tasks --repo-type dataset

huggingface-cli download paulpacaud/put_cube_in_spot \
  --repo-type dataset \
  --local-dir put_cube_in_spot \
  --local-dir-use-symlinks False

huggingface-cli download paulpacaud/put_banana_and_toy_in_plates \
  --repo-type dataset \
  --local-dir put_banana_and_toy_in_plates \
  --local-dir-use-symlinks False

huggingface-cli download paulpacaud/put_socks_into_drawer \
  --repo-type dataset \
  --local-dir put_socks_into_drawer \
  --local-dir-use-symlinks False

huggingface-cli upload paulpacaud/move_plates_from_rack_to_box $HOME/lerobot_datasets/move_plates_from_rack_to_box --repo-type dataset

### Full Pipeline

```bash
python -m examples.post_process_dataset.run_full_pipeline --input_dir=$HOME/lerobot_datasets/put_sockets_into_drawer --output_dir=$HOME/lerobot_datasets/put_sockets_into_drawer_pointact
python -m examples.post_process_dataset.run_full_pipeline --input_dir=$HOME/lerobot_datasets/put_cube_in_spot --output_dir=$HOME/lerobot_datasets/put_cube_in_spot_pointact
python -m examples.post_process_dataset.run_full_pipeline --input_dir=$HOME/lerobot_datasets/put_banana_and_toy_in_plates --output_dir=$HOME/lerobot_datasets/put_banana_and_toy_in_plates_pointact

huggingface-cli upload ${HF_USER}/put_sockets_into_drawer_pointact $HOME/lerobot_datasets/put_sockets_into_drawer_pointact --repo-type dataset
huggingface-cli upload ${HF_USER}/put_cube_in_spot_pointact $HOME/lerobot_datasets/put_cube_in_spot_pointact --repo-type dataset
huggingface-cli upload ${HF_USER}/put_banana_and_toy_in_plates_pointact $HOME/lerobot_datasets/put_banana_and_toy_in_plates_pointact --repo-type dataset
```

### Individual Steps (Manual)

### 1. Convert v3 to v2 format
```bash
python examples/post_process_dataset/convert_lerobot_dataset_v3_to_v2.py \
    --input_dir=$HOME/lerobot_datasets/put_cube_in_spot \
    --output_dir=$HOME/lerobot_datasets/put_cube_in_spot_v2
```

### 3. Add point clouds to dataset
```bash
python -m examples.post_process_dataset.add_point_cloud_to_dataset \
    --dataset_dir=$HOME/lerobot_datasets/put_cube_in_spot_v2 \
    --voxel_size=0.01 --num_workers=8
    
```

### 10. Convert to PointAct format
```bash
python -m examples.post_process_dataset.convert_to_pointact_format --dataset_dir=$HOME/lerobot_datasets/put_cube_in_spot_v2 --output_dir=$HOME/lerobot_datasets/put_cube_in_spot_pointact --urdf_path=./examples/post_process_dataset/constants/SO101/so101_new_calib.urdf
```


### 11. Visualize PointAct dataset
```bash
python examples/post_process_dataset/visualize_pointact_dataset.py --dataset_dir=$HOME/lerobot_datasets/put_sockets_into_drawer_pointact/ --episode_index=10 --pcd_frame=100
```

# 5. Replay episode

xdg-open /home/prl-tiago/lerobot_datasets/put_cube_in_spot_pointact/videos/chunk-000/observation.images.front_image/episode_000010.mp4

## Direct joint replay (no IK)                                                                                                                                                                                                                                                                                                                                                     
python examples/post_process_dataset/lerobot_replay_EE.py --dataset_dir=/home/prl-tiago/lerobot_datasets/put_cube_in_spot_pointact --episode_index=10 --robot_port=/dev/ttyACM0 --replay_target=joint 

## EE cartesian replay with IK                                                                                                                                                                                                                                                                                                                                                     
python examples/post_process_dataset/lerobot_replay_EE.py --dataset_dir=/home/prl-tiago/lerobot_datasets/put_cube_in_spot_pointact --episode_index=10 --robot_port=/dev/ttyACM0 --replay_target=ee                                                                                                                                                                                    


### 12. Push to Hub
huggingface-cli upload ${HF_USER}/put_cube_in_spot_pointact $HOME/lerobot_datasets/put_cube_in_spot_pointact --repo-type dataset

# 13. Merge for training
lerobot-edit-dataset \
    --repo_id  /home/ppacaud/lerobot_datasets/pointact_3tasks \
    --operation.type merge \
    --operation.repo_ids "['/home/ppacaud/lerobot_datasets/put_cube_in_spot_pointact', '/home/ppacaud/lerobot_datasets/put_banana_and_toy_in_plates_pointact', '/home/ppacaud/lerobot_datasets/put_sockets_into_drawer_pointact']"

# Training

# Train baselines on EEF pose 
As we trim the datasets for pointact, we need to reuse the same data for training baselines, but in lerobot v3 format.
```bash

```