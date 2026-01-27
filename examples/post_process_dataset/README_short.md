# Post-Process Dataset Tools

### 0. Push/pull to hub
huggingface-cli upload ${HF_USER}/data_v3_3tasks $HOME/lerobot_datasets/data_v3_3tasks --repo-type dataset

huggingface-cli download paulpacaud/put_cube_in_spot_pointact \
  --repo-type dataset \
  --local-dir put_cube_in_spot_pointact \
  --local-dir-use-symlinks False

### Full Pipeline

```bash
python -m examples.post_process_dataset.run_full_pipeline --input_dir=$HOME/lerobot_datasets/put_banana_and_toy_in_plates --output_dir=$HOME/lerobot_datasets/put_banana_and_toy_in_plates_pointact
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
python examples/post_process_dataset/visualize_pointact_dataset.py --dataset_dir=$HOME/lerobot_datasets/put_cube_in_spot_pointact --episode_index=0 --pcd_frame=0
```

### 12. Push to Hub
huggingface-cli upload ${HF_USER}/put_cube_in_spot_pointact $HOME/lerobot_datasets/put_cube_in_spot_pointact --repo-type dataset

# Training
## Merge datasets

