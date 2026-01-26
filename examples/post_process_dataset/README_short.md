# Post-Process Dataset Tools

## Pipeline

1. Convert v3 → v2 format
2. Define workspace bounds (optional)
3. Add point clouds to dataset
4. Convert joint-space to EE-space
5. Find robot-to-world transform
6. Visualize results

## Files

| File | Description |
|------|-------------|
| `convert_lerobot_dataset_v3_to_v2.py` | Converts LeRobot v3 datasets to v2.1 format (per-episode parquet/video files) |
| `define_workspace.py` | Interactive tool to visualize and choose workspace bounds for point cloud cropping |
| `add_point_cloud_to_dataset.py` | Computes voxelized point clouds from RGB+depth and stores in LMDB (supports v2.1 and v3.0) |
| `convert_joint_to_ee_space.py` | Converts joint-space state/action to end-effector (Cartesian) space using FK |
| `visualize_postprocessed_pcd.py` | 3D visualization of point clouds with coordinate axes |
| `visualize_robot_fk_with_rgb.py` | Side-by-side comparison of RGB image and URDF robot visualization to verify FK |
| `visualize_ee_trajectory_with_transform.py` | Visualize EE trajectory on point cloud with adjustable translation offset |
| `convert_to_pointact_format.py` | Converts dataset to PointAct format with EE+joint states, resized images, and renamed keys |
| `visualize_pointact_dataset.py` | Visualize PointAct dataset with EE trajectory and point cloud |
| `constants/` | Camera calibration files (intrinsics.npz, extrinsics.npz), URDF files, and workspace definition |
| `reference_only/` | Reference implementations (depth projection, point cloud processing) |

## Commands

### Record dataset


### 0. Push/pull to hub
huggingface-cli upload ${HF_USER}/data_v3_3tasks $HOME/lerobot_datasets/data_v3_3tasks --repo-type dataset

huggingface-cli download paulpacaud/hang_mug_pointact \
  --repo-type dataset \
  --local-dir hang_mug_pointact \
  --local-dir-use-symlinks False

huggingface-cli download paulpacaud/put_banana_in_plate_pointact \
  --repo-type dataset \
  --local-dir put_banana_in_plate_pointact \
  --local-dir-use-symlinks False

huggingface-cli download paulpacaud/put_cube_in_spot_pointact \
  --repo-type dataset \
  --local-dir put_cube_in_spot_pointact \
  --local-dir-use-symlinks False

huggingface-cli download paulpacaud/hang_mug \
  --repo-type dataset \
  --local-dir hang_mug \
  --local-dir-use-symlinks False

huggingface-cli download paulpacaud/put_banana_in_plate \
  --repo-type dataset \
  --local-dir put_banana_in_plate \
  --local-dir-use-symlinks False

huggingface-cli download paulpacaud/put_cube_in_spot \
  --repo-type dataset \
  --local-dir put_cube_in_spot \
  --local-dir-use-symlinks False

huggingface-cli download paulpacaud/data_v3_3tasks \
  --repo-type dataset \
  --local-dir data_v3_3tasks \
  --local-dir-use-symlinks False
### 1. Convert v3 to v2 format
```bash
python examples/post_process_dataset/convert_lerobot_dataset_v3_to_v2.py \
    --input_dir=$HOME/lerobot_datasets/hang_mug \
    --output_dir=$HOME/lerobot_datasets/hang_mug_v2
```

### 3. Add point clouds to dataset
```bash
python -m examples.post_process_dataset.add_point_cloud_to_dataset \
    --dataset_dir=$HOME/lerobot_datasets/hang_mug_v2 \
    --voxel_size=0.01 --num_workers=8
```

### 10. Convert to PointAct format
```bash
python examples/post_process_dataset/convert_to_pointact_format.py --dataset_dir=$HOME/lerobot_datasets/hang_mug_v2 --output_dir=$HOME/lerobot_datasets/hang_mug_pointact --urdf_path=./examples/post_process_dataset/constants/SO101/so101_new_calib.urdf --tx=-0.28 --ty=0.03 --tz=0.05
```


### 11. Visualize PointAct dataset
```bash
python examples/post_process_dataset/visualize_pointact_dataset.py --dataset_dir=$HOME/lerobot_datasets/hang_mug_pointact --episode_index=0 --pcd_frame=0
```

### 12. Push to Hub
huggingface-cli upload ${HF_USER}/hang_mug_pointact $HOME/lerobot_datasets/hang_mug_pointact --repo-type dataset