# Post-Process Dataset Tools

## Pipeline

1. Convert v3 → v2 format
2. Define workspace bounds (optional)
3. Add point clouds to dataset
4. Visualize results

## Files

| File | Description |
|------|-------------|
| `convert_lerobot_dataset_v3_to_v2.py` | Converts LeRobot v3 datasets to v2.1 format (per-episode parquet/video files) |
| `define_workspace.py` | Interactive tool to visualize and choose workspace bounds for point cloud cropping |
| `add_point_cloud_to_dataset.py` | Computes voxelized point clouds from RGB+depth and stores in LMDB (supports v2.1 and v3.0) |
| `visualize_postprocessed_pcd.py` | 3D visualization of point clouds with coordinate axes |
| `constants/` | Camera calibration files (intrinsics.npz, extrinsics.npz) and workspace definition |
| `reference_only/` | Reference implementations (depth projection, point cloud processing) |

## Commands

```bash
# 1. Convert v3 to v2 format
python examples/post_process_dataset/convert_lerobot_dataset_v3_to_v2.py \
    --input_dir=$HOME/lerobot_datasets/depth_test \
    --output_dir=$HOME/lerobot_datasets/depth_test_v2

# 2. Define workspace bounds (interactive visualization)
python examples/post_process_dataset/define_workspace.py \
    --dataset_dir=$HOME/lerobot_datasets/depth_test_v2 \
    --intrinsics_file=examples/post_process_dataset/constants/intrinsics.npz \
    --extrinsics_file=examples/post_process_dataset/constants/extrinsics.npz \
    --x_min=-0.21 --x_max=0.23 --y_min=-0.35 --y_max=0.3 --z_min=0.0 --z_max=0.4

# 3. Add point clouds to dataset (uses constants folder defaults)
python -m examples/post_process_dataset/add_point_cloud_to_dataset.py \
    --dataset_dir=$HOME/lerobot_datasets/depth_test_v2 \
    --voxel_size=0.01

# 4. Visualize point cloud
python examples/post_process_dataset/visualize_postprocessed_pcd.py \
    --dataset_dir=$HOME/lerobot_datasets/depth_test_v2 \
    --episode_index=0 --frame_index=100

# Visualize sequence
python examples/post_process_dataset/visualize_postprocessed_pcd.py \
    --dataset_dir=$HOME/lerobot_datasets/depth_test_v2 \
    --sequence --num_frames=100
```
