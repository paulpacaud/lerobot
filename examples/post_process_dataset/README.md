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
| `constants/` | Camera calibration files (intrinsics.npz, extrinsics.npz), URDF files, and workspace definition |
| `reference_only/` | Reference implementations (depth projection, point cloud processing) |

## Commands

### 1. Convert v3 to v2 format
```bash
python examples/post_process_dataset/convert_lerobot_dataset_v3_to_v2.py \
    --input_dir=$HOME/lerobot_datasets/depth_test \
    --output_dir=$HOME/lerobot_datasets/depth_test_v2
```

### 2. Define workspace bounds (interactive visualization)
```bash
python examples/post_process_dataset/define_workspace.py \
    --dataset_dir=$HOME/lerobot_datasets/depth_test_v2 \
    --intrinsics_file=examples/post_process_dataset/constants/intrinsics.npz \
    --extrinsics_file=examples/post_process_dataset/constants/extrinsics.npz \
    --x_min=-0.21 --x_max=0.23 --y_min=-0.35 --y_max=0.3 --z_min=0.0 --z_max=0.4
```

### 3. Add point clouds to dataset
```bash
python examples/post_process_dataset/add_point_cloud_to_dataset.py \
    --dataset_dir=$HOME/lerobot_datasets/depth_test_v2 \
    --voxel_size=0.01
```

### 4. Convert joint-space to EE-space
```bash
# Convert and create new dataset
python examples/post_process_dataset/convert_joint_to_ee_space.py \
    --dataset_dir=$HOME/lerobot_datasets/depth_test_v2 \
    --output_dir=$HOME/lerobot_datasets/depth_test_ee \
    --urdf_path=./examples/post_process_dataset/constants/SO101/so101_new_calib.urdf
```

### 5. Verify FK with RGB image
```bash
python examples/post_process_dataset/visualize_robot_fk_with_rgb.py \
    --urdf_path=./examples/post_process_dataset/constants/SO101/so101_new_calib.urdf \
    --dataset_dir=$HOME/lerobot_datasets/depth_test_v2 \
    --episode_index=0 \
    --frame_index=100
```

### 6. Find robot-to-world transform
```bash
# Visualize EE trajectory on point cloud with translation offset
python examples/post_process_dataset/visualize_ee_trajectory_with_transform.py \
    --dataset_dir=$HOME/lerobot_datasets/depth_test_ee \
    --episode_index=0 \
    --tx=-0.28 --ty=0.0 --tz=0.0
```

### 7. Visualize point cloud
```bash
python examples/post_process_dataset/visualize_postprocessed_pcd.py \
    --dataset_dir=$HOME/lerobot_datasets/depth_test_v2 \
    --episode_index=0 --frame_index=100
```

## Notes

### Calibration
The `convert_joint_to_ee_space.py` script uses forward kinematics with the robot URDF. The URDF joint zeros must match the calibration conventions used when recording the dataset. Use the `so101_new_calib.urdf` from the SO-ARM100 repository for best results.

### Robot-to-World Transform
The EE positions from FK are in the robot base frame. To align with point clouds (which are in world frame), you need to find the transform:
```
ee_world = ee_robot + translation_offset
```
Use `visualize_ee_trajectory_with_transform.py` to manually find this offset by adjusting `--tx`, `--ty`, `--tz` until the trajectory aligns with the gripper in the point cloud.
