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

lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=so100_leader \
   --robot.cameras="{ front: {type: intelrealsense, serial_number_or_name: 147122078460, width: 640, height: 480, fps: 30, use_depth: true}}" \
    --display_data=true \
    --dataset.root=$HOME/lerobot_datasets/put_cube_in_spot \
    --dataset.push_to_hub=False \
    --dataset.repo_id=${HF_USER}/put_cube_in_spot \
    --dataset.num_episodes=10 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Put the banana in the blue plate" \
    --resume=true



### 0. Push/pull to hub
huggingface-cli upload ${HF_USER}/hang_mug_test $HOME/lerobot_datasets/hang_mug_test --repo-type dataset

huggingface-cli download paulpacaud/hang_mug_test \
  --repo-type dataset \
  --local-dir hang_mug_test \
  --local-dir-use-symlinks False

### 1. Convert v3 to v2 format
```bash
python examples/post_process_dataset/convert_lerobot_dataset_v3_to_v2.py \
    --input_dir=$HOME/lerobot_datasets/hang_mug_test \
    --output_dir=$HOME/lerobot_datasets/hang_mug_test_v2
```

### 2. Define workspace bounds (interactive visualization)
```bash
python examples/post_process_dataset/define_workspace.py \
    --dataset_dir=$HOME/lerobot_datasets/put_cube_in_spot_v2 \
    --intrinsics_file=examples/post_process_dataset/constants/intrinsics.npz \
    --extrinsics_file=examples/post_process_dataset/constants/extrinsics.npz \
    --x_min=-0.21 --x_max=0.23 --y_min=-0.35 --y_max=0.3 --z_min=0.0 --z_max=0.4
```

### 3. Add point clouds to dataset
```bash
python -m examples.post_process_dataset.add_point_cloud_to_dataset \
    --dataset_dir=$HOME/lerobot_datasets/hang_mug_test_v2 \
    --voxel_size=0.01
```

### 4. Visualize point cloud
```bash
python examples/post_process_dataset/visualize_postprocessed_pcd.py \
    --dataset_dir=$HOME/lerobot_datasets/put_cube_in_spot_v2 \
    --episode_index=0 --frame_index=100
```

### 5. Convert joint-space to EE-space (in the robot frame!)
```bash
# Convert to EE space with no translation offset (robot frame)
python examples/post_process_dataset/convert_joint_to_ee_space.py \
    --dataset_dir=$HOME/lerobot_datasets/put_cube_in_spot_v2 \
    --output_dir=$HOME/lerobot_datasets/put_cube_in_spot_ee \
    --urdf_path=./examples/post_process_dataset/constants/SO101/so101_new_calib.urdf \
    --tx=0.0 --ty=0.0 --tz=0.0
```

### 6. Verify FK with RGB image
```bash
python examples/post_process_dataset/visualize_robot_fk_with_rgb.py \
    --urdf_path=./examples/post_process_dataset/constants/SO101/so101_new_calib.urdf \
    --dataset_dir=$HOME/lerobot_datasets/put_cube_in_spot_v2 \
    --episode_index=0 \
    --frame_index=100
```

### 7. Find robot-to-world transform
```bash
# Visualize EE trajectory on point cloud with translation offset
python examples/post_process_dataset/visualize_ee_trajectory_with_transform.py \
    --dataset_dir=$HOME/lerobot_datasets/put_cube_in_spot_ee \
    --episode_index=0 \
    --pcd_frame=0 \
    --tx=-0.28 --ty=0.03 --tz=0.05
```

### 8. (Again) Convert joint-space to EE-space (but in the world frame!)
```bash
# Convert to EE space with translation offset (world frame)
# Use the offset found in step 7
python examples/post_process_dataset/convert_joint_to_ee_space.py \
    --dataset_dir=$HOME/lerobot_datasets/hang_mug_test_v2 \
    --output_dir=$HOME/lerobot_datasets/hang_mug_test_v2_ee \
    --urdf_path=./examples/post_process_dataset/constants/SO101/so101_new_calib.urdf \
    --tx=-0.2755 --ty=-0.0599 --tz=0.0257
```

### 9. (sanity check) Check what you got
```bash
# Visualize EE trajectory on point cloud with translation offset
python examples/post_process_dataset/visualize_ee_trajectory_with_transform.py \
    --dataset_dir=$HOME/lerobot_datasets/put_cube_in_spot_v2_ee \
    --episode_index=0 \
    --pcd_frame=0 \
    --tx=-0 --ty=0 --tz=0
```

### 10. Convert to PointAct format
```bash
python examples/post_process_dataset/convert_to_pointact_format.py --dataset_dir=$HOME/lerobot_datasets/hang_mug_test_v2 --output_dir=$HOME/lerobot_datasets/hang_mug_test_pointact --urdf_path=./examples/post_process_dataset/constants/SO101/so101_new_calib.urdf --tx=-0.2755 --ty=-0.0599 --tz=0.0257
```

This converts the dataset to PointAct format with:
- `observation.state`: (7,) [x, y, z, axis_angle1-3, gripper]
- `observation.states.ee_state`: (6,) [x, y, z, axis_angle1-3]
- `observation.states.joint_state`: (6,) joint positions (including gripper)
- `observation.states.gripper_state`: (1,) gripper openness
- `action`: (7,) [x, y, z, axis_angle1-3, gripper]
- `observation.images.front_image`: (256, 256, 3) resized video
- `observation.points.frontview`: point cloud data

### 11. Visualize PointAct dataset
```bash
python examples/post_process_dataset/visualize_pointact_dataset.py --dataset_dir=$HOME/lerobot_datasets/hang_mug_test_pointact --episode_index=0 --pcd_frame=0
```

### 12. Push to Hub
huggingface-cli upload ${HF_USER}/put_cube_in_spot_pointact $HOME/lerobot_datasets/put_cube_in_spot_pointact --repo-type dataset

huggingface-cli download paulpacaud/put_cube_in_spot_pointact \
  --repo-type dataset \
  --local-dir put_cube_in_spot_pointact \
  --local-dir-use-symlinks False

## Notes

### Calibration
The `convert_joint_to_ee_space.py` script uses forward kinematics with the robot URDF. The URDF joint zeros must match the calibration conventions used when recording the dataset. Use the `so101_new_calib.urdf` from the SO-ARM100 repository for best results.

### Robot-to-World Transform
The EE positions from FK are in the robot base frame. To align with point clouds (which are in world frame), you need to find the transform:
```
ee_world = ee_robot + translation_offset
```
Use `visualize_ee_trajectory_with_transform.py` to manually find this offset by adjusting `--tx`, `--ty`, `--tz` until the trajectory aligns with the gripper in the point cloud.





python examples/post_process_dataset/goto_ee_position.py --robot_port=/dev/ttyACM0 --tx=-0.28 --ty=0 --tz=0.05 --duration 50
python examples/post_process_dataset/read_ee_position.py --robot_port=/dev/ttyACM0