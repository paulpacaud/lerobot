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
| `constants/` | Camera calibration files (intrinsics.npz, extrinsics.npz), URDF files, workspace definition, and robot frame position (ROBOT_FRAME) |
| `reference_only/` | Reference implementations (depth projection, point cloud processing) |

## Commands

# Record dataset

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



# 0. Push/pull to hub
huggingface-cli upload ${HF_USER}/hang_mug_test $HOME/lerobot_datasets/hang_mug_test --repo-type dataset

huggingface-cli download paulpacaud/hang_mug_test \
  --repo-type dataset \
  --local-dir hang_mug_test \
  --local-dir-use-symlinks False

# 1. Convert v3 to v2 format
```bash
python examples/post_process_dataset/convert_lerobot_dataset_v3_to_v2.py \
    --input_dir=$HOME/lerobot_datasets/put_cube_in_spot \
    --output_dir=$HOME/lerobot_datasets/put_cube_in_spot_v2
```

# 2. Add point clouds to dataset
```bash
python -m examples.post_process_dataset.add_point_cloud_to_dataset \
    --dataset_dir=$HOME/lerobot_datasets/put_cube_in_spot_v2 \
    --voxel_size=0.01
```

# 3. Convert to PointAct format
```bash
python -m examples.post_process_dataset.convert_to_pointact_format --dataset_dir=/home/prl-tiago/lerobot_datasets/put_cube_in_spot_v2 --output_dir=/home/prl-tiago/lerobot_datasets/put_cube_in_spot_pointact --urdf_path=./examples/post_process_dataset/constants/SO101/so101_new_calib.urdf
```

# 4. Visualize PointAct dataset
```bash
python examples/post_process_dataset/visualize_pointact_dataset.py --dataset_dir=$HOME/lerobot_datasets/put_cube_in_spot_pointact --episode_index=0 --pcd_frame=250
```

# 5. Replay episode

xdg-open /home/prl-tiago/lerobot_datasets/put_cube_in_spot_pointact/videos/chunk-000/observation.images.front_image/episode_000000.mp4

## Direct joint replay (no IK)                                                                                                                                                                                                                                                                                                                                                     
python examples/post_process_dataset/lerobot_replay_EE.py --dataset_dir=/home/prl-tiago/lerobot_datasets/put_cube_in_spot_pointact --episode_index=0 --robot_port=/dev/ttyACM0 --replay_target=joint 

## EE cartesian replay with IK                                                                                                                                                                                                                                                                                                                                                     
python examples/post_process_dataset/lerobot_replay_EE.py --dataset_dir=/home/prl-tiago/lerobot_datasets/put_cube_in_spot_pointact --episode_index=0 --robot_port=/dev/ttyACM0 --replay_target=ee                                                                                                                                                                                    

# 6. Push to Hub
huggingface-cli upload ${HF_USER}/hang_mug_test_pointact $HOME/lerobot_datasets/hang_mug_test_pointact --repo-type dataset

huggingface-cli download paulpacaud/put_cube_in_spot_pointact \
  --repo-type dataset \
  --local-dir put_cube_in_spot_pointact \
  --local-dir-use-symlinks False