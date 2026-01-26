#!/usr/bin/env python

"""
Replays the actions of an episode from an EE-space dataset on a robot using inverse kinematics.

This script takes end-effector (EE) positions from a PointAct-format dataset,
converts them to joint positions using inverse kinematics, and sends them to the robot.

This allows validating that the EE representation is sufficient to accurately reproduce
the recorded trajectories.

Usage:
```bash
python examples/post_process_dataset/lerobot_replay_EE.py --dataset_dir=/home/prl-tiago/lerobot_datasets/put_cube_in_spot_pointact --episode_index=0 --robot_port=/dev/ttyACM0

# With slowdown (5x slower)
python examples/post_process_dataset/lerobot_replay_EE.py --dataset_dir=/home/prl-tiago/lerobot_datasets/put_cube_in_spot_pointact --episode_index=0 --robot_port=/dev/ttyACM0 --slowdown=5.0
```
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tap import Tap

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.rotation import Rotation
from lerobot.utils.utils import log_say

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_URDF_PATH = "./examples/post_process_dataset/constants/SO101/so101_new_calib.urdf"
DEFAULT_CHUNK_SIZE = 1000

# Joint names for the SO100/SO101 arm (excluding gripper for IK)
ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ALL_MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class Args(Tap):
    """Arguments for replaying an EE-space dataset on a robot."""

    dataset_dir: str  # Path to the PointAct dataset directory
    episode_index: int = 0  # Episode to replay
    robot_port: str = "/dev/ttyACM0"  # USB port for the robot
    robot_id: str = "follower_arm"  # Robot identifier
    urdf_path: str = DEFAULT_URDF_PATH  # Path to the robot URDF file
    target_frame: str = "gripper_frame_link"  # End-effector frame in URDF
    use_degrees: bool = True  # Use degrees for motor control (must match calibration)
    play_sounds: bool = True  # Use vocal synthesis for events
    slowdown: float = 1.0  # Slowdown factor (e.g., 5.0 = 5x slower)


def load_info(dataset_path: Path) -> dict:
    """Load info.json from dataset root."""
    with open(dataset_path / "meta" / "info.json") as f:
        return json.load(f)


def load_episode_data(dataset_path: Path, episode_index: int) -> pd.DataFrame:
    """Load episode data from parquet file."""
    chunk_idx = episode_index // DEFAULT_CHUNK_SIZE
    parquet_path = dataset_path / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{episode_index:06d}.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Episode parquet file not found: {parquet_path}")

    return pd.read_parquet(parquet_path)


def ee_action_to_joint_action(
    ee_action: np.ndarray,
    kinematics: RobotKinematics,
    current_joints: np.ndarray,
    translation_offset: np.ndarray,
) -> dict[str, float]:
    """
    Convert EE action to joint action using inverse kinematics.

    Args:
        ee_action: Array of shape (7,) with [x, y, z, rx, ry, rz, gripper]
                   in world frame (with translation offset applied)
        kinematics: RobotKinematics instance for IK computation
        current_joints: Current joint positions in degrees (for IK initial guess)
        translation_offset: [tx, ty, tz] offset applied during dataset creation

    Returns:
        Dictionary mapping motor names to joint positions
    """
    # Extract position and orientation from EE action
    position_world = ee_action[:3]
    rotation_vec = ee_action[3:6]
    gripper_pos = ee_action[6]

    # Convert from world frame to robot frame by undoing the translation offset
    position_robot = position_world - translation_offset

    print(f"EEF_world = T {position_world} R {rotation_vec} S {gripper_pos}")
    print(f"EEF_robot = T {position_robot} R {rotation_vec} S {gripper_pos}")

    # Build 4x4 transformation matrix for IK
    T_desired = np.eye(4, dtype=np.float64)
    T_desired[:3, :3] = Rotation.from_rotvec(rotation_vec).as_matrix()
    T_desired[:3, 3] = position_robot

    # Compute inverse kinematics with high orientation weight to match full 6-DOF pose
    joint_positions = kinematics.inverse_kinematics(
        current_joints[:5], T_desired, position_weight=1.0, orientation_weight=1.0
    )

    # Build action dictionary for robot
    action = {}
    for i, motor_name in enumerate(ARM_JOINT_NAMES):
        action[f"{motor_name}.pos"] = float(joint_positions[i])
    action["gripper.pos"] = float(gripper_pos)

    return action


def main():
    args = Args().parse_args()
    dataset_path = Path(args.dataset_dir)

    # Load dataset info
    logging.info(f"Loading dataset from: {dataset_path}")
    info = load_info(dataset_path)
    fps = info["fps"]
    total_episodes = info["total_episodes"]

    if args.episode_index >= total_episodes:
        raise ValueError(f"Episode index {args.episode_index} out of range (total: {total_episodes})")

    # Get translation offset from dataset conversion info
    conversion_info = info.get("conversion_info", {}).get("pointact_conversion", {})
    translation_offset = np.array(
        conversion_info.get("translation_offset", [0.0, 0.0, 0.0]),
        dtype=np.float64
    )
    logging.info(f"Translation offset (world -> robot): {translation_offset}")

    # Load episode data
    logging.info(f"Loading episode {args.episode_index}...")
    episode_df = load_episode_data(dataset_path, args.episode_index)
    logging.info(f"Loaded {len(episode_df)} frames")

    # Initialize kinematics solver
    logging.info(f"Loading URDF from: {args.urdf_path}")
    kinematics = RobotKinematics(
        urdf_path=args.urdf_path,
        target_frame_name=args.target_frame,
        joint_names=ARM_JOINT_NAMES,
    )

    # Initialize robot
    logging.info(f"Connecting to robot on port: {args.robot_port}")
    robot_config = SO100FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        use_degrees=args.use_degrees,
    )
    robot = SO100Follower(robot_config)
    robot.connect()

    if not robot.is_connected:
        raise RuntimeError("Failed to connect to robot!")

    try:
        effective_fps = fps / args.slowdown
        logging.info(f"Replay speed: {args.slowdown}x slowdown (effective FPS: {effective_fps:.1f})")
        log_say(f"Replaying episode {args.episode_index} with IK", args.play_sounds, blocking=True)

        for idx, (_, row) in enumerate(episode_df.iterrows()):
            start_time = time.perf_counter()

            # Get EE action from dataset
            ee_action = np.array(row["action"], dtype=np.float64)

            # Get current robot observation for IK initial guess
            robot_obs = robot.get_observation()
            current_joints = np.array([
                robot_obs[f"{motor}.pos"] for motor in ALL_MOTOR_NAMES
            ], dtype=np.float64)

            # Convert EE action to joint action using IK
            joint_action = ee_action_to_joint_action(
                ee_action=ee_action,
                kinematics=kinematics,
                current_joints=current_joints,
                translation_offset=translation_offset,
            )

            # Send action to robot
            # robot.send_action(joint_action)

            # Timing with slowdown factor
            dt_s = time.perf_counter() - start_time
            target_dt = args.slowdown / fps
            precise_sleep(max(target_dt - dt_s, 0.0))

            if idx % 50 == 0:
                logging.info(f"Frame {idx}/{len(episode_df)}, dt={dt_s*1000:.1f}ms")

        log_say("Replay complete", args.play_sounds, blocking=True)
        logging.info("Replay complete!")

    finally:
        robot.disconnect()
        logging.info("Robot disconnected.")


if __name__ == "__main__":
    main()
