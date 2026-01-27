#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Replay a PointAct-format LeRobot dataset on a real robot.

This script supports two replay modes:
- "ee": Replay EE trajectory using inverse kinematics (converts EE pose to joint positions)
- "joint": Replay original joint commands directly from action.joints (no IK)

The PointAct dataset has:
- action: [x, y, z, axis_angle1, axis_angle2, axis_angle3, gripper_openness] (EE commands)
- action.joints: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper] (original joint commands)

Usage:
```bash
# Replay using EE cartesian trajectory with IK
python examples/post_process_dataset/lerobot_replay_EE.py --dataset_dir=/home/prl-tiago/lerobot_datasets/put_cube_in_spot_pointact --episode_index=0 --robot_port=/dev/ttyACM0 --replay_target=ee

# Replay using original joint commands directly
python examples/post_process_dataset/lerobot_replay_EE.py --dataset_dir=/home/prl-tiago/lerobot_datasets/put_cube_in_spot_pointact --episode_index=0 --robot_port=/dev/ttyACM0 --replay_target=joint
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
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.robot_kinematic_processor import InverseKinematicsEEToJoints
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.utils.robot_utils import precise_sleep

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_URDF_PATH = "./examples/post_process_dataset/constants/SO101/so101_new_calib.urdf"


class Args(Tap):
    """Arguments for replaying a PointAct dataset."""

    # Required
    dataset_dir: str  # Path to the PointAct-format LeRobot dataset

    # Robot configuration
    robot_port: str = "/dev/ttyACM0"  # USB port for the robot
    robot_id: str = "follower_arm"  # Robot identifier
    use_degrees: bool = False  # Use degrees for motor control (default matches SO100FollowerConfig)

    # Kinematics
    urdf_path: str = DEFAULT_URDF_PATH  # Path to the robot URDF file
    target_frame: str = "gripper_frame_link"  # End-effector frame in URDF

    # Replay configuration
    episode_index: int = 0  # Episode to replay
    start_frame: int = 0  # Frame index to start replay from
    end_frame: int = -1  # Frame index to end replay (-1 for all)
    fps: float | None = None  # Override FPS (None uses dataset FPS)
    replay_target: str = "ee"  # "ee" for EE cartesian with IK, "joint" for direct joint replay


    # Safety
    initial_move_duration: float = 3.0  # Duration (seconds) to move to first position
    initial_move_steps: int = 50  # Number of interpolation steps for initial move


def load_dataset_info(dataset_path: Path) -> dict:
    """Load info.json from dataset."""
    with open(dataset_path / "meta" / "info.json") as f:
        return json.load(f)


def get_translation_offset(info: dict) -> np.ndarray | None:
    """Extract translation offset from dataset info if present."""
    conversion_info = info.get("conversion_info", {})
    pointact_info = conversion_info.get("pointact_conversion", {})
    offset = pointact_info.get("translation_offset")
    if offset is not None:
        return np.array(offset, dtype=np.float64)
    return None


def load_episode_data(dataset_path: Path, episode_index: int) -> pd.DataFrame:
    """Load all frames for a specific episode."""
    data_dir = dataset_path / "data"
    all_frames = []

    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        for parquet_file in sorted(chunk_dir.glob("*.parquet")):
            df = pd.read_parquet(parquet_file)
            episode_frames = df[df["episode_index"] == episode_index]
            if len(episode_frames) > 0:
                all_frames.append(episode_frames)

    if not all_frames:
        raise ValueError(f"Episode {episode_index} not found in dataset")

    return pd.concat(all_frames, ignore_index=True).sort_values("frame_index")


def action_array_to_ee_action(
    action: np.ndarray,
    translation_offset: np.ndarray | None = None,
) -> RobotAction:
    """
    Convert PointAct action array to EE action dict.

    Args:
        action: Array [x, y, z, axis_angle1, axis_angle2, axis_angle3, gripper_openness]
        translation_offset: Offset that was added during conversion (will be subtracted to get robot frame)

    Returns:
        RobotAction dict with ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos
    """
    x, y, z = action[0], action[1], action[2]
    wx, wy, wz = action[3], action[4], action[5]
    gripper = action[6]

    # Transform from world frame to robot frame by reversing the translation offset
    if translation_offset is not None:
        x -= translation_offset[0]
        y -= translation_offset[1]
        z -= translation_offset[2]

    return {
        "ee.x": float(x),
        "ee.y": float(y),
        "ee.z": float(z),
        "ee.wx": float(wx),
        "ee.wy": float(wy),
        "ee.wz": float(wz),
        "ee.gripper_pos": float(gripper),
    }


def joint_array_to_joint_action(joint_state: np.ndarray, motor_names: list[str]) -> RobotAction:
    """
    Convert joint state array to joint action dict.

    Args:
        joint_state: Array [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        motor_names: List of motor names from robot

    Returns:
        RobotAction dict with {motor_name}.pos for each motor
    """
    return {f"{name}.pos": float(joint_state[i]) for i, name in enumerate(motor_names)}


def main():
    args = Args().parse_args()

    # Validate replay_target
    if args.replay_target not in ("ee", "joint"):
        raise ValueError(f"Invalid replay_target: {args.replay_target}. Must be 'ee' or 'joint'")

    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Load dataset info
    logging.info(f"Loading dataset from: {dataset_path}")
    logging.info(f"Replay mode: {args.replay_target}")
    info = load_dataset_info(dataset_path)

    dataset_fps = info.get("fps", 30)
    fps = args.fps if args.fps is not None else dataset_fps
    logging.info(f"Dataset FPS: {dataset_fps}, Replay FPS: {fps}")

    # Get translation offset (world frame -> robot frame) - only needed for EE mode
    translation_offset = None
    if args.replay_target == "ee":
        translation_offset = get_translation_offset(info)
        if translation_offset is not None:
            logging.info(f"Translation offset (world to robot): {translation_offset}")
        else:
            logging.warning("No translation offset found in dataset - assuming data is already in robot frame")

    # Load episode data
    logging.info(f"Loading episode {args.episode_index}...")
    episode_data = load_episode_data(dataset_path, args.episode_index)
    total_frames = len(episode_data)
    logging.info(f"Episode has {total_frames} frames")

    # Determine frame range
    start_frame = args.start_frame
    end_frame = args.end_frame if args.end_frame >= 0 else total_frames
    end_frame = min(end_frame, total_frames)
    logging.info(f"Replaying frames {start_frame} to {end_frame}")

    # Extract data based on replay mode
    if args.replay_target == "ee":
        actions = episode_data["action"].tolist()
        actions = [np.array(a, dtype=np.float32) for a in actions]
    else:  # joint mode - use original joint commands (action.joints), not observed states
        joint_actions = episode_data["action.joints"].tolist()
        joint_actions = [np.array(ja, dtype=np.float32) for ja in joint_actions]

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

    # Get motor names from robot
    motor_names = list(robot.bus.motors.keys())
    logging.info(f"Motor names: {motor_names}")

    # Initialize kinematics (needed for EE mode and for displaying current position)
    logging.info(f"Loading URDF from: {args.urdf_path}")
    kinematics = RobotKinematics(
        urdf_path=args.urdf_path,
        target_frame_name=args.target_frame,
        joint_names=motor_names,
    )

    # Build IK pipeline (only needed for EE mode)
    ee_to_joints_processor = None
    if args.replay_target == "ee":
        ee_to_joints_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
            steps=[
                InverseKinematicsEEToJoints(
                    kinematics=kinematics,
                    motor_names=motor_names,
                    initial_guess_current_joints=False,  # Open-loop replay
                ),
            ],
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )

    try:
        # Get current robot observation
        robot_obs = robot.get_observation()
        current_joints = np.array([robot_obs[f"{m}.pos"] for m in motor_names])
        logging.info(f"Current joints: {current_joints}")

        # Compute FK to show current EE position
        from lerobot.utils.rotation import Rotation

        T_current = kinematics.forward_kinematics(current_joints)
        current_pos = T_current[:3, 3]
        current_rot = Rotation.from_matrix(T_current[:3, :3]).as_rotvec()
        logging.info(f"Current EE position: [{current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}]")
        logging.info(f"Current EE rotation: [{current_rot[0]:.4f}, {current_rot[1]:.4f}, {current_rot[2]:.4f}]")

        # Get first target joints based on replay mode
        if args.replay_target == "ee":
            # Get first target EE action and convert to joints via IK
            first_ee_action = action_array_to_ee_action(
                actions[start_frame],
                translation_offset=translation_offset,
            )
            logging.info("First target EE position:")
            logging.info(f"  Position: [{first_ee_action['ee.x']:.4f}, {first_ee_action['ee.y']:.4f}, {first_ee_action['ee.z']:.4f}]")
            logging.info(f"  Rotation: [{first_ee_action['ee.wx']:.4f}, {first_ee_action['ee.wy']:.4f}, {first_ee_action['ee.wz']:.4f}]")
            logging.info(f"  Gripper: {first_ee_action['ee.gripper_pos']:.2f}")

            first_joint_action = ee_to_joints_processor((first_ee_action, robot_obs))
            first_target_joints = np.array([first_joint_action[f"{m}.pos"] for m in motor_names])
        else:  # joint mode
            first_target_joints = joint_actions[start_frame]
            logging.info(f"First target joints (from action.joints): {first_target_joints}")

        logging.info(f"First target joints: {first_target_joints}")

        # Move to first position with interpolation
        logging.info(f"Moving to start position over {args.initial_move_duration}s...")
        dt = args.initial_move_duration / args.initial_move_steps
        start_joints = current_joints.copy()

        for step in range(args.initial_move_steps + 1):
            t = step / args.initial_move_steps
            interp_joints = start_joints + t * (first_target_joints - start_joints)
            action = {f"{m}.pos": float(interp_joints[i]) for i, m in enumerate(motor_names)}
            robot.send_action(action)
            time.sleep(dt)

        logging.info("At start position. Beginning replay...")
        time.sleep(0.5)

        # Main replay loop
        for frame_idx in range(start_frame, end_frame):
            t0 = time.perf_counter()

            if args.replay_target == "ee":
                # Get EE action from dataset and convert to joints via IK
                ee_action = action_array_to_ee_action(
                    actions[frame_idx],
                    translation_offset=translation_offset,
                )
                robot_obs = robot.get_observation()
                joint_action = ee_to_joints_processor((ee_action, robot_obs))
            else:  # joint mode
                # Use original joint commands directly
                joint_action = joint_array_to_joint_action(joint_actions[frame_idx], motor_names)

            # Send action to robot
            robot.send_action(joint_action)

            # Log progress periodically
            if frame_idx % 100 == 0:
                logging.info(f"Frame {frame_idx}/{end_frame} ({100 * frame_idx / end_frame:.1f}%)")

            # Wait for next frame
            precise_sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))

        logging.info("Replay complete!")

    finally:
        robot.disconnect()
        logging.info("Robot disconnected.")


if __name__ == "__main__":
    main()
