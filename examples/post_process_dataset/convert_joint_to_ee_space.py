#!/usr/bin/env python

"""
This script converts joint-space state and action data to end-effector (Cartesian) space.

It uses forward kinematics to compute the end-effector pose (position + orientation)
from joint positions. The orientation is represented as a rotation vector (axis-angle).

Input format (joint space):
    observation.state: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
    action: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]

Output format (EE space):
    observation.state: [x, y, z, rx, ry, rz, gripper]
    action: [x, y, z, rx, ry, rz, gripper]

Where (rx, ry, rz) is a rotation vector (axis-angle representation).

Supports both v2.1 and v3.0 LeRobot dataset formats.

Usage:
```bash
python examples/post_process_dataset/convert_joint_to_ee_space.py \
    --dataset_dir=$HOME/lerobot_datasets/depth_test \
    --urdf_path=./examples/post_process_dataset/constants/SO101/so101_new_calib.urdf
./examples/post_process_dataset/constants/SO101/so101_new_calib.urdf
# Specify output directory to create a new dataset instead of modifying in place
python examples/post_process_dataset/convert_joint_to_ee_space.py \
    --dataset_dir=$HOME/lerobot_datasets/depth_test \
    --output_dir=$HOME/lerobot_datasets/depth_test_ee \
    --urdf_path=./examples/post_process_dataset/constants/SO101/so101_new_calib.urdf
```

Requirements:
    - placo library: pip install placo
    - URDF file for your robot (SO100/SO101)
"""

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tap import Tap
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Args(Tap):
    """Arguments for converting joint-space to end-effector space."""

    # Required arguments
    dataset_dir: str  # Path to the LeRobot dataset directory
    urdf_path: str  # Path to the robot URDF file

    # Optional arguments
    output_dir: str | None = None  # Output directory (if None, modifies dataset in place)
    target_frame: str = "gripper"  # Name of the end-effector frame in the URDF (use 'jaw' for SO100)

    # Joint configuration
    joint_names: list[str] = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ]  # Joint names for FK (excluding gripper)

    # Keys in the dataset
    state_key: str = "observation.state"  # Key for state in parquet
    action_key: str = "action"  # Key for action in parquet


DEFAULT_CHUNK_SIZE = 1000


def load_info(root: Path) -> dict:
    """Load info.json from dataset root."""
    with open(root / "meta" / "info.json") as f:
        return json.load(f)


def save_info(root: Path, info: dict) -> None:
    """Save info.json to dataset root."""
    with open(root / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)


def load_episodes_v21(root: Path) -> list[dict]:
    """Load episodes metadata from v2.1 jsonl format."""
    import jsonlines
    episodes = []
    with jsonlines.open(root / "meta" / "episodes.jsonl") as reader:
        for ep in reader:
            episodes.append(ep)
    return sorted(episodes, key=lambda x: x["episode_index"])


def joints_to_ee(
    joint_values: np.ndarray,
    kinematics,
    rotation_class,
) -> np.ndarray:
    """
    Convert joint positions to end-effector pose.

    Args:
        joint_values: Array of shape (6,) with joint positions in degrees
                     [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        kinematics: RobotKinematics instance
        rotation_class: Rotation class for converting rotation matrix to rotation vector

    Returns:
        Array of shape (7,) with EE pose [x, y, z, rx, ry, rz, gripper]
    """
    # Extract arm joints (first 5) and gripper (last 1)
    # Convert to float64 for placo compatibility
    arm_joints = joint_values[:5].astype(np.float64)
    gripper_pos = float(joint_values[5])

    # Compute forward kinematics
    T = kinematics.forward_kinematics(arm_joints)

    # Extract position (xyz in meters)
    position = T[:3, 3]

    # Extract orientation as rotation vector (axis-angle)
    rotation_vec = rotation_class.from_matrix(T[:3, :3]).as_rotvec()

    # Combine: [x, y, z, rx, ry, rz, gripper]
    ee_pose = np.concatenate([position, rotation_vec, [gripper_pos]])

    return ee_pose.astype(np.float32)


def get_parquet_files(dataset_path: Path, info: dict) -> list[Path]:
    """Get all parquet data files in the dataset."""
    data_dir = dataset_path / "data"
    parquet_files = []

    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        for parquet_file in sorted(chunk_dir.glob("*.parquet")):
            parquet_files.append(parquet_file)

    return parquet_files


def convert_dataset_to_ee_space(
    dataset_dir: str,
    urdf_path: str,
    output_dir: str | None = None,
    target_frame: str = "gripper",
    joint_names: list[str] | None = None,
    state_key: str = "observation.state",
    action_key: str = "action",
) -> None:
    """
    Convert a LeRobot dataset from joint-space to end-effector space.

    Supports both v2.1 and v3.0 LeRobot dataset formats.

    Args:
        dataset_dir: Path to the input dataset
        urdf_path: Path to the robot URDF file
        output_dir: Path for output dataset (if None, modifies in place)
        target_frame: Name of the end-effector frame in the URDF
        joint_names: List of joint names for FK (excluding gripper)
        state_key: Key for state data in parquet files
        action_key: Key for action data in parquet files
    """
    # Import kinematics (requires placo)
    try:
        from lerobot.model.kinematics import RobotKinematics
        from lerobot.utils.rotation import Rotation
    except ImportError as e:
        raise ImportError(
            "Failed to import kinematics modules. Make sure placo is installed: pip install placo"
        ) from e

    if joint_names is None:
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

    dataset_path = Path(dataset_dir)
    urdf_path = Path(urdf_path)

    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    # Initialize kinematics solver
    logging.info(f"Loading URDF from: {urdf_path}")
    logging.info(f"Target frame: {target_frame}")
    logging.info(f"Joint names: {joint_names}")

    kinematics = RobotKinematics(
        urdf_path=str(urdf_path),
        target_frame_name=target_frame,
        joint_names=joint_names,
    )

    # Handle output directory
    if output_dir is not None:
        output_path = Path(output_dir)
        if output_path.exists():
            logging.warning(f"Output directory exists, removing: {output_path}")
            shutil.rmtree(output_path)
        logging.info(f"Copying dataset to: {output_path}")
        shutil.copytree(dataset_path, output_path)
        dataset_path = output_path

    # Load dataset info
    info = load_info(dataset_path)
    version = info.get("codebase_version", "v2.1")

    logging.info(f"Dataset version: {version}")
    logging.info(f"Total frames: {info.get('total_frames', 'unknown')}")

    # Get all parquet files
    parquet_files = get_parquet_files(dataset_path, info)
    logging.info(f"Found {len(parquet_files)} parquet files")

    # Process each parquet file
    total_frames_processed = 0
    for parquet_path in tqdm(parquet_files, desc="Converting parquet files"):
        # Read parquet file
        df = pd.read_parquet(parquet_path)

        # Convert state and action for each frame
        new_states = []
        new_actions = []

        for _, row in df.iterrows():
            # Get current joint state and action
            state_joints = np.array(row[state_key], dtype=np.float32)
            action_joints = np.array(row[action_key], dtype=np.float32)

            # Convert to EE space
            state_ee = joints_to_ee(state_joints, kinematics, Rotation)
            action_ee = joints_to_ee(action_joints, kinematics, Rotation)

            new_states.append(state_ee)
            new_actions.append(action_ee)

        total_frames_processed += len(df)

        # Update dataframe
        df[state_key] = new_states
        df[action_key] = new_actions

        # Write back to parquet
        table = pa.Table.from_pandas(df)
        pq.write_table(table, parquet_path)

    logging.info(f"Processed {total_frames_processed} frames")

    # Update info.json with new feature definitions
    ee_names = ["x", "y", "z", "rx", "ry", "rz", "gripper"]

    info["features"][state_key] = {
        "dtype": "float32",
        "shape": [7],
        "names": ee_names,
    }

    info["features"][action_key] = {
        "dtype": "float32",
        "shape": [7],
        "names": ee_names,
    }

    # Add metadata about the conversion
    if "conversion_info" not in info:
        info["conversion_info"] = {}
    info["conversion_info"]["joint_to_ee"] = {
        "urdf_file": str(urdf_path.name),
        "target_frame": target_frame,
        "joint_names": joint_names,
        "representation": "position_xyz_rotation_vector",
    }

    save_info(dataset_path, info)

    logging.info("Conversion complete!")
    logging.info(f"Updated features:")
    logging.info(f"  {state_key}: shape [7], names {ee_names}")
    logging.info(f"  {action_key}: shape [7], names {ee_names}")


if __name__ == "__main__":
    args = Args().parse_args()

    convert_dataset_to_ee_space(
        dataset_dir=args.dataset_dir,
        urdf_path=args.urdf_path,
        output_dir=args.output_dir,
        target_frame=args.target_frame,
        joint_names=args.joint_names,
        state_key=args.state_key,
        action_key=args.action_key,
    )
