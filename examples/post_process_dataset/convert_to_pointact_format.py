#!/usr/bin/env python

"""
Convert a LeRobot dataset to the PointAct format.

This script takes a LeRobot dataset (with point clouds added, in joint-space) and
converts it to the PointAct format which includes:
- EE-space state and action (computed via FK)
- Separate joint_state, ee_state, and gripper_state
- Resized images (256x256)
- Renamed point cloud key

Input format (joint space):
    observation.state: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
    action: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
    observation.images.front: video
    observation.point_cloud: LMDB point clouds

Output format (PointAct):
    observation.state: [x, y, z, axis_angle1, axis_angle2, axis_angle3, gripper_openness]
    observation.states.ee_state: [x, y, z, axis_angle1, axis_angle2, axis_angle3]
    observation.states.joint_state: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
    observation.states.gripper_state: [gripper_openness]
    action: [x, y, z, axis_angle1, axis_angle2, axis_angle3, gripper_openness]
    observation.images.front_image: video (256, 256, 3)
    observation.points.frontview: point cloud

Usage:
```bash
python examples/post_process_dataset/convert_to_pointact_format.py --dataset_dir=$HOME/lerobot_datasets/depth_test_v2 --output_dir=$HOME/lerobot_datasets/depth_test_pointact --urdf_path=./examples/post_process_dataset/constants/SO101/so101_new_calib.urdf --tx=-0.28 --ty=0.03 --tz=0.05
```
"""

import io
import json
import logging
import shutil
from pathlib import Path

import cv2
import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tap import Tap
from tqdm import tqdm

msgpack_numpy.patch()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Args(Tap):
    """Arguments for converting dataset to PointAct format."""

    # Required arguments
    dataset_dir: str  # Path to the LeRobot dataset directory (with point clouds, joint-space)
    urdf_path: str  # Path to the robot URDF file

    # Optional arguments
    output_dir: str | None = None  # Output directory (if None, modifies dataset in place)
    target_frame: str = "gripper_frame_link"  # Name of the EE frame in URDF

    # Robot-to-world translation offset
    tx: float = 0.0  # X translation offset (meters)
    ty: float = 0.0  # Y translation offset (meters)
    tz: float = 0.0  # Z translation offset (meters)

    # Joint configuration
    joint_names: list[str] = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ]

    # Image resize
    image_size: int = 256  # Target image size (square)

    # Data keys (input)
    state_key: str = "observation.state"
    action_key: str = "action"
    rgb_key: str = "observation.images.front"
    point_cloud_key: str = "observation.point_cloud"

    # Data keys (output)
    output_image_key: str = "observation.images.front_image"
    output_point_cloud_key: str = "observation.points.frontview"
    output_ee_state_key: str = "observation.states.ee_state"
    output_joint_state_key: str = "observation.states.joint_state"
    output_gripper_state_key: str = "observation.states.gripper_state"


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
    translation_offset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert joint positions to end-effector pose.

    Args:
        joint_values: Array of shape (6,) with joint positions
                     [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        kinematics: RobotKinematics instance
        rotation_class: Rotation class for converting rotation matrix to rotation vector
        translation_offset: Optional (3,) array with [tx, ty, tz]

    Returns:
        Tuple of:
            - ee_pose: Array of shape (6,) with [x, y, z, rx, ry, rz]
            - ee_pose_with_gripper: Array of shape (7,) with [x, y, z, rx, ry, rz, gripper]
    """
    arm_joints = joint_values[:5].astype(np.float64)
    gripper_pos = float(joint_values[5])

    T = kinematics.forward_kinematics(arm_joints)
    position = T[:3, 3]

    if translation_offset is not None:
        position = position + translation_offset

    rotation_vec = rotation_class.from_matrix(T[:3, :3]).as_rotvec()

    ee_pose = np.concatenate([position, rotation_vec]).astype(np.float32)
    ee_pose_with_gripper = np.concatenate([ee_pose, [gripper_pos]]).astype(np.float32)

    return ee_pose, ee_pose_with_gripper


def get_parquet_files(dataset_path: Path) -> list[Path]:
    """Get all parquet data files in the dataset."""
    data_dir = dataset_path / "data"
    parquet_files = []

    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        for parquet_file in sorted(chunk_dir.glob("*.parquet")):
            parquet_files.append(parquet_file)

    return parquet_files


def resize_image_bytes(image_bytes: bytes, target_size: int) -> bytes:
    """Resize image bytes to target square size."""
    img = Image.open(io.BytesIO(image_bytes))
    img_resized = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img_resized.save(buffer, format="PNG")
    return buffer.getvalue()


def resize_video_file(input_path: Path, output_path: Path, target_size: int) -> None:
    """Resize video file to target square size using ffmpeg via cv2."""
    import av
    from fractions import Fraction

    # Read input video
    frames = []
    with av.open(str(input_path)) as container:
        stream = container.streams.video[0]
        original_fps = stream.average_rate  # Keep as Fraction
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="rgb24")
            img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            frames.append(img_resized)

    # Write output video
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(output_path), mode="w") as output_container:
        output_stream = output_container.add_stream("libx264", rate=Fraction(original_fps))
        output_stream.width = target_size
        output_stream.height = target_size
        output_stream.pix_fmt = "yuv420p"

        for img in frames:
            av_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in output_stream.encode(av_frame):
                output_container.mux(packet)

        for packet in output_stream.encode():
            output_container.mux(packet)


def convert_to_pointact_format(
    dataset_dir: str,
    urdf_path: str,
    output_dir: str | None = None,
    target_frame: str = "gripper_frame_link",
    joint_names: list[str] | None = None,
    tx: float = 0.0,
    ty: float = 0.0,
    tz: float = 0.0,
    image_size: int = 256,
    state_key: str = "observation.state",
    action_key: str = "action",
    rgb_key: str = "observation.images.front",
    point_cloud_key: str = "observation.point_cloud",
    output_image_key: str = "observation.images.front_image",
    output_point_cloud_key: str = "observation.points.frontview",
    output_ee_state_key: str = "observation.states.ee_state",
    output_joint_state_key: str = "observation.states.joint_state",
    output_gripper_state_key: str = "observation.states.gripper_state",
) -> None:
    """
    Convert a LeRobot dataset to PointAct format.

    Args:
        dataset_dir: Path to input dataset
        urdf_path: Path to robot URDF file
        output_dir: Path for output dataset
        target_frame: Name of the EE frame in URDF
        joint_names: List of joint names for FK
        tx, ty, tz: Translation offset (robot to world frame)
        image_size: Target image size (square)
        state_key: Input state key
        action_key: Input action key
        rgb_key: Input RGB image key
        point_cloud_key: Input point cloud key
        output_*: Output key names
    """
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

    logging.info(f"Loading URDF from: {urdf_path}")
    logging.info(f"Target frame: {target_frame}")
    logging.info(f"Joint names: {joint_names}")
    logging.info(f"Translation offset: [{tx}, {ty}, {tz}]")
    logging.info(f"Target image size: {image_size}x{image_size}")

    translation_offset = None
    if not (tx == 0.0 and ty == 0.0 and tz == 0.0):
        translation_offset = np.array([tx, ty, tz], dtype=np.float64)

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
    else:
        output_path = dataset_path

    info = load_info(dataset_path)
    version = info.get("codebase_version", "v2.1")

    logging.info(f"Dataset version: {version}")
    logging.info(f"Total frames: {info.get('total_frames', 'unknown')}")

    # Get parquet files
    parquet_files = get_parquet_files(dataset_path)
    logging.info(f"Found {len(parquet_files)} parquet files")

    # Process each parquet file
    total_frames_processed = 0
    for parquet_path in tqdm(parquet_files, desc="Converting parquet files"):
        df = pd.read_parquet(parquet_path)

        new_states = []  # [x, y, z, rx, ry, rz, gripper]
        new_actions = []  # [x, y, z, rx, ry, rz, gripper]
        ee_states = []  # [x, y, z, rx, ry, rz]
        joint_states = []  # [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]
        gripper_states = []  # [gripper]

        for _, row in df.iterrows():
            state_joints = np.array(row[state_key], dtype=np.float32)
            action_joints = np.array(row[action_key], dtype=np.float32)

            # Compute EE poses
            ee_pose, state_ee = joints_to_ee(state_joints, kinematics, Rotation, translation_offset)
            _, action_ee = joints_to_ee(action_joints, kinematics, Rotation, translation_offset)

            new_states.append(state_ee)
            new_actions.append(action_ee)
            ee_states.append(ee_pose)
            joint_states.append(state_joints.astype(np.float32))  # All 6 joints including gripper
            gripper_states.append(np.array([state_joints[5]], dtype=np.float32))

        total_frames_processed += len(df)

        # Update dataframe
        df[state_key] = new_states
        df[action_key] = new_actions
        df[output_ee_state_key] = ee_states
        df[output_joint_state_key] = joint_states
        df[output_gripper_state_key] = gripper_states

        # Write back to parquet
        table = pa.Table.from_pandas(df)
        pq.write_table(table, parquet_path)

    logging.info(f"Processed {total_frames_processed} frames")

    # Resize videos
    logging.info("Resizing videos...")
    videos_dir = dataset_path / "videos"
    if videos_dir.exists():
        # Find all video files for the RGB key
        for chunk_dir in sorted(videos_dir.glob("chunk-*")):
            rgb_video_dir = chunk_dir / rgb_key
            if rgb_video_dir.exists():
                # Rename to new key
                new_video_dir = chunk_dir / output_image_key
                for video_file in sorted(rgb_video_dir.glob("*.mp4")):
                    new_video_file = new_video_dir / video_file.name
                    logging.info(f"Resizing {video_file.name}...")
                    resize_video_file(video_file, new_video_file, image_size)
                # Remove old videos
                shutil.rmtree(rgb_video_dir)

            # Remove depth videos if they exist
            depth_video_dir = chunk_dir / "observation.images.front_depth"
            if depth_video_dir.exists():
                logging.info(f"Removing depth videos from {chunk_dir.name}...")
                shutil.rmtree(depth_video_dir)

    # Rename point cloud LMDB (just rename the directory reference in info.json)
    # The actual LMDB data doesn't need to change

    # Update info.json with new feature definitions
    ee_names = ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3"]
    state_names = ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper_openness"]

    # Update state and action features
    info["features"][state_key] = {
        "dtype": "float32",
        "shape": [7],
        "names": {"motors": state_names},
    }

    info["features"][action_key] = {
        "dtype": "float32",
        "shape": [7],
        "names": {"motors": state_names},
    }

    # Add new state features
    info["features"][output_ee_state_key] = {
        "dtype": "float32",
        "shape": [6],
        "names": {"motors": ee_names},
    }

    all_joint_names = joint_names + ["gripper"]
    info["features"][output_joint_state_key] = {
        "dtype": "float32",
        "shape": [len(all_joint_names)],
        "names": {"motors": all_joint_names},
    }

    info["features"][output_gripper_state_key] = {
        "dtype": "float32",
        "shape": [1],
        "names": {"motors": ["gripper_openness"]},
    }

    # Update image feature
    if rgb_key in info["features"]:
        original_image_feature = info["features"].pop(rgb_key)
        info["features"][output_image_key] = {
            "dtype": "video",
            "shape": [image_size, image_size, 3],
            "names": ["height", "width", "rgb"],
        }

    # Update point cloud feature
    if point_cloud_key in info["features"]:
        pcd_feature = info["features"].pop(point_cloud_key)
        info["features"][output_point_cloud_key] = pcd_feature

    # Remove depth feature (not used in PointAct format)
    depth_key = "observation.images.front_depth"
    if depth_key in info["features"]:
        info["features"].pop(depth_key)
        logging.info(f"Removed feature: {depth_key}")

    # Add conversion metadata
    if "conversion_info" not in info:
        info["conversion_info"] = {}
    info["conversion_info"]["pointact_conversion"] = {
        "urdf_file": str(urdf_path.name),
        "target_frame": target_frame,
        "joint_names": joint_names,
        "translation_offset": [tx, ty, tz],
        "image_size": image_size,
    }

    save_info(dataset_path, info)

    logging.info("Conversion to PointAct format complete!")
    logging.info(f"Output features:")
    logging.info(f"  {state_key}: shape [7], {state_names}")
    logging.info(f"  {action_key}: shape [7], {state_names}")
    logging.info(f"  {output_ee_state_key}: shape [6], {ee_names}")
    logging.info(f"  {output_joint_state_key}: shape [6], {joint_names + ['gripper']}")
    logging.info(f"  {output_gripper_state_key}: shape [1], ['gripper_openness']")
    logging.info(f"  {output_image_key}: shape [{image_size}, {image_size}, 3]")
    logging.info(f"  {output_point_cloud_key}: point cloud data")


if __name__ == "__main__":
    args = Args().parse_args()

    convert_to_pointact_format(
        dataset_dir=args.dataset_dir,
        urdf_path=args.urdf_path,
        output_dir=args.output_dir,
        target_frame=args.target_frame,
        joint_names=args.joint_names,
        tx=args.tx,
        ty=args.ty,
        tz=args.tz,
        image_size=args.image_size,
        state_key=args.state_key,
        action_key=args.action_key,
        rgb_key=args.rgb_key,
        point_cloud_key=args.point_cloud_key,
        output_image_key=args.output_image_key,
        output_point_cloud_key=args.output_point_cloud_key,
        output_ee_state_key=args.output_ee_state_key,
        output_joint_state_key=args.output_joint_state_key,
        output_gripper_state_key=args.output_gripper_state_key,
    )
