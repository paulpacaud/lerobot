#!/usr/bin/env python

"""
Convert a LeRobot dataset to the PointAct format.

This script takes a LeRobot dataset (with point clouds added, in joint-space) and
converts it to the PointAct format which includes:
- EE-space state and action (computed via FK)
- Separate joint_state, ee_state, and gripper_state
- Resized images (256x256)
- Renamed point cloud key
- Removal of depth images
- Auto-trimming of idle frames at episode start/end

Input format (joint space):
    observation.state: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
    action: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
    observation.images.front: video
    observation.images.front_depth: video (removed)
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

    # Idle frame trimming
    trim_idle_frames: bool = True  # Enable trimming of idle frames at episode start/end
    trim_threshold_factor: float = 0.1  # Threshold = median(deltas) * this factor

    # Data keys (input)
    state_key: str = "observation.state"
    action_key: str = "action"
    rgb_key: str = "observation.images.front"
    depth_key: str = "observation.images.front_depth"
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


def save_episodes_v21(root: Path, episodes: list[dict]) -> None:
    """Save episodes metadata to v2.1 jsonl format."""
    import jsonlines

    with jsonlines.open(root / "meta" / "episodes.jsonl", mode="w") as writer:
        for ep in episodes:
            writer.write(ep)


def compute_trim_boundaries(
    states: np.ndarray, threshold_factor: float = 0.1
) -> tuple[int, int, float, list[float]]:
    """
    Compute trim boundaries based on state deltas.

    Only trims from the beginning and end of the episode. All frames between
    the first and last active frames are kept regardless of their delta values.

    Args:
        states: Array of shape (N, state_dim) with state values
        threshold_factor: Threshold = median(deltas) * threshold_factor

    Returns:
        Tuple of:
            - first_active: First frame index with significant movement
            - last_active: Last frame index with significant movement
            - threshold: The computed threshold value
            - deltas: List of delta norms for each frame
    """
    if len(states) < 2:
        return 0, len(states) - 1, 0.0, [0.0]

    # Compute state deltas (norm of difference between consecutive frames)
    deltas = []
    for i in range(len(states) - 1):
        delta = np.linalg.norm(states[i + 1] - states[i])
        deltas.append(delta)
    # Last frame has no delta, use 0
    deltas.append(0.0)

    # Compute threshold from median of non-zero deltas
    non_zero_deltas = [d for d in deltas if d > 1e-8]
    if len(non_zero_deltas) == 0:
        # No movement at all, keep all frames
        return 0, len(states) - 1, 0.0, deltas

    median_delta = np.median(non_zero_deltas)
    threshold = median_delta * threshold_factor

    # Find first active frame (scanning from start)
    first_active = 0
    for i in range(len(deltas)):
        if deltas[i] > threshold:
            first_active = i
            break

    # Find last active frame (scanning from end)
    last_active = len(states) - 1
    for i in range(len(deltas) - 1, -1, -1):
        if deltas[i] > threshold:
            # The last active frame is the one AFTER the last significant delta
            last_active = min(i + 1, len(states) - 1)
            break

    # Ensure valid range
    if first_active > last_active:
        first_active = 0
        last_active = len(states) - 1

    return first_active, last_active, threshold, deltas


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


def resize_and_trim_video_file(
    input_path: Path,
    output_path: Path,
    target_size: int,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> int:
    """
    Resize and optionally trim video file.

    Args:
        input_path: Path to input video
        output_path: Path to output video
        target_size: Target square size
        start_frame: First frame to include (0-indexed)
        end_frame: Last frame to include (exclusive), None for all frames

    Returns:
        Number of frames written
    """
    import av
    from fractions import Fraction

    # Read input video
    frames = []
    with av.open(str(input_path)) as container:
        stream = container.streams.video[0]
        original_fps = stream.average_rate  # Keep as Fraction
        for frame_idx, frame in enumerate(container.decode(video=0)):
            if frame_idx < start_frame:
                continue
            if end_frame is not None and frame_idx >= end_frame:
                break
            img = frame.to_ndarray(format="rgb24")
            img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            frames.append(img_resized)

    if len(frames) == 0:
        logging.warning(f"No frames to write for {output_path}")
        return 0

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

    return len(frames)


def trim_lmdb_point_clouds(
    lmdb_path: Path,
    episode_trims: dict[int, tuple[int, int, int]],
) -> None:
    """
    Trim LMDB point clouds by removing entries for trimmed frames and reindexing.

    Args:
        lmdb_path: Path to the LMDB directory
        episode_trims: Dict mapping episode_index to (trim_start, trim_end, original_length)
                       where trim_start is the new first frame index in original indexing,
                       and trim_end is the new last frame index (exclusive) in original indexing
    """
    if not lmdb_path.exists():
        logging.info("No LMDB point clouds to trim")
        return

    logging.info("Trimming LMDB point clouds...")

    # Create temporary LMDB for output
    temp_lmdb_path = lmdb_path.parent / "point_clouds_temp"
    if temp_lmdb_path.exists():
        shutil.rmtree(temp_lmdb_path)

    # Open source LMDB
    src_env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    dst_env = lmdb.open(str(temp_lmdb_path), map_size=src_env.info()["map_size"])

    entries_copied = 0
    entries_removed = 0

    with src_env.begin() as src_txn:
        with dst_env.begin(write=True) as dst_txn:
            cursor = src_txn.cursor()
            for key, value in cursor:
                key_str = key.decode("ascii")
                # Parse key format: "episode_idx-frame_idx"
                parts = key_str.split("-")
                if len(parts) != 2:
                    # Unknown key format, keep it
                    dst_txn.put(key, value)
                    entries_copied += 1
                    continue

                ep_idx = int(parts[0])
                frame_idx = int(parts[1])

                if ep_idx not in episode_trims:
                    # Episode not trimmed, keep all frames
                    dst_txn.put(key, value)
                    entries_copied += 1
                    continue

                trim_start, trim_end, original_length = episode_trims[ep_idx]

                if frame_idx < trim_start or frame_idx >= trim_end:
                    # Frame was trimmed
                    entries_removed += 1
                    continue

                # Reindex frame: new_frame_idx = frame_idx - trim_start
                new_frame_idx = frame_idx - trim_start
                new_key = f"{ep_idx}-{new_frame_idx}".encode("ascii")
                dst_txn.put(new_key, value)
                entries_copied += 1

    src_env.close()
    dst_env.close()

    # Replace original with trimmed version
    shutil.rmtree(lmdb_path)
    shutil.move(str(temp_lmdb_path), str(lmdb_path))

    logging.info(f"LMDB trimming complete: {entries_copied} entries kept, {entries_removed} removed")


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
    trim_idle_frames: bool = True,
    trim_threshold_factor: float = 0.1,
    state_key: str = "observation.state",
    action_key: str = "action",
    rgb_key: str = "observation.images.front",
    depth_key: str = "observation.images.front_depth",
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
        trim_idle_frames: Enable trimming of idle frames at episode start/end
        trim_threshold_factor: Threshold = median(deltas) * this factor
        state_key: Input state key
        action_key: Input action key
        rgb_key: Input RGB image key
        depth_key: Input depth image key (will be removed)
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
    logging.info(f"Trim idle frames: {trim_idle_frames}")
    if trim_idle_frames:
        logging.info(f"Trim threshold factor: {trim_threshold_factor}")

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

    # Remove depth video directory if it exists
    logging.info("Removing depth images...")
    videos_dir = dataset_path / "videos"
    if videos_dir.exists():
        for chunk_dir in sorted(videos_dir.glob("chunk-*")):
            depth_video_dir = chunk_dir / depth_key
            if depth_video_dir.exists():
                logging.info(f"Removing {depth_video_dir}")
                shutil.rmtree(depth_video_dir)

    # Remove depth feature from info.json
    if depth_key in info.get("features", {}):
        logging.info(f"Removing feature: {depth_key}")
        del info["features"][depth_key]

    # Get parquet files
    parquet_files = get_parquet_files(dataset_path)
    logging.info(f"Found {len(parquet_files)} parquet files")

    # Load all episodes metadata
    episodes = load_episodes_v21(dataset_path)
    logging.info(f"Found {len(episodes)} episodes")

    # First pass: Load all data, compute FK, and determine trim boundaries per episode
    logging.info("First pass: Computing FK and trim boundaries...")

    # Dict to store trim info: {episode_idx: (trim_start, trim_end, original_length)}
    episode_trims: dict[int, tuple[int, int, int]] = {}

    # Dict to store episode data: {episode_idx: DataFrame}
    episode_data: dict[int, pd.DataFrame] = {}

    # Load all parquet files
    all_data = []
    for parquet_path in parquet_files:
        df = pd.read_parquet(parquet_path)
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)

    # Group by episode
    for episode_idx, ep_df in combined_df.groupby("episode_index"):
        ep_df = ep_df.sort_values("frame_index").reset_index(drop=True)
        episode_data[episode_idx] = ep_df

        # Compute FK for all frames
        new_states = []
        new_actions = []
        ee_states = []
        joint_states = []
        gripper_states = []

        for _, row in ep_df.iterrows():
            state_joints = np.array(row[state_key], dtype=np.float32)
            action_joints = np.array(row[action_key], dtype=np.float32)

            # Compute EE poses
            ee_pose, state_ee = joints_to_ee(state_joints, kinematics, Rotation, translation_offset)
            _, action_ee = joints_to_ee(action_joints, kinematics, Rotation, translation_offset)

            new_states.append(state_ee)
            new_actions.append(action_ee)
            ee_states.append(ee_pose)
            joint_states.append(state_joints.astype(np.float32))
            gripper_states.append(np.array([state_joints[5]], dtype=np.float32))

        # Update episode dataframe with FK results
        ep_df[state_key] = new_states
        ep_df[action_key] = new_actions
        ep_df[output_ee_state_key] = ee_states
        ep_df[output_joint_state_key] = joint_states
        ep_df[output_gripper_state_key] = gripper_states
        episode_data[episode_idx] = ep_df

        # Compute trim boundaries if enabled
        original_length = len(ep_df)
        if trim_idle_frames:
            states_array = np.array(new_states)
            trim_start, trim_end, threshold, deltas = compute_trim_boundaries(
                states_array, trim_threshold_factor
            )
            # trim_end is the index of the last frame to keep, we need exclusive end
            trim_end_exclusive = trim_end + 1
            episode_trims[episode_idx] = (trim_start, trim_end_exclusive, original_length)
        else:
            episode_trims[episode_idx] = (0, original_length, original_length)

    # Print trim report
    if trim_idle_frames:
        logging.info("")
        logging.info("=== Idle Frame Trimming Report ===")
        total_original = 0
        total_trimmed = 0
        boundary_deltas = []

        for episode_idx in sorted(episode_trims.keys()):
            trim_start, trim_end, original_length = episode_trims[episode_idx]
            new_length = trim_end - trim_start
            frames_from_start = trim_start
            frames_from_end = original_length - trim_end
            total_original += original_length
            total_trimmed += new_length

            logging.info(
                f"Episode {episode_idx}: {original_length} -> {new_length} frames "
                f"(trimmed {frames_from_start} from start, {frames_from_end} from end)"
            )

            # Compute boundary deltas for verification
            if episode_idx in episode_data:
                ep_df = episode_data[episode_idx]
                states_array = np.array(ep_df[state_key].tolist())
                if trim_start < len(states_array) - 1:
                    start_delta = np.linalg.norm(states_array[trim_start + 1] - states_array[trim_start])
                    boundary_deltas.append(start_delta)
                if trim_end - 1 > 0 and trim_end - 1 < len(states_array):
                    end_delta = np.linalg.norm(
                        states_array[trim_end - 1] - states_array[max(0, trim_end - 2)]
                    )
                    boundary_deltas.append(end_delta)

        frames_removed = total_original - total_trimmed
        pct_trimmed = 100.0 * frames_removed / total_original if total_original > 0 else 0.0
        logging.info("")
        logging.info(f"Total: {total_original} -> {total_trimmed} frames ({pct_trimmed:.1f}% trimmed)")

        if boundary_deltas:
            logging.info(
                f"Boundary deltas after trim: min={min(boundary_deltas):.6f}, "
                f"max={max(boundary_deltas):.6f}, mean={np.mean(boundary_deltas):.6f}"
            )
        logging.info("")

    # Second pass: Apply trimming and write parquet files
    logging.info("Second pass: Writing trimmed parquet files...")

    total_frames_written = 0
    for parquet_path in tqdm(parquet_files, desc="Writing parquet files"):
        # Determine which episode this parquet file belongs to
        # For v2.1, each parquet is typically episode_XXXXXX.parquet
        parquet_name = parquet_path.stem
        if parquet_name.startswith("episode_"):
            ep_idx = int(parquet_name.split("_")[1])
        else:
            # Try to infer from the data
            original_df = pd.read_parquet(parquet_path)
            if "episode_index" in original_df.columns:
                ep_idx = original_df["episode_index"].iloc[0]
            else:
                logging.warning(f"Could not determine episode index for {parquet_path}, skipping trim")
                continue

        if ep_idx not in episode_data:
            logging.warning(f"Episode {ep_idx} not found in episode_data, skipping")
            continue

        ep_df = episode_data[ep_idx]
        trim_start, trim_end, original_length = episode_trims[ep_idx]

        # Apply trim
        trimmed_df = ep_df.iloc[trim_start:trim_end].copy()

        # Update frame_index to be 0-indexed from new start
        trimmed_df["frame_index"] = range(len(trimmed_df))

        # Update timestamp if present
        if "timestamp" in trimmed_df.columns:
            fps = info.get("fps", 30)
            trimmed_df["timestamp"] = [i / fps for i in range(len(trimmed_df))]

        total_frames_written += len(trimmed_df)

        # Write back to parquet
        table = pa.Table.from_pandas(trimmed_df, preserve_index=False)
        pq.write_table(table, parquet_path)

    logging.info(f"Written {total_frames_written} frames total")

    # Update episode metadata
    logging.info("Updating episode metadata...")
    for ep in episodes:
        ep_idx = ep["episode_index"]
        if ep_idx in episode_trims:
            trim_start, trim_end, original_length = episode_trims[ep_idx]
            ep["length"] = trim_end - trim_start

    save_episodes_v21(dataset_path, episodes)

    # Resize and trim videos
    logging.info("Resizing and trimming videos...")
    if videos_dir.exists():
        # Find all video files for the RGB key
        for chunk_dir in sorted(videos_dir.glob("chunk-*")):
            rgb_video_dir = chunk_dir / rgb_key
            if rgb_video_dir.exists():
                # Rename to new key
                new_video_dir = chunk_dir / output_image_key
                for video_file in sorted(rgb_video_dir.glob("*.mp4")):
                    # Parse episode index from video filename
                    video_name = video_file.stem
                    if video_name.startswith("episode_"):
                        ep_idx = int(video_name.split("_")[1])
                    else:
                        ep_idx = None

                    new_video_file = new_video_dir / video_file.name

                    if ep_idx is not None and ep_idx in episode_trims:
                        trim_start, trim_end, original_length = episode_trims[ep_idx]
                        logging.info(
                            f"Resizing and trimming {video_file.name} "
                            f"(frames {trim_start}-{trim_end} of {original_length})..."
                        )
                        resize_and_trim_video_file(
                            video_file, new_video_file, image_size, trim_start, trim_end
                        )
                    else:
                        logging.info(f"Resizing {video_file.name}...")
                        resize_and_trim_video_file(video_file, new_video_file, image_size)

                # Remove old videos
                shutil.rmtree(rgb_video_dir)

    # Trim LMDB point clouds if they exist
    lmdb_path = dataset_path / "point_clouds"
    if lmdb_path.exists() and trim_idle_frames:
        trim_lmdb_point_clouds(lmdb_path, episode_trims)

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
        info["features"].pop(rgb_key)
        info["features"][output_image_key] = {
            "dtype": "video",
            "shape": [image_size, image_size, 3],
            "names": ["height", "width", "rgb"],
        }

    # Update point cloud feature
    if point_cloud_key in info["features"]:
        pcd_feature = info["features"].pop(point_cloud_key)
        info["features"][output_point_cloud_key] = pcd_feature

    # Update total frames
    info["total_frames"] = total_frames_written

    # Add conversion metadata
    if "conversion_info" not in info:
        info["conversion_info"] = {}
    info["conversion_info"]["pointact_conversion"] = {
        "urdf_file": str(urdf_path.name),
        "target_frame": target_frame,
        "joint_names": joint_names,
        "translation_offset": [tx, ty, tz],
        "image_size": image_size,
        "trim_idle_frames": trim_idle_frames,
        "trim_threshold_factor": trim_threshold_factor if trim_idle_frames else None,
    }

    save_info(dataset_path, info)

    logging.info("Conversion to PointAct format complete!")
    logging.info("Output features:")
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
        trim_idle_frames=args.trim_idle_frames,
        trim_threshold_factor=args.trim_threshold_factor,
        state_key=args.state_key,
        action_key=args.action_key,
        rgb_key=args.rgb_key,
        depth_key=args.depth_key,
        point_cloud_key=args.point_cloud_key,
        output_image_key=args.output_image_key,
        output_point_cloud_key=args.output_point_cloud_key,
        output_ee_state_key=args.output_ee_state_key,
        output_joint_state_key=args.output_joint_state_key,
        output_gripper_state_key=args.output_gripper_state_key,
    )
