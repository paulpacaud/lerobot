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

Trimming explanation:
trim_threshold_factor
Suppose your deltas are: [0.001, 0.05, 0.04, 0.06, 0.001, 0.05, 0.001]
                                ↑                      ↑           ↑
                              idle?                  idle?       idle?

  Median of non-zero deltas ≈ 0.05

  With trim_threshold_factor = 0.1 (default):
    threshold = 0.05 × 0.1 = 0.005

    → 0.001 < 0.005  → IDLE
    → 0.05  > 0.005  → MOVING
    → 0.04  > 0.005  → MOVING

Effect of different values:
  ┌───────────────┬─────────────────┬───────────────────────────────────────────────────────┐
  │     Value     │    Threshold    │                        Effect                         │
  ├───────────────┼─────────────────┼───────────────────────────────────────────────────────┤
  │ 0.05          │ 0.25% of median │ Very strict - only truly stationary frames are "idle" │
  ├───────────────┼─────────────────┼───────────────────────────────────────────────────────┤
  │ 0.1 (default) │ 10% of median   │ Balanced - small movements count as idle              │
  ├───────────────┼─────────────────┼───────────────────────────────────────────────────────┤
  │ 0.3           │ 30% of median   │ Aggressive - slow movements also count as idle        │
  └───────────────┴─────────────────┴───────────────────────────────────────────────────────┘

How it works:

  1. The algorithm computes the movement delta between each consecutive frame
  2. Frames with delta below the threshold are marked as "idle"
  3. It finds contiguous runs of idle frames (idle segments)
  4. Only idle segments with length >= min_idle_segment get trimmed

  Example with min_idle_segment=5 (default):

  Frame:    0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
  Movement: M  M  I  I  I  M  M  I  I  I  I  I  I  M  M  M
                  ─────        ───────────────────
                  3 idle         7 idle frames
                  frames         → TRIMMED (≥5)
                  → KEPT (<5)

  - The 3-frame idle segment (frames 2-4) is kept because 3 < 5
  - The 7-frame idle segment (frames 7-13) is trimmed because 7 ≥ 5

Input format (joint space):
    observation.state: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
    action: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
    observation.images.front: video
    observation.images.front_depth: video (removed)
    observation.point_cloud: LMDB point clouds

Output format (PointAct):
    observation.state: [x, y, z, qw, qx, qy, qz, gripper_openness]
    observation.states.ee_state: [x, y, z, qw, qx, qy, qz]
    observation.states.joint_state: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
    observation.states.gripper_state: [gripper_openness]
    action: [x, y, z, qw, qx, qy, qz, gripper_openness]
    observation.images.front_image: video (256, 256, 3)
    observation.points.frontview: point cloud

Usage:
```bash
python examples/post_process_dataset/convert_to_pointact_format.py --dataset_dir=$HOME/lerobot_datasets/depth_test_v2 --output_dir=$HOME/lerobot_datasets/depth_test_pointact --urdf_path=./examples/post_process_dataset/constants/SO101/so101_new_calib.urdf
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

from examples.post_process_dataset.constants.constants import ROBOT_FRAME

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
    no_trim_idle_frames: bool = False  # Disable trimming of idle frames at episode start/end
    trim_threshold_factor: float = 0.05  # Threshold = median(deltas) * this factor
    min_idle_segment: int = 5  # Minimum consecutive idle frames to consider for removal
    keep_frames_per_idle: int = 1  # Number of frames to keep from each internal idle segment

    # Depth preservation
    include_depth: bool = False  # Preserve depth images in the output dataset (dropped by default)

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
    output_joint_action_key: str = "action.joints"  # Original joint commands for replay


DEFAULT_CHUNK_SIZE = 1000


def load_info(root: Path) -> dict:
    """Load info.json from dataset root."""
    with open(root / "meta" / "info.json") as f:
        return json.load(f)


def load_stats(root: Path) -> dict:
    """Load stats.json from dataset root."""
    stats_path = root / "meta" / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            return json.load(f)
    return {}


def save_stats(root: Path, stats: dict) -> None:
    """Save stats.json to dataset root."""
    with open(root / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)


def compute_array_stats(arr: np.ndarray) -> dict:
    """Compute statistics for a numpy array.

    Returns dict with min, max, mean, std, count, and quantiles (q01, q10, q50, q90, q99).
    """
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    return {
        "min": np.min(arr, axis=0).tolist(),
        "max": np.max(arr, axis=0).tolist(),
        "mean": np.mean(arr, axis=0).tolist(),
        "std": np.std(arr, axis=0).tolist(),
        "count": [len(arr)],
        "q01": np.percentile(arr, 1, axis=0).tolist(),
        "q10": np.percentile(arr, 10, axis=0).tolist(),
        "q50": np.percentile(arr, 50, axis=0).tolist(),
        "q90": np.percentile(arr, 90, axis=0).tolist(),
        "q99": np.percentile(arr, 99, axis=0).tolist(),
    }


def regenerate_stats(
    dataset_path: Path,
    info: dict,
    state_key: str,
    action_key: str,
    rgb_key: str,
    depth_key: str,
    point_cloud_key: str,
    output_image_key: str,
    output_point_cloud_key: str,
    output_ee_state_key: str,
    output_joint_state_key: str,
    output_gripper_state_key: str,
    output_joint_action_key: str,
    include_depth: bool = False,
) -> None:
    """Regenerate stats.json to match the new feature definitions after conversion.

    This function:
    1. Loads existing stats
    2. Renames keys for features that were renamed but have compatible stats
    3. Computes new stats from parquet files for transformed features
    4. Removes stats for deleted features
    5. Saves the updated stats
    """
    logging.info("Regenerating dataset statistics...")

    # Load existing stats
    stats = load_stats(dataset_path)

    # Remove stats for deleted features
    if not include_depth and depth_key in stats:
        del stats[depth_key]
        logging.info(f"  Removed stats for deleted feature: {depth_key}")

    # Rename keys for features that were renamed
    if rgb_key in stats and rgb_key != output_image_key:
        stats[output_image_key] = stats.pop(rgb_key)
        logging.info(f"  Renamed stats key: {rgb_key} -> {output_image_key}")

    if point_cloud_key in stats and point_cloud_key != output_point_cloud_key:
        stats[output_point_cloud_key] = stats.pop(point_cloud_key)
        logging.info(f"  Renamed stats key: {point_cloud_key} -> {output_point_cloud_key}")

    # Remove old metadata stats so they get recomputed with correct counts
    METADATA_FIELDS = ["index", "timestamp", "episode_index", "frame_index", "task_index"]
    for field in METADATA_FIELDS:
        if field in stats:
            del stats[field]
            logging.info(f"  Removing old stats for metadata field: {field} (will recompute)")

    # Collect all data from parquet files for computing new stats
    logging.info("  Reading parquet files to compute new stats...")
    data_dir = dataset_path / "data"
    parquet_files = list(data_dir.glob("**/*.parquet"))

    # Collect arrays for each feature that needs recomputation
    feature_data = {
        state_key: [],
        action_key: [],
        output_ee_state_key: [],
        output_joint_state_key: [],
        output_gripper_state_key: [],
        output_joint_action_key: [],
        "index": [],
        "timestamp": [],
        "episode_index": [],
        "frame_index": [],
        "task_index": [],
    }

    for parquet_path in tqdm(parquet_files, desc="  Reading parquet files"):
        df = pd.read_parquet(parquet_path)

        for key in feature_data:
            if key in df.columns:
                # Convert list-of-arrays column to numpy array
                values = np.array(df[key].tolist())
                feature_data[key].append(values)

    # Compute stats for each feature
    for key, data_list in feature_data.items():
        if data_list:
            all_data = np.concatenate(data_list, axis=0)
            stats[key] = compute_array_stats(all_data)
            logging.info(f"  Computed stats for: {key} (shape: {all_data.shape})")

    # Save updated stats
    save_stats(dataset_path, stats)
    logging.info("Stats regeneration complete!")


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


def load_episodes_v30(root: Path) -> list[dict]:
    """Load episodes metadata from v3.0 parquet format."""
    episodes_dir = root / "meta" / "episodes"
    episodes = []

    for chunk_dir in sorted(episodes_dir.glob("chunk-*")):
        for parquet_file in sorted(chunk_dir.glob("*.parquet")):
            df = pd.read_parquet(parquet_file)
            for _, row in df.iterrows():
                ep = row.to_dict()
                episodes.append(ep)

    return sorted(episodes, key=lambda x: x["episode_index"])


def save_episodes_v30(root: Path, episodes: list[dict]) -> None:
    """Save episodes metadata to v3.0 parquet format."""
    episodes_dir = root / "meta" / "episodes"

    # Load existing structure to preserve chunk organization
    # For simplicity, write all episodes to chunk-000/file-000.parquet
    chunk_dir = episodes_dir / "chunk-000"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(episodes)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, chunk_dir / "file-000.parquet")


def load_episodes(root: Path, version: str) -> list[dict]:
    """Load episodes metadata based on dataset version."""
    if version.startswith("v3"):
        return load_episodes_v30(root)
    else:
        return load_episodes_v21(root)


def save_episodes(root: Path, episodes: list[dict], version: str) -> None:
    """Save episodes metadata based on dataset version."""
    if version.startswith("v3"):
        save_episodes_v30(root, episodes)
    else:
        save_episodes_v21(root, episodes)


def compute_frames_to_keep(
    states: np.ndarray,
    threshold_factor: float = 0.1,
    min_idle_segment: int = 5,
    keep_frames_per_idle: int = 1,
) -> tuple[list[int], float, list[float], dict]:
    """
    Compute which frames to keep, removing idle segments throughout the trajectory.

    Removes idle segments at start, end, AND within the trajectory.

    Args:
        states: Array of shape (N, state_dim) with state values
        threshold_factor: Threshold = median(deltas) * threshold_factor
        min_idle_segment: Minimum consecutive idle frames to consider for removal
        keep_frames_per_idle: Number of frames to keep from each internal idle segment

    Returns:
        Tuple of:
            - frames_to_keep: Sorted list of frame indices to keep
            - threshold: The computed threshold value
            - deltas: List of delta norms for each frame
            - trim_info: Dict with details about what was trimmed
    """
    if len(states) < 2:
        return list(range(len(states))), 0.0, [0.0], {"idle_segments": []}

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
        return list(range(len(states))), 0.0, deltas, {"idle_segments": []}

    median_delta = np.median(non_zero_deltas)
    threshold = median_delta * threshold_factor

    # Mark frames as active (delta > threshold) or idle
    is_active = [d > threshold for d in deltas]

    # Find contiguous idle segments
    idle_segments = []
    i = 0
    while i < len(is_active):
        if not is_active[i]:
            # Start of idle segment
            start = i
            while i < len(is_active) and not is_active[i]:
                i += 1
            end = i  # exclusive
            idle_segments.append((start, end))
        else:
            i += 1

    # Determine frames to remove
    frames_to_remove = set()
    segment_info = []

    for start, end in idle_segments:
        segment_length = end - start
        is_at_start = start == 0
        is_at_end = end == len(states)

        if segment_length >= min_idle_segment:
            if is_at_start:
                # Trim from beginning, keep only last frame of segment
                for j in range(start, end - 1):
                    frames_to_remove.add(j)
                segment_info.append({
                    "type": "start",
                    "range": (start, end),
                    "original_length": segment_length,
                    "frames_removed": segment_length - 1,
                })
            elif is_at_end:
                # Trim from end, keep only first frame of segment
                for j in range(start + 1, end):
                    frames_to_remove.add(j)
                segment_info.append({
                    "type": "end",
                    "range": (start, end),
                    "original_length": segment_length,
                    "frames_removed": segment_length - 1,
                })
            else:
                # Middle segment - keep only a few frames for continuity
                frames_to_keep_in_segment = min(keep_frames_per_idle, segment_length)
                if frames_to_keep_in_segment == 1:
                    # Keep middle frame
                    keep_indices = {start + segment_length // 2}
                else:
                    # Spread frames evenly
                    step = segment_length / (frames_to_keep_in_segment + 1)
                    keep_indices = {start + int(step * (k + 1)) for k in range(frames_to_keep_in_segment)}

                for j in range(start, end):
                    if j not in keep_indices:
                        frames_to_remove.add(j)

                segment_info.append({
                    "type": "middle",
                    "range": (start, end),
                    "original_length": segment_length,
                    "frames_removed": segment_length - len(keep_indices),
                    "frames_kept": sorted(keep_indices),
                })

    frames_to_keep = sorted([i for i in range(len(states)) if i not in frames_to_remove])

    trim_info = {
        "idle_segments": segment_info,
        "total_idle_segments": len(idle_segments),
        "segments_trimmed": len(segment_info),
        "threshold": threshold,
        "median_delta": median_delta,
    }

    return frames_to_keep, threshold, deltas, trim_info


def joints_to_ee(
    joint_values: np.ndarray,
    kinematics,
    rotation_class,
    translation_offset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert joint positions to end-effector pose using quaternion orientation.

    Args:
        joint_values: Array of shape (6,) with joint positions
                     [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        kinematics: RobotKinematics instance
        rotation_class: Rotation class for converting rotation matrix to quaternion
        translation_offset: Optional (3,) array with [tx, ty, tz]

    Returns:
        Tuple of:
            - ee_pose: Array of shape (7,) with [x, y, z, qw, qx, qy, qz]
            - ee_pose_with_gripper: Array of shape (8,) with [x, y, z, qw, qx, qy, qz, gripper]
    """
    arm_joints = joint_values[:5].astype(np.float64)
    gripper_pos = float(joint_values[5])

    T = kinematics.forward_kinematics(arm_joints)
    position = T[:3, 3]

    if translation_offset is not None:
        position = position + translation_offset

    # as_quat() returns [x, y, z, w], reorder to [w, x, y, z]
    quat_xyzw = rotation_class.from_matrix(T[:3, :3]).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    # Enforce canonical form: qw >= 0
    if quat_wxyz[0] < 0:
        quat_wxyz = -quat_wxyz

    ee_pose = np.concatenate([position, quat_wxyz]).astype(np.float32)
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


def resize_parquet_depth_images(parquet_path: Path, depth_key: str, target_size: int) -> int:
    """Resize depth images stored as image columns in a parquet file.

    Depth images stored in parquet have struct<bytes, path> format.
    This reads each image, resizes it, and writes it back.

    Args:
        parquet_path: Path to the parquet file
        depth_key: Column name for depth images
        target_size: Target square size

    Returns:
        Number of images resized
    """
    df = pd.read_parquet(parquet_path)
    if depth_key not in df.columns:
        return 0

    resized_count = 0
    new_depth_entries = []

    for entry in df[depth_key]:
        if entry is None:
            new_depth_entries.append(entry)
            continue

        # Handle struct<bytes, path> format
        if isinstance(entry, dict) and "bytes" in entry:
            image_bytes = entry["bytes"]
            resized_bytes = resize_image_bytes(image_bytes, target_size)
            new_entry = {**entry, "bytes": resized_bytes}
            new_depth_entries.append(new_entry)
            resized_count += 1
        else:
            new_depth_entries.append(entry)

    df[depth_key] = new_depth_entries
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, parquet_path)

    return resized_count


def resize_and_trim_video_file(
    input_path: Path,
    output_path: Path,
    target_size: int,
    frames_to_keep: list[int] | None = None,
) -> int:
    """
    Resize and optionally trim video file.

    Args:
        input_path: Path to input video
        output_path: Path to output video
        target_size: Target square size
        frames_to_keep: List of frame indices to keep (0-indexed), None for all frames

    Returns:
        Number of frames written
    """
    import av
    from fractions import Fraction

    # Convert to set for O(1) lookup
    keep_set = set(frames_to_keep) if frames_to_keep is not None else None

    # Read input video
    frames = []
    with av.open(str(input_path)) as container:
        stream = container.streams.video[0]
        original_fps = stream.average_rate  # Keep as Fraction
        for frame_idx, frame in enumerate(container.decode(video=0)):
            if keep_set is not None and frame_idx not in keep_set:
                continue
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
    episode_frames_to_keep: dict[int, tuple[list[int], int]],
) -> None:
    """
    Trim LMDB point clouds by removing entries for trimmed frames and reindexing.

    Args:
        lmdb_path: Path to the LMDB directory
        episode_frames_to_keep: Dict mapping episode_index to (frames_to_keep, original_length)
                                where frames_to_keep is a list of frame indices to keep
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

    # Build reindex maps for each episode: old_frame_idx -> new_frame_idx
    reindex_maps = {}
    for ep_idx, (frames_to_keep, _) in episode_frames_to_keep.items():
        reindex_maps[ep_idx] = {old_idx: new_idx for new_idx, old_idx in enumerate(frames_to_keep)}

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

                if ep_idx not in reindex_maps:
                    # Episode not trimmed, keep all frames
                    dst_txn.put(key, value)
                    entries_copied += 1
                    continue

                reindex_map = reindex_maps[ep_idx]
                if frame_idx not in reindex_map:
                    # Frame was trimmed
                    entries_removed += 1
                    continue

                # Reindex frame
                new_frame_idx = reindex_map[frame_idx]
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
    image_size: int = 256,
    trim_idle_frames: bool = True,
    trim_threshold_factor: float = 0.1,
    min_idle_segment: int = 5,
    keep_frames_per_idle: int = 1,
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
    output_joint_action_key: str = "action.joints",
    include_depth: bool = False,
) -> None:
    """
    Convert a LeRobot dataset to PointAct format.

    Args:
        dataset_dir: Path to input dataset
        urdf_path: Path to robot URDF file
        output_dir: Path for output dataset
        target_frame: Name of the EE frame in URDF
        joint_names: List of joint names for FK
        image_size: Target image size (square)
        trim_idle_frames: Enable trimming of idle frames
        trim_threshold_factor: Threshold = median(deltas) * this factor
        min_idle_segment: Minimum consecutive idle frames to consider for removal
        keep_frames_per_idle: Number of frames to keep from each internal idle segment
        state_key: Input state key
        action_key: Input action key
        rgb_key: Input RGB image key
        depth_key: Input depth image key
        point_cloud_key: Input point cloud key
        output_*: Output key names
        include_depth: If True, preserve depth images (resized and trimmed). Dropped by default.
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
    logging.info(f"Translation offset: [{ROBOT_FRAME['tx']}, {ROBOT_FRAME['ty']}, {ROBOT_FRAME['tz']}]")
    logging.info(f"Target image size: {image_size}x{image_size}")
    logging.info(f"Include depth: {include_depth}")
    logging.info(f"Trim idle frames: {trim_idle_frames}")
    if trim_idle_frames:
        logging.info(f"Trim threshold factor: {trim_threshold_factor}")
        logging.info(f"Min idle segment length: {min_idle_segment}")
        logging.info(f"Frames to keep per idle segment: {keep_frames_per_idle}")

    tx, ty, tz = ROBOT_FRAME['tx'], ROBOT_FRAME['ty'], ROBOT_FRAME['tz']
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

    videos_dir = dataset_path / "videos"
    if not include_depth:
        # Remove depth video directory if it exists
        logging.info("Removing depth images...")
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
    episodes = load_episodes(dataset_path, version)
    logging.info(f"Found {len(episodes)} episodes")

    # First pass: Load all data, compute FK, and determine frames to keep per episode
    logging.info("First pass: Computing FK and determining frames to keep...")

    # Dict to store trim info: {episode_idx: (frames_to_keep, original_length, trim_info)}
    episode_frames_to_keep: dict[int, tuple[list[int], int, dict]] = {}

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
        joint_actions = []  # Original joint commands for replay
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
            joint_actions.append(action_joints.astype(np.float32))  # Save original joint commands
            gripper_states.append(np.array([state_joints[5]], dtype=np.float32))

        # Update episode dataframe with FK results
        ep_df[state_key] = new_states
        ep_df[action_key] = new_actions
        ep_df[output_ee_state_key] = ee_states
        ep_df[output_joint_state_key] = joint_states
        ep_df[output_joint_action_key] = joint_actions  # Original joint commands for replay
        ep_df[output_gripper_state_key] = gripper_states
        episode_data[episode_idx] = ep_df

        # Compute frames to keep if trimming is enabled
        original_length = len(ep_df)
        if trim_idle_frames:
            states_array = np.array(new_states)
            frames_to_keep, threshold, deltas, trim_info = compute_frames_to_keep(
                states_array, trim_threshold_factor, min_idle_segment, keep_frames_per_idle
            )
            episode_frames_to_keep[episode_idx] = (frames_to_keep, original_length, trim_info)
        else:
            episode_frames_to_keep[episode_idx] = (list(range(original_length)), original_length, {})

    # Print trim report
    if trim_idle_frames:
        logging.info("")
        logging.info("=" * 60)
        logging.info("=== Idle Frame Trimming Report ===")
        logging.info("=" * 60)
        total_original = 0
        total_kept = 0
        total_start_trimmed = 0
        total_end_trimmed = 0
        total_middle_trimmed = 0
        total_middle_segments = 0

        for episode_idx in sorted(episode_frames_to_keep.keys()):
            frames_to_keep, original_length, trim_info = episode_frames_to_keep[episode_idx]
            new_length = len(frames_to_keep)
            frames_removed = original_length - new_length
            total_original += original_length
            total_kept += new_length

            # Count by segment type
            start_removed = 0
            end_removed = 0
            middle_removed = 0
            middle_count = 0

            for seg in trim_info.get("idle_segments", []):
                seg_type = seg.get("type", "")
                removed = seg.get("frames_removed", 0)
                if seg_type == "start":
                    start_removed += removed
                    total_start_trimmed += removed
                elif seg_type == "end":
                    end_removed += removed
                    total_end_trimmed += removed
                elif seg_type == "middle":
                    middle_removed += removed
                    middle_count += 1
                    total_middle_trimmed += removed
                    total_middle_segments += 1

            # Build detailed message
            details = []
            if start_removed > 0:
                details.append(f"start: -{start_removed}")
            if end_removed > 0:
                details.append(f"end: -{end_removed}")
            if middle_removed > 0:
                details.append(f"middle: -{middle_removed} ({middle_count} segments)")

            detail_str = ", ".join(details) if details else "no trimming"
            logging.info(
                f"Episode {episode_idx}: {original_length} -> {new_length} frames "
                f"(-{frames_removed}) [{detail_str}]"
            )

            # Log detailed middle segment info for debugging
            for seg in trim_info.get("idle_segments", []):
                if seg.get("type") == "middle":
                    seg_range = seg.get("range", (0, 0))
                    seg_len = seg.get("original_length", 0)
                    removed = seg.get("frames_removed", 0)
                    kept_frames = seg.get("frames_kept", [])
                    logging.info(
                        f"  - Middle idle segment: frames {seg_range[0]}-{seg_range[1]} "
                        f"({seg_len} frames), removed {removed}, kept {len(kept_frames)}"
                    )

        frames_removed_total = total_original - total_kept
        pct_trimmed = 100.0 * frames_removed_total / total_original if total_original > 0 else 0.0

        logging.info("")
        logging.info("=" * 60)
        logging.info("=== Summary ===")
        logging.info(f"Total frames: {total_original} -> {total_kept} ({pct_trimmed:.1f}% trimmed)")
        logging.info(f"  - Start trimming: {total_start_trimmed} frames removed")
        logging.info(f"  - End trimming: {total_end_trimmed} frames removed")
        logging.info(
            f"  - Middle trimming: {total_middle_trimmed} frames removed "
            f"({total_middle_segments} idle segments)"
        )
        logging.info("=" * 60)
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
        frames_to_keep, original_length, trim_info = episode_frames_to_keep[ep_idx]

        # Apply trim - select only the frames to keep
        trimmed_df = ep_df.iloc[frames_to_keep].copy()

        # Drop depth column if present (we removed the video files, now remove from parquet)
        if not include_depth and depth_key in trimmed_df.columns:
            trimmed_df = trimmed_df.drop(columns=[depth_key])

        # Drop original RGB key if present (renamed to output_image_key, stored as video)
        if rgb_key in trimmed_df.columns:
            trimmed_df = trimmed_df.drop(columns=[rgb_key])

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

    # Resize depth images stored in parquet files (if preserving depth)
    if include_depth:
        logging.info("Resizing depth images in parquet files...")
        total_depth_resized = 0
        for parquet_path in tqdm(parquet_files, desc="Resizing depth in parquet"):
            resized = resize_parquet_depth_images(parquet_path, depth_key, image_size)
            total_depth_resized += resized
        logging.info(f"Resized {total_depth_resized} depth images in parquet files")

    # Update episode metadata
    logging.info("Updating episode metadata...")
    for ep in episodes:
        ep_idx = ep["episode_index"]
        if ep_idx in episode_frames_to_keep:
            frames_to_keep, original_length, trim_info = episode_frames_to_keep[ep_idx]
            ep["length"] = len(frames_to_keep)

    save_episodes(dataset_path, episodes, version)

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

                    if ep_idx is not None and ep_idx in episode_frames_to_keep:
                        frames_to_keep, original_length, trim_info = episode_frames_to_keep[ep_idx]
                        logging.info(
                            f"Resizing and trimming {video_file.name} "
                            f"(keeping {len(frames_to_keep)} of {original_length} frames)..."
                        )
                        resize_and_trim_video_file(
                            video_file, new_video_file, image_size, frames_to_keep
                        )
                    else:
                        logging.info(f"Resizing {video_file.name}...")
                        resize_and_trim_video_file(video_file, new_video_file, image_size)

                # Remove old videos
                shutil.rmtree(rgb_video_dir)

    # Resize and trim depth videos if preserving depth
    if include_depth and videos_dir.exists():
        logging.info("Resizing and trimming depth videos...")
        for chunk_dir in sorted(videos_dir.glob("chunk-*")):
            depth_video_dir = chunk_dir / depth_key
            if depth_video_dir.exists():
                for video_file in sorted(depth_video_dir.glob("*.mp4")):
                    video_name = video_file.stem
                    ep_idx = int(video_name.split("_")[1]) if video_name.startswith("episode_") else None

                    temp_output = video_file.with_suffix(".tmp.mp4")
                    if ep_idx is not None and ep_idx in episode_frames_to_keep:
                        frames_to_keep, original_length, trim_info = episode_frames_to_keep[ep_idx]
                        logging.info(
                            f"Resizing and trimming depth {video_file.name} "
                            f"(keeping {len(frames_to_keep)} of {original_length} frames)..."
                        )
                        resize_and_trim_video_file(video_file, temp_output, image_size, frames_to_keep)
                    else:
                        logging.info(f"Resizing depth {video_file.name}...")
                        resize_and_trim_video_file(video_file, temp_output, image_size)

                    # Replace original with resized/trimmed version
                    temp_output.rename(video_file)

    # Trim LMDB point clouds if they exist
    lmdb_path = dataset_path / "point_clouds"
    if lmdb_path.exists() and trim_idle_frames:
        # Convert to the format expected by trim_lmdb_point_clouds
        lmdb_trim_data = {
            ep_idx: (frames, orig_len)
            for ep_idx, (frames, orig_len, _) in episode_frames_to_keep.items()
        }
        trim_lmdb_point_clouds(lmdb_path, lmdb_trim_data)

    # Update info.json with new feature definitions
    ee_names = ["x", "y", "z", "qw", "qx", "qy", "qz"]
    state_names = ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper_openness"]

    # Update state and action features
    info["features"][state_key] = {
        "dtype": "float32",
        "shape": [8],
        "names": {"motors": state_names},
    }

    info["features"][action_key] = {
        "dtype": "float32",
        "shape": [8],
        "names": {"motors": state_names},
    }

    # Add new state features
    info["features"][output_ee_state_key] = {
        "dtype": "float32",
        "shape": [7],
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

    # Add original joint action for replay
    info["features"][output_joint_action_key] = {
        "dtype": "float32",
        "shape": [len(all_joint_names)],
        "names": {"motors": all_joint_names},
    }

    # Update image feature
    if rgb_key in info["features"]:
        info["features"].pop(rgb_key)
        info["features"][output_image_key] = {
            "dtype": "video",
            "shape": [image_size, image_size, 3],
            "names": ["height", "width", "rgb"],
        }

    # Update depth feature shape if preserving depth
    if include_depth and depth_key in info.get("features", {}):
        depth_feature = info["features"][depth_key]
        if "shape" in depth_feature and len(depth_feature["shape"]) >= 2:
            depth_feature["shape"][0] = image_size
            depth_feature["shape"][1] = image_size

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
        "min_idle_segment": min_idle_segment if trim_idle_frames else None,
        "keep_frames_per_idle": keep_frames_per_idle if trim_idle_frames else None,
    }

    save_info(dataset_path, info)

    # Regenerate stats.json to match the new feature definitions
    regenerate_stats(
        dataset_path=dataset_path,
        info=info,
        state_key=state_key,
        action_key=action_key,
        rgb_key=rgb_key,
        depth_key=depth_key,
        point_cloud_key=point_cloud_key,
        output_image_key=output_image_key,
        output_point_cloud_key=output_point_cloud_key,
        output_ee_state_key=output_ee_state_key,
        output_joint_state_key=output_joint_state_key,
        output_gripper_state_key=output_gripper_state_key,
        output_joint_action_key=output_joint_action_key,
        include_depth=include_depth,
    )

    logging.info("Conversion to PointAct format complete!")
    logging.info("Output features:")
    logging.info(f"  {state_key}: shape [8], {state_names}")
    logging.info(f"  {action_key}: shape [8], {state_names}")
    logging.info(f"  {output_ee_state_key}: shape [7], {ee_names}")
    logging.info(f"  {output_joint_state_key}: shape [6], {joint_names + ['gripper']}")
    logging.info(f"  {output_joint_action_key}: shape [6], {joint_names + ['gripper']} (original joint commands)")
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
        image_size=args.image_size,
        trim_idle_frames=not args.no_trim_idle_frames,
        trim_threshold_factor=args.trim_threshold_factor,
        min_idle_segment=args.min_idle_segment,
        keep_frames_per_idle=args.keep_frames_per_idle,
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
        output_joint_action_key=args.output_joint_action_key,
        include_depth=args.include_depth,
    )
