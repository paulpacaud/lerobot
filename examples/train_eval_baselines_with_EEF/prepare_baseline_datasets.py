#!/usr/bin/env python
"""
Prepare baseline datasets for EE vs Joint action space comparison.

This script takes a LeRobot dataset with joint-space actions and creates two
trimmed datasets for fair comparison:
  - <name>_ee: Actions are EE poses [x, y, z, wx, wy, wz, gripper]
  - <name>_joints: Actions are joint positions (original format, but trimmed)

Both datasets undergo the same idle frame trimming (computed on EE pose deltas)
to ensure fair comparison between the two action representations.

Pipeline approach (leverages existing converters for reliability):
  1. Convert v3.0 → v2.1 (simpler per-episode format)
  2. Process in v2.1 format (compute FK, trim idle frames)
  3. Convert v2.1 → v3.0 (handles all complex metadata)

Trimming strategy (same as PointAct):
  1. Compute EE poses via forward kinematics
  2. Compute delta (L2 norm) between consecutive EE poses
  3. Threshold = median(deltas) * trim_threshold_factor
  4. Find contiguous idle segments (delta < threshold)
  5. Trim segments with length >= min_idle_segment

Usage:
    python examples/train_eval_baselines_with_EEF/prepare_baseline_datasets.py \
        --input_dir=/path/to/joint_dataset \
        --urdf_path=./URDF/SO101/so101_new_calib.urdf

Output datasets are created in the same directory as input_dir:
    /path/to/joint_dataset_ee
    /path/to/joint_dataset_joints
"""

import json
import logging
import multiprocessing
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import av
import jsonlines
import numpy as np
import pandas as pd
from tap import Tap
from tqdm import tqdm

from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.rotation import Rotation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
EE_FEATURE_NAMES = ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "ee.gripper_pos"]
EE_ACTION_DIM = 7
GRIPPER_IDX = 5

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

DEFAULT_CHUNK_SIZE = 1000

# Features to remove from output datasets (depth not needed for baseline comparison)
FEATURES_TO_REMOVE = ["observation.images.front_depth"]


class Args(Tap):
    """Arguments for preparing baseline datasets."""

    # Required
    input_dir: str  # Path to input dataset with joint actions
    urdf_path: str  # Path to URDF file

    # Kinematics
    target_frame: str = "gripper_frame_link"  # End-effector frame name

    # Trimming parameters (same defaults as PointAct)
    no_trim: bool = False  # Disable trimming entirely
    trim_threshold_factor: float = 0.05  # Threshold = median(deltas) * this factor
    min_idle_segment: int = 5  # Minimum consecutive idle frames to trim
    keep_frames_per_idle: int = 1  # Frames to keep from internal idle segments

    # Parallelization
    num_workers: int = 0  # Number of parallel workers (0 = auto, based on CPU count)


# =============================================================================
# JSON/JSONL helpers for v2.1 format
# =============================================================================


def load_info(dataset_path: Path) -> dict:
    """Load info.json from dataset."""
    with open(dataset_path / "meta" / "info.json") as f:
        return json.load(f)


def save_info(dataset_path: Path, info: dict) -> None:
    """Save info.json to dataset."""
    with open(dataset_path / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)


def load_episodes_v21(root: Path) -> list[dict]:
    """Load episodes metadata from v2.1 jsonl format."""
    episodes = []
    episodes_file = root / "meta" / "episodes.jsonl"
    if episodes_file.exists():
        with jsonlines.open(episodes_file) as reader:
            for ep in reader:
                episodes.append(ep)
    return sorted(episodes, key=lambda x: x["episode_index"])


def save_episodes_v21(root: Path, episodes: list[dict]) -> None:
    """Save episodes metadata to v2.1 jsonl format."""
    with jsonlines.open(root / "meta" / "episodes.jsonl", mode="w") as writer:
        for ep in episodes:
            writer.write(ep)


def load_episodes_stats_v21(root: Path) -> list[dict]:
    """Load episode stats from v2.1 jsonl format."""
    stats = []
    stats_file = root / "meta" / "episodes_stats.jsonl"
    if stats_file.exists():
        with jsonlines.open(stats_file) as reader:
            for s in reader:
                stats.append(s)
    return sorted(stats, key=lambda x: x["episode_index"])


def save_episodes_stats_v21(root: Path, stats: list[dict]) -> None:
    """Save episode stats to v2.1 jsonl format."""
    with jsonlines.open(root / "meta" / "episodes_stats.jsonl", mode="w") as writer:
        for s in stats:
            writer.write(s)


def cleanup_removed_features_from_metadata(dataset_path: Path, features_to_remove: list[str]) -> None:
    """
    Remove stale feature entries from v3.0 metadata files.

    After v2.1 → v3.0 conversion, the converter may re-introduce entries for
    features that were removed during processing. This function cleans them up.

    Args:
        dataset_path: Path to the v3.0 dataset
        features_to_remove: List of feature names to remove from metadata
    """
    removed_from_stats = []
    removed_from_episodes_stats = []

    # Clean meta/stats.json
    stats_file = dataset_path / "meta" / "stats.json"
    if stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)

        for feature in features_to_remove:
            if feature in stats:
                del stats[feature]
                removed_from_stats.append(feature)

        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

    # Clean meta/episodes_stats.jsonl
    episodes_stats_file = dataset_path / "meta" / "episodes_stats.jsonl"
    if episodes_stats_file.exists():
        modified_entries = []
        with jsonlines.open(episodes_stats_file) as reader:
            for entry in reader:
                if "stats" in entry:
                    for feature in features_to_remove:
                        if feature in entry["stats"]:
                            del entry["stats"][feature]
                            if feature not in removed_from_episodes_stats:
                                removed_from_episodes_stats.append(feature)
                modified_entries.append(entry)

        with jsonlines.open(episodes_stats_file, mode="w") as writer:
            for entry in modified_entries:
                writer.write(entry)

    # Log what was removed
    if removed_from_stats:
        logging.info(f"Removed from stats.json: {removed_from_stats}")
    if removed_from_episodes_stats:
        logging.info(f"Removed from episodes_stats.jsonl: {removed_from_episodes_stats}")


# =============================================================================
# Conversion wrappers
# =============================================================================


def convert_v3_to_v21(input_path: Path, output_path: Path) -> None:
    """Wrap existing v3→v2.1 converter."""
    from examples.post_process_dataset.convert_lerobot_dataset_v3_to_v2 import convert_dataset

    convert_dataset(str(input_path), str(output_path))


def convert_v21_to_v30_local(input_path: Path, output_path: Path) -> None:
    """
    Convert v2.1 dataset to v3.0 format locally (without push to hub).

    Uses the existing converter functions but operates entirely locally.
    """
    from lerobot.datasets.v30.convert_dataset_v21_to_v30 import (
        convert_data,
        convert_episodes_metadata,
        convert_info,
        convert_tasks,
        convert_videos,
    )
    from lerobot.datasets.utils import (
        DEFAULT_DATA_FILE_SIZE_IN_MB,
        DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    )

    # Clean output if exists
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Converting v2.1 -> v3.0: {input_path} -> {output_path}")

    # Convert components
    convert_info(input_path, output_path, DEFAULT_DATA_FILE_SIZE_IN_MB, DEFAULT_VIDEO_FILE_SIZE_IN_MB)
    convert_tasks(input_path, output_path)
    episodes_metadata = convert_data(input_path, output_path, DEFAULT_DATA_FILE_SIZE_IN_MB)
    episodes_videos_metadata = convert_videos(input_path, output_path, DEFAULT_VIDEO_FILE_SIZE_IN_MB)
    convert_episodes_metadata(input_path, output_path, episodes_metadata, episodes_videos_metadata)


# =============================================================================
# Trimming logic (kept from original)
# =============================================================================


def compute_frames_to_keep(
    ee_states: np.ndarray,
    threshold_factor: float = 0.05,
    min_idle_segment: int = 5,
    keep_frames_per_idle: int = 1,
) -> tuple[list[int], dict]:
    """
    Compute which frames to keep based on EE pose movement.

    This is the same trimming strategy as PointAct:
    - Compute delta between consecutive EE poses
    - Threshold = median(deltas) * threshold_factor
    - Trim idle segments (start, end, and middle)

    Args:
        ee_states: Array of shape (N, 7) with EE poses [x,y,z,wx,wy,wz,gripper]
        threshold_factor: Threshold = median(deltas) * this factor
        min_idle_segment: Minimum consecutive idle frames to consider for removal
        keep_frames_per_idle: Number of frames to keep from each internal idle segment

    Returns:
        Tuple of (frames_to_keep, trim_info)
    """
    if len(ee_states) < 2:
        return list(range(len(ee_states))), {
            "idle_segments": [],
            "original_length": len(ee_states),
            "trimmed_length": len(ee_states),
        }

    # Compute deltas (L2 norm between consecutive frames)
    deltas = []
    for i in range(len(ee_states) - 1):
        delta = np.linalg.norm(ee_states[i + 1] - ee_states[i])
        deltas.append(delta)
    deltas.append(0.0)  # Last frame has no delta

    # Compute threshold from median of non-zero deltas
    non_zero_deltas = [d for d in deltas if d > 1e-8]
    if len(non_zero_deltas) == 0:
        return list(range(len(ee_states))), {
            "idle_segments": [],
            "original_length": len(ee_states),
            "trimmed_length": len(ee_states),
        }

    median_delta = np.median(non_zero_deltas)
    threshold = median_delta * threshold_factor

    # Mark frames as active or idle
    is_active = [d > threshold for d in deltas]

    # Find contiguous idle segments
    idle_segments = []
    i = 0
    while i < len(is_active):
        if not is_active[i]:
            start = i
            while i < len(is_active) and not is_active[i]:
                i += 1
            end = i
            idle_segments.append((start, end))
        else:
            i += 1

    # Determine frames to remove
    frames_to_remove = set()
    segment_info = []

    for start, end in idle_segments:
        segment_length = end - start
        is_at_start = start == 0
        is_at_end = end == len(ee_states)

        if segment_length >= min_idle_segment:
            if is_at_start:
                # Trim from beginning, keep only last frame
                for j in range(start, end - 1):
                    frames_to_remove.add(j)
                segment_info.append({
                    "type": "start", "range": (start, end),
                    "frames_removed": segment_length - 1,
                })
            elif is_at_end:
                # Trim from end, keep only first frame
                for j in range(start + 1, end):
                    frames_to_remove.add(j)
                segment_info.append({
                    "type": "end", "range": (start, end),
                    "frames_removed": segment_length - 1,
                })
            else:
                # Middle segment - keep a few frames for continuity
                frames_to_keep_in_segment = min(keep_frames_per_idle, segment_length)
                if frames_to_keep_in_segment == 1:
                    keep_indices = {start + segment_length // 2}
                else:
                    step = segment_length / (frames_to_keep_in_segment + 1)
                    keep_indices = {start + int(step * (k + 1)) for k in range(frames_to_keep_in_segment)}

                for j in range(start, end):
                    if j not in keep_indices:
                        frames_to_remove.add(j)

                segment_info.append({
                    "type": "middle", "range": (start, end),
                    "frames_removed": segment_length - len(keep_indices),
                })

    frames_to_keep = sorted([i for i in range(len(ee_states)) if i not in frames_to_remove])

    trim_info = {
        "idle_segments": segment_info,
        "threshold": threshold,
        "median_delta": median_delta,
        "original_length": len(ee_states),
        "trimmed_length": len(frames_to_keep),
    }

    return frames_to_keep, trim_info


def joints_to_ee(
    joints: np.ndarray,
    kinematics: RobotKinematics,
    gripper_idx: int,
) -> np.ndarray:
    """Convert joint positions to EE pose [x, y, z, wx, wy, wz, gripper]."""
    gripper_pos = float(joints[gripper_idx])
    fk_joints = np.delete(joints, gripper_idx).astype(np.float64)

    transform = kinematics.forward_kinematics(fk_joints)
    position = transform[:3, 3]
    rotation = Rotation.from_matrix(transform[:3, :3]).as_rotvec()

    return np.concatenate([position, rotation, [gripper_pos]]).astype(np.float32)


def trim_video_file(
    input_path: Path,
    output_path: Path,
    frames_to_keep: list[int],
) -> int:
    """
    Trim video file to keep only specified frames.

    Args:
        input_path: Path to input video
        output_path: Path to output video
        frames_to_keep: List of frame indices to keep (0-indexed)

    Returns:
        Number of frames written
    """
    keep_set = set(frames_to_keep)

    # Read input video and filter frames
    frames = []
    with av.open(str(input_path)) as container:
        stream = container.streams.video[0]
        original_fps = stream.average_rate
        width = stream.width
        height = stream.height

        for frame_idx, frame in enumerate(container.decode(video=0)):
            if frame_idx in keep_set:
                img = frame.to_ndarray(format="rgb24")
                frames.append(img)

    if len(frames) == 0:
        logging.warning(f"No frames to write for {output_path}")
        return 0

    # Write output video
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(output_path), mode="w") as output_container:
        output_stream = output_container.add_stream("libx264", rate=Fraction(original_fps))
        output_stream.width = width
        output_stream.height = height
        output_stream.pix_fmt = "yuv420p"

        for img in frames:
            av_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in output_stream.encode(av_frame):
                output_container.mux(packet)

        for packet in output_stream.encode():
            output_container.mux(packet)

    return len(frames)


# =============================================================================
# v2.1 Processing
# =============================================================================


def _trim_video_task(task: tuple) -> tuple[str, int]:
    """
    Worker function for parallel video trimming.

    Args:
        task: Tuple of (input_path, output_path, frames_to_keep, rel_path_str)

    Returns:
        Tuple of (rel_path_str, num_frames_written)
    """
    input_path, output_path, frames_to_keep, rel_path_str = task
    num_frames = trim_video_file(Path(input_path), Path(output_path), frames_to_keep)
    return rel_path_str, num_frames


def process_v21_dataset(
    input_path: Path,
    output_ee_path: Path,
    output_joints_path: Path,
    kinematics: RobotKinematics,
    args: Args,
    num_workers: int = 0,
) -> dict:
    """
    Process a v2.1 dataset: compute FK, trim idle frames, create EE and joints variants.

    v2.1 format has one parquet per episode (data/chunk-XXX/episode_YYYYYY.parquet)
    and one video per episode (videos/chunk-XXX/<camera>/episode_YYYYYY.mp4).

    Returns:
        Dictionary with processing stats
    """
    info = load_info(input_path)
    fps = info.get("fps", 30)
    motor_names = info.get("features", {}).get("action", {}).get("names", [])

    # Load episodes metadata
    episodes = load_episodes_v21(input_path)
    episodes_stats = load_episodes_stats_v21(input_path)

    # Create output directories
    for out_path in [output_ee_path, output_joints_path]:
        if out_path.exists():
            shutil.rmtree(out_path)
        out_path.mkdir(parents=True)
        (out_path / "meta").mkdir()
        (out_path / "data").mkdir()
        (out_path / "videos").mkdir()

        # Copy tasks.jsonl
        shutil.copy(input_path / "meta" / "tasks.jsonl", out_path / "meta" / "tasks.jsonl")

        # Copy stats.json if exists
        stats_file = input_path / "meta" / "stats.json"
        if stats_file.exists():
            shutil.copy(stats_file, out_path / "meta" / "stats.json")

    # Process each episode
    episode_results = {}
    all_ee_actions = []
    all_joint_actions = []

    data_dir = input_path / "data"
    episode_files = sorted(data_dir.glob("**/episode_*.parquet"))

    for ep_file in tqdm(episode_files, desc="Processing episodes"):
        # Parse episode index from filename
        ep_idx = int(ep_file.stem.split("_")[1])
        chunk_dir = ep_file.parent.name  # e.g., "chunk-000"

        # Read episode data
        df = pd.read_parquet(ep_file)
        df = df.sort_values("frame_index").reset_index(drop=True)

        # Compute EE poses
        ee_states = []
        ee_actions = []
        joint_actions = []

        for _, row in df.iterrows():
            # State (actual position) - used for trimming
            state_joints = np.array(row["observation.state"], dtype=np.float32)
            ee_state = joints_to_ee(state_joints, kinematics, GRIPPER_IDX)
            ee_states.append(ee_state)

            # Action (commanded target) - used for dataset output
            action_joints = np.array(row["action"], dtype=np.float32)
            ee_action = joints_to_ee(action_joints, kinematics, GRIPPER_IDX)
            ee_actions.append(ee_action)
            joint_actions.append(action_joints)

        ee_states = np.array(ee_states)
        ee_actions = np.array(ee_actions)
        joint_actions = np.array(joint_actions)

        # Compute frames to keep
        if args.no_trim:
            frames_to_keep = list(range(len(df)))
            trim_info = {"idle_segments": [], "original_length": len(df), "trimmed_length": len(df)}
        else:
            frames_to_keep, trim_info = compute_frames_to_keep(
                ee_states,
                args.trim_threshold_factor,
                args.min_idle_segment,
                args.keep_frames_per_idle,
            )

        # Collect trimmed actions for this episode
        ep_ee_actions = np.array([ee_actions[i] for i in frames_to_keep])
        ep_joint_actions = np.array([joint_actions[i] for i in frames_to_keep])

        episode_results[ep_idx] = {
            "frames_to_keep": frames_to_keep,
            "trim_info": trim_info,
            "original_length": len(df),
            "ee_actions": ep_ee_actions,
            "joint_actions": ep_joint_actions,
        }

        # Collect actions for global stats
        for idx in frames_to_keep:
            all_ee_actions.append(ee_actions[idx])
            all_joint_actions.append(joint_actions[idx])

        # Create trimmed dataframes
        trimmed_df_ee = df.iloc[frames_to_keep].copy()
        trimmed_df_joints = df.iloc[frames_to_keep].copy()

        # Remove depth columns (and other features to remove)
        for feature in FEATURES_TO_REMOVE:
            if feature in trimmed_df_ee.columns:
                trimmed_df_ee = trimmed_df_ee.drop(columns=[feature])
            if feature in trimmed_df_joints.columns:
                trimmed_df_joints = trimmed_df_joints.drop(columns=[feature])

        # Update actions
        trimmed_df_ee["action"] = [ee_actions[i].tolist() for i in frames_to_keep]
        trimmed_df_joints["action"] = [joint_actions[i].tolist() for i in frames_to_keep]

        # Store original joints in EE dataset for reference
        trimmed_df_ee["action.joints"] = [joint_actions[i].tolist() for i in frames_to_keep]

        # Update frame indices and timestamps
        trimmed_df_ee["frame_index"] = range(len(trimmed_df_ee))
        trimmed_df_joints["frame_index"] = range(len(trimmed_df_joints))
        trimmed_df_ee["timestamp"] = [i / fps for i in range(len(trimmed_df_ee))]
        trimmed_df_joints["timestamp"] = [i / fps for i in range(len(trimmed_df_joints))]

        # Write parquet files
        out_chunk_dir_ee = output_ee_path / "data" / chunk_dir
        out_chunk_dir_joints = output_joints_path / "data" / chunk_dir
        out_chunk_dir_ee.mkdir(parents=True, exist_ok=True)
        out_chunk_dir_joints.mkdir(parents=True, exist_ok=True)

        trimmed_df_ee.to_parquet(out_chunk_dir_ee / ep_file.name, index=False)
        trimmed_df_joints.to_parquet(out_chunk_dir_joints / ep_file.name, index=False)

    # Print trimming report
    if not args.no_trim:
        logging.info("")
        logging.info("=" * 60)
        logging.info("=== Trimming Report (based on EE pose deltas) ===")
        logging.info("=" * 60)

        total_original = 0
        total_trimmed = 0

        for ep_idx in sorted(episode_results.keys()):
            info_ep = episode_results[ep_idx]["trim_info"]
            orig = info_ep["original_length"]
            trimmed = info_ep["trimmed_length"]
            removed = orig - trimmed

            total_original += orig
            total_trimmed += trimmed

            segments = info_ep.get("idle_segments", [])
            details = []
            for seg in segments:
                details.append(f"{seg['type']}: -{seg['frames_removed']}")

            detail_str = ", ".join(details) if details else "no trimming"
            logging.info(f"Episode {ep_idx}: {orig} -> {trimmed} frames (-{removed}) [{detail_str}]")

        pct = 100.0 * (total_original - total_trimmed) / total_original if total_original > 0 else 0
        logging.info("")
        logging.info(f"Total: {total_original} -> {total_trimmed} frames ({pct:.1f}% trimmed)")
        logging.info("=" * 60)
        logging.info("")

    # Compute EE statistics
    all_ee_actions = np.array(all_ee_actions)
    ee_stats = {
        "mean": all_ee_actions.mean(axis=0).tolist(),
        "std": all_ee_actions.std(axis=0).tolist(),
        "min": all_ee_actions.min(axis=0).tolist(),
        "max": all_ee_actions.max(axis=0).tolist(),
    }

    logging.info("EE action statistics (from trimmed data):")
    logging.info(f"  mean: {[f'{x:.4f}' for x in ee_stats['mean']]}")
    logging.info(f"  std: {[f'{x:.4f}' for x in ee_stats['std']]}")

    # Update info.json for EE dataset
    info_ee = load_info(input_path).copy()
    info_ee["total_frames"] = len(all_ee_actions)
    info_ee["features"]["action"] = {
        "dtype": "float32",
        "shape": [EE_ACTION_DIM],
        "names": EE_FEATURE_NAMES,
    }
    info_ee["features"]["action.joints"] = {
        "dtype": "float32",
        "shape": [len(motor_names)],
        "names": motor_names,
    }
    # Remove depth features
    for feature in FEATURES_TO_REMOVE:
        if feature in info_ee["features"]:
            del info_ee["features"][feature]
            logging.info(f"Removed feature '{feature}' from EE dataset")
    # Update total_videos count if present (v2.1)
    if "total_videos" in info_ee:
        video_keys = [k for k, v in info_ee["features"].items() if v.get("dtype") == "video"]
        info_ee["total_videos"] = info_ee["total_episodes"] * len(video_keys)
    info_ee["conversion_info"] = {
        "source_dataset": str(input_path),
        "action_space": "ee",
        "urdf_path": args.urdf_path,
        "target_frame": args.target_frame,
        "trimmed": not args.no_trim,
        "trim_threshold_factor": args.trim_threshold_factor,
        "min_idle_segment": args.min_idle_segment,
        "ee_stats": ee_stats,
        "removed_features": FEATURES_TO_REMOVE,
    }
    save_info(output_ee_path, info_ee)

    # Update info.json for Joints dataset
    info_joints = load_info(input_path).copy()
    info_joints["total_frames"] = len(all_joint_actions)
    # Remove depth features
    for feature in FEATURES_TO_REMOVE:
        if feature in info_joints["features"]:
            del info_joints["features"][feature]
            logging.info(f"Removed feature '{feature}' from Joints dataset")
    # Update total_videos count if present (v2.1)
    if "total_videos" in info_joints:
        video_keys = [k for k, v in info_joints["features"].items() if v.get("dtype") == "video"]
        info_joints["total_videos"] = info_joints["total_episodes"] * len(video_keys)
    info_joints["conversion_info"] = {
        "source_dataset": str(input_path),
        "action_space": "joints",
        "trimmed": not args.no_trim,
        "trim_threshold_factor": args.trim_threshold_factor,
        "min_idle_segment": args.min_idle_segment,
        "removed_features": FEATURES_TO_REMOVE,
    }
    save_info(output_joints_path, info_joints)

    # Update stats.json for both datasets
    for out_path, action_stats in [(output_ee_path, ee_stats), (output_joints_path, None)]:
        stats_file = out_path / "meta" / "stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                stats = json.load(f)
            # Update action stats for EE dataset
            if action_stats is not None:
                stats["action"] = action_stats
            # Remove depth stats
            for feature in FEATURES_TO_REMOVE:
                if feature in stats:
                    del stats[feature]
            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=2)

    # Update episodes.jsonl
    new_episodes_ee = []
    new_episodes_joints = []
    for ep in episodes:
        ep_idx = ep["episode_index"]
        if ep_idx in episode_results:
            new_length = len(episode_results[ep_idx]["frames_to_keep"])
            new_episodes_ee.append({**ep, "length": new_length})
            new_episodes_joints.append({**ep, "length": new_length})

    save_episodes_v21(output_ee_path, new_episodes_ee)
    save_episodes_v21(output_joints_path, new_episodes_joints)

    # Update episodes_stats.jsonl with new action stats (critical for v2.1→v3.0 conversion)
    # The v2.1→v3.0 converter aggregates these per-episode stats to create stats.json
    logging.info("Computing per-episode stats for new action dimensions...")

    def compute_array_stats(arr: np.ndarray) -> dict:
        """Compute mean, std, min, max, count stats for an array.

        The 'count' field is required for weighted aggregation across episodes.
        All values must be numpy arrays (not Python lists) for the v2.1→v3.0 converter.
        """
        return {
            "mean": np.atleast_1d(arr.mean(axis=0)),
            "std": np.atleast_1d(arr.std(axis=0)),
            "min": np.atleast_1d(arr.min(axis=0)),
            "max": np.atleast_1d(arr.max(axis=0)),
            "count": np.array([len(arr)]),  # Shape (1,) as required
        }

    def convert_stats_to_lists(stats_dict: dict) -> dict:
        """Convert numpy arrays in stats to lists for JSON serialization.

        The v2.1→v3.0 converter will convert them back to numpy arrays.
        """
        result = {}
        for key, value in stats_dict.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, dict):
                result[key] = convert_stats_to_lists(value)
            else:
                result[key] = value
        return result

    # Build per-episode stats for EE dataset
    ee_episode_stats = []
    joints_episode_stats = []

    for ep_stat in episodes_stats:
        ep_idx = ep_stat["episode_index"]
        if ep_idx not in episode_results:
            continue

        # Start with existing stats, remove depth and old action stats
        ee_ep_stats = {}
        joints_ep_stats = {}

        if "stats" in ep_stat:
            for key, value in ep_stat["stats"].items():
                # Skip features to remove (depth)
                if key in FEATURES_TO_REMOVE:
                    continue
                # Skip old action stats (we'll recompute)
                if key == "action":
                    continue
                # Keep other stats (observation.state, etc.)
                ee_ep_stats[key] = value
                joints_ep_stats[key] = value

        # Compute new action stats from trimmed data
        ep_ee_actions = episode_results[ep_idx]["ee_actions"]
        ep_joint_actions = episode_results[ep_idx]["joint_actions"]

        # EE dataset: 7-dimensional action stats (convert numpy to lists for JSON)
        ee_ep_stats["action"] = convert_stats_to_lists(compute_array_stats(ep_ee_actions))
        # Also add action.joints stats for EE dataset
        ee_ep_stats["action.joints"] = convert_stats_to_lists(compute_array_stats(ep_joint_actions))

        # Joints dataset: original dimension action stats (but from trimmed frames)
        joints_ep_stats["action"] = convert_stats_to_lists(compute_array_stats(ep_joint_actions))

        ee_episode_stats.append({"episode_index": ep_idx, "stats": ee_ep_stats})
        joints_episode_stats.append({"episode_index": ep_idx, "stats": joints_ep_stats})

    save_episodes_stats_v21(output_ee_path, ee_episode_stats)
    save_episodes_stats_v21(output_joints_path, joints_episode_stats)

    # Process videos (skip depth videos) - with parallel processing
    videos_dir = input_path / "videos"
    if videos_dir.exists():
        logging.info("Trimming videos...")

        # Determine number of workers
        if num_workers == 0:
            num_workers = max(1, multiprocessing.cpu_count() - 1)

        for output_path in [output_ee_path, output_joints_path]:
            output_name = output_path.name
            logging.info(f"Processing videos for {output_name} (using {num_workers} workers)...")

            # Build list of video trimming tasks
            video_tasks = []
            for video_file in sorted(videos_dir.glob("**/*.mp4")):
                # Parse episode index from filename
                if not video_file.stem.startswith("episode_"):
                    continue

                # Skip depth videos (check if any parent directory matches a feature to remove)
                rel_path = video_file.relative_to(videos_dir)
                skip_video = False
                for feature in FEATURES_TO_REMOVE:
                    # v2.1 video path: chunk-XXX/<video_key>/episode_YYYYYY.mp4
                    # The video_key is typically the feature name (e.g., observation.images.front_depth)
                    if feature in str(rel_path):
                        skip_video = True
                        break
                if skip_video:
                    continue

                ep_idx = int(video_file.stem.split("_")[1])

                if ep_idx not in episode_results:
                    continue

                frames_to_keep = episode_results[ep_idx]["frames_to_keep"]

                # Determine output path
                output_video_path = output_path / "videos" / rel_path

                # Add task tuple (must be picklable - use strings for paths)
                video_tasks.append((
                    str(video_file),
                    str(output_video_path),
                    frames_to_keep,
                    str(rel_path),
                ))

            # Process videos in parallel
            if num_workers > 1 and len(video_tasks) > 1:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = {executor.submit(_trim_video_task, task): task for task in video_tasks}
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Trimming videos"):
                        rel_path_str, num_frames = future.result()
            else:
                # Sequential fallback
                for task in tqdm(video_tasks, desc="Trimming videos"):
                    _trim_video_task(task)

    return {
        "total_episodes": len(episode_results),
        "total_frames": len(all_ee_actions),
        "ee_stats": ee_stats,
        "motor_names": motor_names,
    }


# =============================================================================
# Main
# =============================================================================


def main():
    args = Args().parse_args()

    input_path = Path(args.input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    # Check dataset version
    info = load_info(input_path)
    version = info.get("codebase_version", "v2.1")

    # Determine output paths (same directory as input)
    dataset_name = input_path.name
    output_base = input_path.parent
    output_ee_path = output_base / f"{dataset_name}_ee"
    output_joints_path = output_base / f"{dataset_name}_joints"

    logging.info(f"Input dataset: {input_path}")
    logging.info(f"Dataset version: {version}")
    logging.info(f"Output EE dataset: {output_ee_path}")
    logging.info(f"Output Joints dataset: {output_joints_path}")

    # Get motor names from features
    action_feature = info.get("features", {}).get("action", {})
    motor_names = action_feature.get("names", [])
    if not motor_names:
        raise ValueError("Dataset does not have action names defined")

    logging.info(f"Motor names: {motor_names}")
    logging.info(f"Joint names (for FK): {JOINT_NAMES}")

    # Initialize kinematics
    logging.info(f"Initializing kinematics with URDF: {args.urdf_path}")
    kinematics = RobotKinematics(
        urdf_path=args.urdf_path,
        target_frame_name=args.target_frame,
        joint_names=JOINT_NAMES,
    )

    if version.startswith("v3"):
        # Use 3-step pipeline for v3.0 datasets
        logging.info("Using 3-step pipeline: v3.0 → v2.1 → process → v3.0")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Step 1: v3.0 → v2.1
            v21_path = temp_path / "v21_intermediate"
            logging.info("Step 1: Converting v3.0 → v2.1...")
            convert_v3_to_v21(input_path, v21_path)

            # Step 2: Process in v2.1 format
            v21_ee_path = temp_path / "v21_ee"
            v21_joints_path = temp_path / "v21_joints"
            logging.info("Step 2: Processing in v2.1 format...")
            result = process_v21_dataset(
                v21_path,
                v21_ee_path,
                v21_joints_path,
                kinematics,
                args,
                num_workers=args.num_workers,
            )

            # Step 3: v2.1 → v3.0
            logging.info("Step 3: Converting v2.1 → v3.0...")
            convert_v21_to_v30_local(v21_ee_path, output_ee_path)
            cleanup_removed_features_from_metadata(output_ee_path, FEATURES_TO_REMOVE)

            convert_v21_to_v30_local(v21_joints_path, output_joints_path)
            cleanup_removed_features_from_metadata(output_joints_path, FEATURES_TO_REMOVE)

    else:
        # v2.1 dataset - process directly and leave as v2.1
        logging.info("Processing v2.1 dataset directly...")
        result = process_v21_dataset(
            input_path,
            output_ee_path,
            output_joints_path,
            kinematics,
            args,
            num_workers=args.num_workers,
        )
        cleanup_removed_features_from_metadata(output_ee_path, FEATURES_TO_REMOVE)
        cleanup_removed_features_from_metadata(output_joints_path, FEATURES_TO_REMOVE)

    logging.info("")
    logging.info("=" * 60)
    logging.info("Preparation complete!")
    logging.info("=" * 60)
    logging.info(f"EE dataset:     {output_ee_path}")
    logging.info(f"  - action: {EE_FEATURE_NAMES}")
    logging.info(f"Joints dataset: {output_joints_path}")
    logging.info(f"  - action: {motor_names}")
    logging.info("")
    logging.info("Train with:")
    logging.info(f"  python src/lerobot/scripts/lerobot_train.py --dataset.root={output_ee_path} ...")
    logging.info(f"  python src/lerobot/scripts/lerobot_train.py --dataset.root={output_joints_path} ...")


if __name__ == "__main__":
    main()
