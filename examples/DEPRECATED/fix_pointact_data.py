#!/usr/bin/env python

"""
Fix already-converted PointAct datasets: convert EE orientation from axis-angle
to quaternion (w, x, y, z) and regenerate all metadata.

After running convert_to_pointact_format.py, the EE pose features use axis-angle
orientation (3 values). This script converts them to unit quaternions (4 values),
which changes the feature shapes:

  observation.state:           (7,) -> (8,)  [x, y, z, qw, qx, qy, qz, gripper]
  action:                      (7,) -> (8,)  [x, y, z, qw, qx, qy, qz, gripper]
  observation.states.ee_state: (6,) -> (7,)  [x, y, z, qw, qx, qy, qz]

It also fixes stale metadata:
  - stats.json: renames/removes old keys, recomputes stats on the new quaternion data
  - episodes_stats.jsonl: same per-episode
  - info.json: updates feature shapes and names

Quaternion convention: (w, x, y, z), right-handed, active rotations.


Usage:
```bash
python examples/post_process_dataset/fix_pointact_data.py --dataset_dir=$HOME/lerobot_datasets/put_socks_into_drawer_pointact
```

Multiple datasets:
```bash
python examples/post_process_dataset/fix_pointact_data.py --dataset_dir=$HOME/lerobot_datasets/ds1 --dataset_dir=$HOME/lerobot_datasets/ds2
```
"""

import json
import logging
from pathlib import Path

import jsonlines
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tap import Tap
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Old keys (pre-conversion, may still be in stats)
OLD_RGB_KEY = "observation.images.front"
OLD_DEPTH_KEY = "observation.images.front_depth"
OLD_POINT_CLOUD_KEY = "observation.point_cloud"

# New keys (post-conversion)
NEW_IMAGE_KEY = "observation.images.front_image"
NEW_POINT_CLOUD_KEY = "observation.points.frontview"

# EE-space features that contain axis-angle orientation
STATE_KEY = "observation.state"
ACTION_KEY = "action"
EE_STATE_KEY = "observation.states.ee_state"

# Features that are unchanged by this script (no axis-angle)
JOINT_STATE_KEY = "observation.states.joint_state"
GRIPPER_STATE_KEY = "observation.states.gripper_state"
JOINT_ACTION_KEY = "action.joints"

# All numeric feature keys that need stats recomputation
NUMERIC_KEYS = [STATE_KEY, ACTION_KEY, EE_STATE_KEY, JOINT_STATE_KEY, GRIPPER_STATE_KEY, JOINT_ACTION_KEY]

# Features whose parquet data is rewritten (axis-angle -> quaternion)
EE_FEATURES_WITH_GRIPPER = [STATE_KEY, ACTION_KEY]  # (7,) -> (8,)
EE_FEATURES_WITHOUT_GRIPPER = [EE_STATE_KEY]  # (6,) -> (7,)

QUAT_NAMES = ["qw", "qx", "qy", "qz"]


class Args(Tap):
    """Arguments for fixing PointAct dataset data and metadata."""

    dataset_dir: list[str]  # Path(s) to converted PointAct dataset(s)


def axis_angle_to_quat_wxyz(rotvec: np.ndarray) -> np.ndarray:
    """Convert an axis-angle vector to a unit quaternion (w, x, y, z).

    Args:
        rotvec: shape (3,) rotation vector; direction = axis, magnitude = angle (radians).

    Returns:
        shape (4,) quaternion [w, x, y, z], always with w >= 0.
    """
    angle = np.linalg.norm(rotvec)
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    axis = rotvec / angle
    half = angle / 2.0
    sin_half = np.sin(half)
    cos_half = np.cos(half)

    w, x, y, z = cos_half, axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half

    # Canonical form: enforce w >= 0
    if w < 0:
        w, x, y, z = -w, -x, -y, -z

    return np.array([w, x, y, z], dtype=np.float32)


def convert_ee_with_gripper(row: np.ndarray) -> np.ndarray:
    """Convert [x, y, z, ax, ay, az, gripper] -> [x, y, z, qw, qx, qy, qz, gripper]."""
    pos = row[:3]
    rotvec = row[3:6]
    gripper = row[6:]
    quat = axis_angle_to_quat_wxyz(rotvec)
    return np.concatenate([pos, quat, gripper]).astype(np.float32)


def convert_ee_without_gripper(row: np.ndarray) -> np.ndarray:
    """Convert [x, y, z, ax, ay, az] -> [x, y, z, qw, qx, qy, qz]."""
    pos = row[:3]
    rotvec = row[3:6]
    quat = axis_angle_to_quat_wxyz(rotvec)
    return np.concatenate([pos, quat]).astype(np.float32)


def convert_parquet_files(dataset_path: Path) -> None:
    """Rewrite parquet files, converting axis-angle columns to quaternion."""
    data_dir = dataset_path / "data"
    parquet_files = sorted(data_dir.glob("**/*.parquet"))

    for parquet_path in tqdm(parquet_files, desc="  Converting parquet files"):
        df = pd.read_parquet(parquet_path)

        for key in EE_FEATURES_WITH_GRIPPER:
            if key in df.columns:
                df[key] = df[key].apply(lambda v: convert_ee_with_gripper(np.asarray(v)))

        for key in EE_FEATURES_WITHOUT_GRIPPER:
            if key in df.columns:
                df[key] = df[key].apply(lambda v: convert_ee_without_gripper(np.asarray(v)))

        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, parquet_path)


def update_info(dataset_path: Path) -> None:
    """Update info.json feature definitions to reflect quaternion shapes/names."""
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    features = info.get("features", {})

    quat_names_with_gripper = ["x", "y", "z"] + QUAT_NAMES + ["gripper_openness"]
    quat_names_without_gripper = ["x", "y", "z"] + QUAT_NAMES

    if STATE_KEY in features:
        features[STATE_KEY]["shape"] = [8]
        features[STATE_KEY]["names"] = {"motors": quat_names_with_gripper}

    if ACTION_KEY in features:
        features[ACTION_KEY]["shape"] = [8]
        features[ACTION_KEY]["names"] = {"motors": quat_names_with_gripper}

    if EE_STATE_KEY in features:
        features[EE_STATE_KEY]["shape"] = [7]
        features[EE_STATE_KEY]["names"] = {"motors": quat_names_without_gripper}

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    logging.info("  Updated info.json")


def compute_array_stats(arr: np.ndarray) -> dict:
    """Compute statistics for a numpy array.

    Returns dict with min, max, mean, std, count, and quantiles.
    All values are plain Python lists for JSON serialization.
    """
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    return {
        "min": np.min(arr, axis=0).tolist(),
        "max": np.max(arr, axis=0).tolist(),
        "mean": np.mean(arr, axis=0).tolist(),
        "std": np.std(arr, axis=0).tolist(),
        "count": [int(len(arr))],
        "q01": np.percentile(arr, 1, axis=0).tolist(),
        "q10": np.percentile(arr, 10, axis=0).tolist(),
        "q50": np.percentile(arr, 50, axis=0).tolist(),
        "q90": np.percentile(arr, 90, axis=0).tolist(),
        "q99": np.percentile(arr, 99, axis=0).tolist(),
    }


def rename_and_clean_stats(stats: dict) -> dict:
    """Rename old keys and remove deleted features from a stats dict."""
    if OLD_RGB_KEY in stats:
        stats[NEW_IMAGE_KEY] = stats.pop(OLD_RGB_KEY)

    if OLD_POINT_CLOUD_KEY in stats:
        stats[NEW_POINT_CLOUD_KEY] = stats.pop(OLD_POINT_CLOUD_KEY)

    if OLD_DEPTH_KEY in stats:
        del stats[OLD_DEPTH_KEY]

    return stats


def read_parquet_data(data_dir: Path, episode_index: int | None = None) -> dict[str, list[np.ndarray]]:
    """Read numeric feature data from parquet files.

    Args:
        data_dir: Path to the data/ directory.
        episode_index: If provided, only read this episode. Otherwise read all.

    Returns:
        Dict mapping feature key to list of numpy arrays (one per parquet file).
    """
    if episode_index is not None:
        parquet_files = list(data_dir.glob(f"**/episode_{episode_index:06d}.parquet"))
    else:
        parquet_files = sorted(data_dir.glob("**/*.parquet"))

    feature_data: dict[str, list[np.ndarray]] = {k: [] for k in NUMERIC_KEYS}

    for parquet_path in parquet_files:
        df = pd.read_parquet(parquet_path)
        for key in NUMERIC_KEYS:
            if key in df.columns:
                values = np.array(df[key].tolist())
                feature_data[key].append(values)

    return feature_data


def fix_global_stats(dataset_path: Path) -> None:
    """Fix meta/stats.json by renaming keys and recomputing numeric stats."""
    stats_path = dataset_path / "meta" / "stats.json"
    if not stats_path.exists():
        logging.warning(f"  No stats.json found at {stats_path}, skipping")
        return

    with open(stats_path) as f:
        stats = json.load(f)

    stats = rename_and_clean_stats(stats)

    data_dir = dataset_path / "data"
    feature_data = read_parquet_data(data_dir)

    for key, data_list in feature_data.items():
        if data_list:
            all_data = np.concatenate(data_list, axis=0)
            stats[key] = compute_array_stats(all_data)
            logging.info(f"  Computed stats for: {key} (shape: {all_data.shape})")

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    logging.info(f"  Updated {stats_path}")


def fix_episodes_stats(dataset_path: Path) -> None:
    """Fix meta/episodes_stats.jsonl by renaming keys and recomputing per-episode stats."""
    episodes_stats_path = dataset_path / "meta" / "episodes_stats.jsonl"
    if not episodes_stats_path.exists():
        logging.warning(f"  No episodes_stats.jsonl found at {episodes_stats_path}, skipping")
        return

    with jsonlines.open(episodes_stats_path) as reader:
        all_episodes = list(reader)

    data_dir = dataset_path / "data"

    for entry in tqdm(all_episodes, desc="  Fixing per-episode stats"):
        ep_idx = entry["episode_index"]
        ep_stats = entry["stats"]

        ep_stats = rename_and_clean_stats(ep_stats)

        feature_data = read_parquet_data(data_dir, episode_index=ep_idx)

        for key, data_list in feature_data.items():
            if data_list:
                all_data = np.concatenate(data_list, axis=0)
                ep_stats[key] = compute_array_stats(all_data)

        entry["stats"] = ep_stats

    with jsonlines.open(episodes_stats_path, mode="w") as writer:
        for entry in all_episodes:
            writer.write(entry)

    logging.info(f"  Updated {episodes_stats_path}")


def fix_dataset(dataset_dir: str) -> None:
    """Fix data and metadata for a single converted PointAct dataset."""
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        logging.error(f"Dataset path does not exist: {dataset_path}")
        return

    logging.info(f"Fixing dataset: {dataset_path}")

    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        logging.error("  No info.json found, is this a valid dataset?")
        return

    with open(info_path) as f:
        info = json.load(f)

    features = info.get("features", {})
    if EE_STATE_KEY not in features:
        logging.warning(f"  {EE_STATE_KEY} not in info.json features, this may not be a PointAct dataset")

    # Step 1: Convert axis-angle -> quaternion in parquet data
    logging.info("  Step 1/4: Converting axis-angle to quaternion in parquet files...")
    convert_parquet_files(dataset_path)

    # Step 2: Update info.json feature definitions
    logging.info("  Step 2/4: Updating info.json...")
    update_info(dataset_path)

    # Step 3: Recompute global stats on the new quaternion data
    logging.info("  Step 3/4: Recomputing global stats...")
    fix_global_stats(dataset_path)

    # Step 4: Recompute per-episode stats on the new quaternion data
    logging.info("  Step 4/4: Recomputing per-episode stats...")
    fix_episodes_stats(dataset_path)

    # Verify alignment
    with open(info_path) as f:
        info = json.load(f)
    with open(dataset_path / "meta" / "stats.json") as f:
        stats = json.load(f)

    info_keys = set(info["features"].keys())
    stats_keys = set(stats.keys())
    missing = info_keys - stats_keys
    extra = stats_keys - info_keys - {"frame_index", "index", "timestamp", "episode_index", "task_index"}

    if missing:
        logging.warning(f"  Still missing from stats.json: {sorted(missing)}")
    if extra:
        logging.warning(f"  Unexpected extra keys in stats.json: {sorted(extra)}")

    logging.info(f"Done: {dataset_path}")
    logging.info("")


if __name__ == "__main__":
    args = Args().parse_args()

    for dataset_dir in args.dataset_dir:
        fix_dataset(dataset_dir)
