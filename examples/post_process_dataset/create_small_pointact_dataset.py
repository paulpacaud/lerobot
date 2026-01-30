#!/usr/bin/env python

"""
Create a small subset of a PointAct dataset by keeping only the first N episodes.

This script copies the first N episodes (default 2) from a PointAct dataset,
producing a new dataset at `<dataset_dir>_small` with correct metadata, stats,
parquet files, videos, and LMDB point clouds.

Usage:
```bash
python examples/post_process_dataset/create_small_pointact_dataset.py --dataset_dir=/home/ppacaud/lerobot_datasets/stack_cups_pointact --num_episodes=2
```
"""

import json
import logging
import shutil
from pathlib import Path

import jsonlines
import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import pandas as pd
from tap import Tap
from tqdm import tqdm

from examples.post_process_dataset.convert_to_pointact_format import (
    compute_array_stats,
    load_info,
    save_info,
    save_stats,
)

msgpack_numpy.patch()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Features stored in parquet (non-video, non-point-cloud)
PARQUET_FEATURE_KEYS = [
    "observation.state",
    "observation.states.ee_state",
    "observation.states.joint_state",
    "observation.states.gripper_state",
    "action",
    "action.joints",
]


class Args(Tap):
    """Arguments for creating a small PointAct dataset."""

    dataset_dir: str  # Path to the source PointAct dataset directory
    num_episodes: int = 2  # Number of episodes to keep


def copy_parquet_files(src_dir: Path, dst_dir: Path, num_episodes: int) -> None:
    """Copy the first num_episodes parquet files."""
    src_data = src_dir / "data" / "chunk-000"
    dst_data = dst_dir / "data" / "chunk-000"
    dst_data.mkdir(parents=True, exist_ok=True)

    for ep_idx in range(num_episodes):
        filename = f"episode_{ep_idx:06d}.parquet"
        src_file = src_data / filename
        if src_file.exists():
            shutil.copy2(src_file, dst_data / filename)
            logging.info(f"Copied {filename}")
        else:
            logging.warning(f"Parquet file not found: {src_file}")


def copy_video_files(src_dir: Path, dst_dir: Path, num_episodes: int) -> None:
    """Copy the first num_episodes video files for each video key."""
    src_videos = src_dir / "videos" / "chunk-000"
    if not src_videos.exists():
        logging.info("No videos directory found, skipping video copy")
        return

    for video_key_dir in sorted(src_videos.iterdir()):
        if not video_key_dir.is_dir():
            continue

        dst_video_dir = dst_dir / "videos" / "chunk-000" / video_key_dir.name
        dst_video_dir.mkdir(parents=True, exist_ok=True)

        for ep_idx in range(num_episodes):
            filename = f"episode_{ep_idx:06d}.mp4"
            src_file = video_key_dir / filename
            if src_file.exists():
                shutil.copy2(src_file, dst_video_dir / filename)
                logging.info(f"Copied video {video_key_dir.name}/{filename}")
            else:
                logging.warning(f"Video file not found: {src_file}")


def subset_lmdb(src_dir: Path, dst_dir: Path, num_episodes: int) -> None:
    """Copy LMDB entries for the first num_episodes only."""
    src_lmdb = src_dir / "point_clouds"
    if not src_lmdb.exists():
        logging.info("No LMDB point clouds found, skipping")
        return

    dst_lmdb = dst_dir / "point_clouds"
    if dst_lmdb.exists():
        shutil.rmtree(dst_lmdb)

    src_env = lmdb.open(str(src_lmdb), readonly=True, lock=False)
    dst_env = lmdb.open(str(dst_lmdb), map_size=src_env.info()["map_size"])

    entries_copied = 0
    entries_skipped = 0

    with src_env.begin() as src_txn:
        with dst_env.begin(write=True) as dst_txn:
            cursor = src_txn.cursor()
            for key, value in tqdm(cursor, desc="Subsetting LMDB"):
                key_str = key.decode("ascii")
                parts = key_str.split("-")
                if len(parts) != 2:
                    # Unknown key format, skip
                    entries_skipped += 1
                    continue

                ep_idx = int(parts[0])
                if ep_idx < num_episodes:
                    dst_txn.put(key, value)
                    entries_copied += 1
                else:
                    entries_skipped += 1

    src_env.close()
    dst_env.close()

    logging.info(f"LMDB subset: {entries_copied} entries copied, {entries_skipped} skipped")


def update_metadata(src_dir: Path, dst_dir: Path, num_episodes: int) -> None:
    """Copy and update metadata files for the subset."""
    dst_meta = dst_dir / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)

    # --- episodes.jsonl: keep first num_episodes lines ---
    src_episodes_path = src_dir / "meta" / "episodes.jsonl"
    dst_episodes_path = dst_meta / "episodes.jsonl"
    episodes_kept = []
    with jsonlines.open(src_episodes_path) as reader:
        for ep in reader:
            if ep["episode_index"] < num_episodes:
                episodes_kept.append(ep)
    with jsonlines.open(dst_episodes_path, mode="w") as writer:
        for ep in episodes_kept:
            writer.write(ep)
    logging.info(f"Wrote {len(episodes_kept)} episodes to episodes.jsonl")

    total_frames = sum(ep["length"] for ep in episodes_kept)

    # --- episodes_stats.jsonl: keep first num_episodes lines ---
    src_ep_stats_path = src_dir / "meta" / "episodes_stats.jsonl"
    dst_ep_stats_path = dst_meta / "episodes_stats.jsonl"
    if src_ep_stats_path.exists():
        ep_stats_kept = []
        with jsonlines.open(src_ep_stats_path) as reader:
            for ep_stat in reader:
                if ep_stat["episode_index"] < num_episodes:
                    ep_stats_kept.append(ep_stat)
        with jsonlines.open(dst_ep_stats_path, mode="w") as writer:
            for ep_stat in ep_stats_kept:
                writer.write(ep_stat)
        logging.info(f"Wrote {len(ep_stats_kept)} entries to episodes_stats.jsonl")

    # --- tasks.jsonl: copy as-is ---
    src_tasks = src_dir / "meta" / "tasks.jsonl"
    if src_tasks.exists():
        shutil.copy2(src_tasks, dst_meta / "tasks.jsonl")
        logging.info("Copied tasks.jsonl")

    # --- info.json: update counts ---
    info = load_info(src_dir)
    info["total_episodes"] = num_episodes
    info["total_frames"] = total_frames
    info["total_videos"] = num_episodes
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{num_episodes}"}
    save_info(dst_dir, info)
    logging.info(
        f"Updated info.json: total_episodes={num_episodes}, "
        f"total_frames={total_frames}, splits=0:{num_episodes}"
    )

    # --- .gitattributes: copy as-is ---
    src_gitattr = src_dir / ".gitattributes"
    if src_gitattr.exists():
        shutil.copy2(src_gitattr, dst_dir / ".gitattributes")
        logging.info("Copied .gitattributes")


def recompute_stats(dst_dir: Path, num_episodes: int) -> None:
    """Recompute stats.json from the kept parquet files."""
    logging.info("Recomputing stats.json from kept episodes...")

    # Collect data from all kept parquet files
    feature_data: dict[str, list[np.ndarray]] = {key: [] for key in PARQUET_FEATURE_KEYS}
    # Also collect scalar metadata columns
    scalar_keys = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
    scalar_data: dict[str, list[np.ndarray]] = {key: [] for key in scalar_keys}

    data_dir = dst_dir / "data" / "chunk-000"
    for ep_idx in range(num_episodes):
        parquet_path = data_dir / f"episode_{ep_idx:06d}.parquet"
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path)

        for key in PARQUET_FEATURE_KEYS:
            if key in df.columns:
                values = np.array(df[key].tolist())
                feature_data[key].append(values)

        for key in scalar_keys:
            if key in df.columns:
                values = np.array(df[key].tolist()).reshape(-1, 1)
                scalar_data[key].append(values)

    stats = {}

    # Compute stats for array features
    for key, data_list in feature_data.items():
        if data_list:
            all_data = np.concatenate(data_list, axis=0)
            stats[key] = compute_array_stats(all_data)
            logging.info(f"  Computed stats for {key} (shape: {all_data.shape})")

    # Compute stats for scalar features
    for key, data_list in scalar_data.items():
        if data_list:
            all_data = np.concatenate(data_list, axis=0)
            stats[key] = compute_array_stats(all_data)
            logging.info(f"  Computed stats for {key} (shape: {all_data.shape})")

    # Aggregate image stats from episodes_stats.jsonl
    ep_stats_path = dst_dir / "meta" / "episodes_stats.jsonl"
    if ep_stats_path.exists():
        image_key = "observation.images.front_image"
        image_stats_list = []
        with jsonlines.open(ep_stats_path) as reader:
            for ep_stat in reader:
                if image_key in ep_stat.get("stats", {}):
                    image_stats_list.append(ep_stat["stats"][image_key])

        if image_stats_list:
            # Aggregate: min of mins, max of maxes, mean of means
            agg_min = image_stats_list[0]["min"]
            agg_max = image_stats_list[0]["max"]

            # Weighted mean by count
            counts = [s["count"][0] for s in image_stats_list]
            total_count = sum(counts)

            # Initialize mean accumulator from first entry
            agg_mean = _scale_nested(image_stats_list[0]["mean"], counts[0] / total_count)

            for i, s in enumerate(image_stats_list):
                if i == 0:
                    continue
                agg_min = _nested_min(agg_min, s["min"])
                agg_max = _nested_max(agg_max, s["max"])
                agg_mean = _nested_add(agg_mean, _scale_nested(s["mean"], counts[i] / total_count))

            stats[image_key] = {
                "min": agg_min,
                "max": agg_max,
                "mean": agg_mean,
                "std": image_stats_list[0]["std"],  # not trivially aggregable, use first
                "count": [total_count],
            }
            # Copy quantiles from first episode as approximation
            for qkey in ["q01", "q10", "q50", "q90", "q99"]:
                if qkey in image_stats_list[0]:
                    stats[image_key][qkey] = image_stats_list[0][qkey]

            logging.info(f"  Aggregated image stats for {image_key}")

    save_stats(dst_dir, stats)
    logging.info("Stats recomputation complete!")


def _nested_min(a, b):
    """Element-wise min for nested lists."""
    if isinstance(a, list):
        return [_nested_min(x, y) for x, y in zip(a, b)]
    return min(a, b)


def _nested_max(a, b):
    """Element-wise max for nested lists."""
    if isinstance(a, list):
        return [_nested_max(x, y) for x, y in zip(a, b)]
    return max(a, b)


def _scale_nested(a, factor):
    """Scale all leaf values in nested list by factor."""
    if isinstance(a, list):
        return [_scale_nested(x, factor) for x in a]
    return a * factor


def _nested_add(a, b):
    """Element-wise add for nested lists."""
    if isinstance(a, list):
        return [_nested_add(x, y) for x, y in zip(a, b)]
    return a + b


def create_small_dataset(dataset_dir: str, num_episodes: int) -> None:
    """Create a small subset of a PointAct dataset."""
    src_dir = Path(dataset_dir)
    dst_dir = Path(f"{dataset_dir}_small")

    if not src_dir.exists():
        raise FileNotFoundError(f"Source dataset not found: {src_dir}")

    if dst_dir.exists():
        logging.warning(f"Output directory exists, removing: {dst_dir}")
        shutil.rmtree(dst_dir)

    dst_dir.mkdir(parents=True)

    logging.info(f"Source: {src_dir}")
    logging.info(f"Output: {dst_dir}")
    logging.info(f"Keeping {num_episodes} episodes")

    # 1. Copy parquet files
    logging.info("=== Copying parquet files ===")
    copy_parquet_files(src_dir, dst_dir, num_episodes)

    # 2. Copy video files
    logging.info("=== Copying video files ===")
    copy_video_files(src_dir, dst_dir, num_episodes)

    # 3. Subset LMDB point clouds
    logging.info("=== Subsetting LMDB point clouds ===")
    subset_lmdb(src_dir, dst_dir, num_episodes)

    # 4. Update metadata
    logging.info("=== Updating metadata ===")
    update_metadata(src_dir, dst_dir, num_episodes)

    # 5. Recompute stats
    logging.info("=== Recomputing stats ===")
    recompute_stats(dst_dir, num_episodes)

    logging.info(f"Done! Small dataset created at: {dst_dir}")


if __name__ == "__main__":
    args = Args().parse_args()
    create_small_dataset(
        dataset_dir=args.dataset_dir,
        num_episodes=args.num_episodes,
    )
