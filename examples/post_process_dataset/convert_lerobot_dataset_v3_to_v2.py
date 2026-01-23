"""
This script converts a LeRobot dataset from codebase version 3.0 to 2.1.

It will:
- Split concatenated data files into per-episode parquet files
- Split concatenated video files into per-episode mp4 files
- Convert tasks.parquet to tasks.jsonl
- Convert episodes metadata to episodes.jsonl and episodes_stats.jsonl
- Update info.json with v2.1 format

Usage:

Convert a local dataset:
```bash
python examples/post_process_dataset/convert_lerobot_dataset_v3_to_v2.py \
    --input-dir=/path/to/v3/dataset \
    --output-dir=/path/to/output
```

Example:
```bash
python examples/post_process_dataset/convert_lerobot_dataset_v3_to_v2.py \
    --input-dir=$HOME/lerobot_datasets/depth_test \
    --output-dir=$HOME/lerobot_datasets/depth_test_v2
```
"""

import json
import logging
import shutil
import subprocess
from pathlib import Path

import jsonlines
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, Features, Image
from tap import Tap
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Args(Tap):
    """Arguments for converting LeRobot dataset from v3.0 to v2.1 format."""

    input_dir: str  # Path to the input v3.0 dataset directory
    output_dir: str  # Path to the output v2.1 dataset directory

V21 = "v2.1"
V30 = "v3.0"

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


def load_tasks_v3(root: Path) -> pd.DataFrame:
    """Load tasks from v3 parquet format."""
    tasks_path = root / "meta" / "tasks.parquet"
    return pd.read_parquet(tasks_path)


def load_episodes_v3(root: Path) -> list[dict]:
    """Load episodes metadata from v3 parquet format."""
    episodes_dir = root / "meta" / "episodes"
    all_episodes = []
    for parquet_file in sorted(episodes_dir.glob("*/*.parquet")):
        df = pd.read_parquet(parquet_file)
        for _, row in df.iterrows():
            all_episodes.append(row.to_dict())
    return sorted(all_episodes, key=lambda x: x["episode_index"])


def get_video_keys(info: dict) -> list[str]:
    """Get video keys from features."""
    return [key for key, ft in info["features"].items() if ft["dtype"] == "video"]


def get_image_keys(info: dict) -> list[str]:
    """Get image keys from features."""
    return [key for key, ft in info["features"].items() if ft["dtype"] == "image"]


def convert_info(input_root: Path, output_root: Path) -> dict:
    """Convert info.json from v3 to v2.1 format."""
    info = load_info(input_root)

    # Update version
    info["codebase_version"] = V21

    # Add v2.1 specific fields
    info["total_chunks"] = (info["total_episodes"] + DEFAULT_CHUNK_SIZE - 1) // DEFAULT_CHUNK_SIZE
    info["total_videos"] = info["total_episodes"] * len(get_video_keys(info))

    # Remove v3 specific fields if present
    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)

    # Update paths to v2.1 format
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"

    video_keys = get_video_keys(info)
    if video_keys:
        info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    else:
        info["video_path"] = None

    # Remove fps from individual features (v3 adds this)
    for key in info["features"]:
        info["features"][key].pop("fps", None)

    # Write info.json
    output_meta = output_root / "meta"
    output_meta.mkdir(parents=True, exist_ok=True)
    with open(output_meta / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    logging.info(f"Converted info.json to v2.1 format")
    return info


def convert_tasks(input_root: Path, output_root: Path):
    """Convert tasks.parquet to tasks.jsonl."""
    tasks_df = load_tasks_v3(input_root)

    output_meta = output_root / "meta"
    output_meta.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(output_meta / "tasks.jsonl", "w") as writer:
        for task_str, row in tasks_df.iterrows():
            writer.write({
                "task_index": int(row["task_index"]),
                "task": task_str
            })

    logging.info(f"Converted tasks to tasks.jsonl ({len(tasks_df)} tasks)")


def unflatten_dict(d: dict, sep: str = "/") -> dict:
    """Unflatten a dictionary with delimited keys into a nested dictionary."""
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def convert_numpy_to_list(obj):
    """Recursively convert numpy arrays and types to Python lists and types."""
    if isinstance(obj, np.ndarray):
        # Convert array to list and recursively process elements
        return convert_numpy_to_list(obj.tolist())
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_list(item) for item in obj]
    return obj


def convert_episodes(input_root: Path, output_root: Path):
    """Convert episodes metadata to episodes.jsonl and episodes_stats.jsonl."""
    episodes = load_episodes_v3(input_root)

    output_meta = output_root / "meta"
    output_meta.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(output_meta / "episodes.jsonl", "w") as ep_writer, \
         jsonlines.open(output_meta / "episodes_stats.jsonl", "w") as stats_writer:

        for ep in episodes:
            # Extract episode basic info - handle numpy types
            tasks = ep["tasks"]
            if isinstance(tasks, np.ndarray):
                tasks = tasks.tolist()
            elif not isinstance(tasks, list):
                tasks = [tasks]

            ep_info = {
                "episode_index": int(ep["episode_index"]),
                "tasks": tasks,
                "length": int(ep["length"])
            }
            ep_writer.write(ep_info)

            # Extract stats from the episode
            stats = {}
            for key, value in ep.items():
                if key.startswith("stats/"):
                    stats[key] = value

            if stats:
                # Unflatten the stats dictionary
                unflattened = unflatten_dict(stats)
                ep_stats = unflattened.get("stats", {})
                # Convert numpy types to Python types
                ep_stats = convert_numpy_to_list(ep_stats)
                stats_writer.write({
                    "episode_index": int(ep["episode_index"]),
                    "stats": ep_stats
                })

    logging.info(f"Converted episodes to episodes.jsonl and episodes_stats.jsonl ({len(episodes)} episodes)")


def convert_data(input_root: Path, output_root: Path, info: dict):
    """Convert concatenated data parquet files to per-episode parquet files."""
    data_dir = input_root / "data"
    episodes = load_episodes_v3(input_root)

    image_keys = get_image_keys(info)

    # Load all data
    all_data_files = sorted(data_dir.glob("*/*.parquet"))
    if not all_data_files:
        logging.warning("No data files found")
        return

    # Read all data into a single dataframe
    dfs = [pd.read_parquet(f) for f in all_data_files]
    full_df = pd.concat(dfs, ignore_index=True)

    logging.info(f"Loaded {len(full_df)} frames from {len(all_data_files)} data files")

    for ep in tqdm(episodes, desc="Converting data files"):
        ep_idx = ep["episode_index"]

        # Extract episode data
        ep_df = full_df[full_df["episode_index"] == ep_idx].copy()

        # Determine chunk
        chunk_idx = ep_idx // DEFAULT_CHUNK_SIZE

        # Output path
        output_path = output_root / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write parquet - for image columns the data is already stored as dicts with 'bytes' and 'path'
        # which is the correct v2.1 format
        ep_df.to_parquet(output_path, index=False)

    logging.info(f"Converted data to per-episode parquet files")


def get_video_timestamp_range(video_path: Path) -> float:
    """Get the duration of a video file using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def extract_video_segment(input_path: Path, output_path: Path, start_time: float, end_time: float):
    """Extract a segment from a video file using ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = end_time - start_time

    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ss", str(start_time), "-t", str(duration),
        "-c", "copy",  # Copy codec without re-encoding
        str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Try with re-encoding if copy fails
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-ss", str(start_time), "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast",
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)


def convert_videos(input_root: Path, output_root: Path, info: dict):
    """Convert concatenated video files to per-episode video files."""
    video_keys = get_video_keys(info)
    if not video_keys:
        logging.info("No video keys found, skipping video conversion")
        return

    episodes = load_episodes_v3(input_root)

    for video_key in video_keys:
        logging.info(f"Converting videos for {video_key}")

        # Track which input video files we've processed
        for ep in tqdm(episodes, desc=f"Converting {video_key}"):
            ep_idx = ep["episode_index"]
            chunk_idx = ep_idx // DEFAULT_CHUNK_SIZE

            # Get video file info from episode metadata
            video_chunk_idx = ep.get(f"videos/{video_key}/chunk_index", 0)
            video_file_idx = ep.get(f"videos/{video_key}/file_index", 0)
            from_timestamp = ep.get(f"videos/{video_key}/from_timestamp", 0.0)
            to_timestamp = ep.get(f"videos/{video_key}/to_timestamp", 0.0)

            # Input video path (v3 format)
            input_video = input_root / "videos" / video_key / f"chunk-{video_chunk_idx:03d}" / f"file-{video_file_idx:03d}.mp4"

            if not input_video.exists():
                logging.warning(f"Video file not found: {input_video}")
                continue

            # Output video path (v2.1 format)
            output_video = output_root / "videos" / f"chunk-{chunk_idx:03d}" / video_key / f"episode_{ep_idx:06d}.mp4"

            # Extract video segment
            extract_video_segment(input_video, output_video, from_timestamp, to_timestamp)

    logging.info("Converted videos to per-episode mp4 files")


def copy_stats(input_root: Path, output_root: Path):
    """Copy stats.json to output directory."""
    input_stats = input_root / "meta" / "stats.json"
    output_meta = output_root / "meta"
    output_meta.mkdir(parents=True, exist_ok=True)

    if input_stats.exists():
        shutil.copy(input_stats, output_meta / "stats.json")
        logging.info("Copied stats.json")


def convert_dataset(input_dir: str, output_dir: str):
    """Convert a LeRobot dataset from v3.0 to v2.1 format."""
    input_root = Path(input_dir)
    output_root = Path(output_dir)

    # Validate input
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory not found: {input_root}")

    info = load_info(input_root)
    if info.get("codebase_version") != V30:
        raise ValueError(f"Expected v3.0 dataset, got {info.get('codebase_version')}")

    # Clean output directory if exists
    if output_root.exists():
        logging.warning(f"Output directory exists, removing: {output_root}")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    # Convert components
    logging.info(f"Converting dataset from {input_root} to {output_root}")

    converted_info = convert_info(input_root, output_root)
    convert_tasks(input_root, output_root)
    convert_episodes(input_root, output_root)
    copy_stats(input_root, output_root)
    convert_data(input_root, output_root, converted_info)
    convert_videos(input_root, output_root, converted_info)

    logging.info(f"Successfully converted dataset to v2.1 format at {output_root}")


if __name__ == "__main__":
    args = Args().parse_args()
    convert_dataset(args.input_dir, args.output_dir)
