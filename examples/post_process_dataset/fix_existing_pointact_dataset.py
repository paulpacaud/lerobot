#!/usr/bin/env python3
"""
Fix an existing PointAct dataset that has two known issues:

1. Depth images not resized: Actual depth images are 640x480 but metadata claims 256x256.
   The depth images were correctly trimmed but never resized when stored as parquet images.

2. Stats counts incorrect: Metadata fields (index, timestamp, episode_index, frame_index,
   task_index) still show pre-trim counts instead of post-trim counts.

This script fixes both issues in-place on an already-converted PointAct dataset.

Usage:
    python -m examples.post_process_dataset.fix_existing_pointact_dataset --dataset_dir=/path/to/pointact_dataset
"""

import io
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tap import Tap
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Args(Tap):
    """Arguments for fixing an existing PointAct dataset."""

    dataset_dir: str  # Path to the PointAct dataset directory to fix
    image_size: int = 256  # Target image size (square)
    depth_key: str = "observation.images.front_depth"  # Depth column name in parquet


METADATA_FIELDS = ["index", "timestamp", "episode_index", "frame_index", "task_index"]


def resize_image_bytes(image_bytes: bytes, target_size: int) -> bytes:
    """Resize image bytes to target square size."""
    img = Image.open(io.BytesIO(image_bytes))
    img_resized = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img_resized.save(buffer, format="PNG")
    return buffer.getvalue()


def compute_array_stats(arr: np.ndarray) -> dict:
    """Compute statistics for a numpy array."""
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


def fix_depth_images(dataset_path: Path, depth_key: str, target_size: int) -> int:
    """Resize depth images stored in parquet files to target size.

    Returns:
        Total number of images resized.
    """
    data_dir = dataset_path / "data"
    parquet_files = sorted(data_dir.glob("**/*.parquet"))

    if not parquet_files:
        logging.warning("No parquet files found")
        return 0

    # Check if depth column exists and needs resizing
    sample_df = pd.read_parquet(parquet_files[0])
    if depth_key not in sample_df.columns:
        logging.info(f"No depth column '{depth_key}' found in parquet files, skipping depth fix")
        return 0

    # Check current size of first depth image
    first_entry = sample_df[depth_key].iloc[0]
    if isinstance(first_entry, dict) and "bytes" in first_entry:
        img = Image.open(io.BytesIO(first_entry["bytes"]))
        current_size = img.size
        if current_size == (target_size, target_size):
            logging.info(f"Depth images already {target_size}x{target_size}, skipping resize")
            return 0
        logging.info(f"Depth images are {current_size[0]}x{current_size[1]}, resizing to {target_size}x{target_size}")
    else:
        logging.info("Depth data is not in image (struct<bytes, path>) format, skipping")
        return 0

    total_resized = 0
    for parquet_path in tqdm(parquet_files, desc="Resizing depth images"):
        df = pd.read_parquet(parquet_path)
        if depth_key not in df.columns:
            continue

        new_depth_entries = []
        for entry in df[depth_key]:
            if entry is None:
                new_depth_entries.append(entry)
                continue

            if isinstance(entry, dict) and "bytes" in entry:
                resized_bytes = resize_image_bytes(entry["bytes"], target_size)
                new_depth_entries.append({**entry, "bytes": resized_bytes})
                total_resized += 1
            else:
                new_depth_entries.append(entry)

        df[depth_key] = new_depth_entries
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, parquet_path)

    return total_resized


def fix_stats(dataset_path: Path) -> None:
    """Recompute metadata field stats with correct counts from actual parquet data."""
    stats_path = dataset_path / "meta" / "stats.json"
    if not stats_path.exists():
        logging.warning("No stats.json found, skipping stats fix")
        return

    with open(stats_path) as f:
        stats = json.load(f)

    # Check if any metadata field has wrong count
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    expected_count = info.get("total_frames", 0)

    needs_fix = False
    for field in METADATA_FIELDS:
        if field in stats and "count" in stats[field]:
            current_count = stats[field]["count"][0]
            if current_count != expected_count:
                logging.info(
                    f"  Stats count mismatch for '{field}': "
                    f"got {current_count}, expected {expected_count}"
                )
                needs_fix = True

    if not needs_fix:
        logging.info("Stats counts are already correct, skipping stats fix")
        return

    # Read all parquet files and recompute metadata stats
    data_dir = dataset_path / "data"
    parquet_files = sorted(data_dir.glob("**/*.parquet"))

    feature_data = {field: [] for field in METADATA_FIELDS}

    for parquet_path in tqdm(parquet_files, desc="Reading parquet for stats"):
        df = pd.read_parquet(parquet_path)
        for field in METADATA_FIELDS:
            if field in df.columns:
                values = np.array(df[field].tolist())
                feature_data[field].append(values)

    # Recompute and update stats for metadata fields
    for field, data_list in feature_data.items():
        if data_list:
            all_data = np.concatenate(data_list, axis=0)
            stats[field] = compute_array_stats(all_data)
            logging.info(f"  Recomputed stats for '{field}' (count: {len(all_data)})")

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    logging.info("Stats fix complete")


def fix_dataset(dataset_dir: str, image_size: int = 256, depth_key: str = "observation.images.front_depth") -> None:
    """Fix an existing PointAct dataset in-place.

    Fixes:
    1. Resize depth images in parquet files to target size
    2. Recompute metadata stats with correct counts
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    logging.info(f"Fixing PointAct dataset: {dataset_path}")

    # Fix 1: Resize depth images
    logging.info("=" * 60)
    logging.info("FIX 1: Resizing depth images in parquet files")
    logging.info("=" * 60)
    total_resized = fix_depth_images(dataset_path, depth_key, image_size)
    logging.info(f"Resized {total_resized} depth images")

    # Fix 2: Recompute metadata stats
    logging.info("=" * 60)
    logging.info("FIX 2: Recomputing metadata stats")
    logging.info("=" * 60)
    fix_stats(dataset_path)

    logging.info("=" * 60)
    logging.info("ALL FIXES APPLIED SUCCESSFULLY")
    logging.info("=" * 60)


if __name__ == "__main__":
    args = Args().parse_args()
    fix_dataset(
        dataset_dir=args.dataset_dir,
        image_size=args.image_size,
        depth_key=args.depth_key,
    )
