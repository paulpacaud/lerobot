#!/usr/bin/env python
"""
Analyze depth data consistency across frames in a recorded dataset.

This script loads depth data from parquet files and checks for:
1. Sudden changes in depth statistics between frames
2. Patterns that indicate desync (e.g., depth jumps)

Usage:
    python examples/debug_camera_depth/analyze_dataset_depth.py ~/lerobot_datasets/depth_test
    python examples/debug_camera_depth/analyze_dataset_depth.py ~/lerobot_datasets/depth_test --episode 0
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def load_depth_from_parquet(dataset_path: Path, episode_idx: int = 0) -> list[dict]:
    """Load depth data from parquet files for an episode."""
    data_dir = dataset_path / "data"

    # Find parquet files (may be in chunk subdirectories)
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        # Try chunk directories
        parquet_files = sorted(data_dir.glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    # Load first file to check schema
    table = pq.read_table(parquet_files[0])
    depth_cols = [col for col in table.column_names if 'depth' in col.lower()]

    if not depth_cols:
        raise ValueError(f"No depth columns found. Available: {table.column_names}")

    depth_col = depth_cols[0]
    print(f"Using depth column: {depth_col}")

    # Load all data for the episode
    frames_data = []

    for pq_file in parquet_files:
        table = pq.read_table(pq_file)
        df = table.to_pandas()

        # Filter by episode
        if 'episode_index' in df.columns:
            df = df[df['episode_index'] == episode_idx]

        for idx, row in df.iterrows():
            frame_idx = row.get('frame_index', idx)
            depth_data = row[depth_col]

            # Handle different depth data formats
            if isinstance(depth_data, dict) and 'bytes' in depth_data:
                # Embedded PNG bytes - decode from memory
                from PIL import Image
                import io
                depth = np.array(Image.open(io.BytesIO(depth_data['bytes'])))
            elif isinstance(depth_data, dict) and 'path' in depth_data:
                # Load from image file
                from PIL import Image
                img_path = dataset_path / depth_data['path']
                if img_path.exists():
                    depth = np.array(Image.open(img_path))
                else:
                    print(f"Warning: depth image not found: {img_path}")
                    continue
            elif isinstance(depth_data, np.ndarray):
                depth = depth_data
            else:
                print(f"Warning: unknown depth format at frame {frame_idx}: {type(depth_data)}")
                continue

            # Squeeze if needed
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[:, :, 0]

            # Compute statistics
            valid_depth = depth[depth > 0]
            if len(valid_depth) > 0:
                frame_info = {
                    'frame_idx': frame_idx,
                    'depth_mean': valid_depth.mean(),
                    'depth_std': valid_depth.std(),
                    'depth_min': valid_depth.min(),
                    'depth_max': valid_depth.max(),
                    'valid_pct': 100 * len(valid_depth) / depth.size,
                }
            else:
                frame_info = {
                    'frame_idx': frame_idx,
                    'depth_mean': 0,
                    'depth_std': 0,
                    'depth_min': 0,
                    'depth_max': 0,
                    'valid_pct': 0,
                }

            frames_data.append(frame_info)

    return sorted(frames_data, key=lambda x: x['frame_idx'])


def analyze_depth_consistency(frames_data: list[dict]) -> None:
    """Analyze depth consistency across frames."""
    if not frames_data:
        print("No frames to analyze")
        return

    df = pd.DataFrame(frames_data)

    print(f"\n{'='*60}")
    print("DEPTH DATA ANALYSIS")
    print(f"{'='*60}")
    print(f"Total frames: {len(df)}")
    print(f"Depth mean range: {df['depth_mean'].min():.1f} - {df['depth_mean'].max():.1f} mm")
    print(f"Depth std range: {df['depth_std'].min():.1f} - {df['depth_std'].max():.1f} mm")
    print(f"Valid depth %: {df['valid_pct'].min():.1f}% - {df['valid_pct'].max():.1f}%")

    # Detect sudden changes
    df['depth_change'] = df['depth_mean'].diff().abs()
    large_changes = df[df['depth_change'] > 100]  # >100mm change

    print(f"\n{'='*60}")
    print("LARGE DEPTH CHANGES (>100mm between consecutive frames)")
    print(f"{'='*60}")

    if len(large_changes) > 0:
        print(f"Found {len(large_changes)} large changes:")
        for _, row in large_changes.iterrows():
            prev_idx = int(row['frame_idx']) - 1
            prev_row = df[df['frame_idx'] == prev_idx]
            if not prev_row.empty:
                prev_mean = prev_row.iloc[0]['depth_mean']
                print(f"  Frame {prev_idx} -> {int(row['frame_idx'])}: "
                      f"{prev_mean:.1f}mm -> {row['depth_mean']:.1f}mm "
                      f"(change: {row['depth_change']:.1f}mm)")
    else:
        print("No large depth changes detected - depth is consistent!")

    # Show frame-by-frame stats for key frames
    print(f"\n{'='*60}")
    print("FRAME-BY-FRAME DEPTH STATS")
    print(f"{'='*60}")
    print(f"{'Frame':<8} {'Mean (mm)':<12} {'Std (mm)':<12} {'Valid %':<10}")
    print("-" * 42)

    # Show every 25th frame
    for i in range(0, len(df), 25):
        row = df.iloc[i]
        print(f"{int(row['frame_idx']):<8} {row['depth_mean']:<12.1f} {row['depth_std']:<12.1f} {row['valid_pct']:<10.1f}")

    # Also show last frame
    if len(df) > 0:
        row = df.iloc[-1]
        print(f"{int(row['frame_idx']):<8} {row['depth_mean']:<12.1f} {row['depth_std']:<12.1f} {row['valid_pct']:<10.1f}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Analyze depth data consistency in dataset")
    parser.add_argument("dataset_path", type=str, help="Path to dataset")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to analyze")

    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)

    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        return

    try:
        frames_data = load_depth_from_parquet(dataset_path, args.episode)
        analyze_depth_consistency(frames_data)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
