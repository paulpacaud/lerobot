#!/usr/bin/env python3
"""
Crop existing point clouds in a LeRobot dataset to match workspace bounds.

This script is useful when point clouds were generated without proper table height
cropping and need to be re-cropped using updated workspace bounds.

Usage:
```bash
python examples/post_process_dataset/crop_height.py --dataset_dir=$HOME/lerobot_datasets/put_cube_in_spot_pointact
```
"""

import json
import logging
from pathlib import Path

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
from tap import Tap
from tqdm import tqdm

from examples.post_process_dataset.constants import WORKSPACE

msgpack_numpy.patch()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Args(Tap):
    """Arguments for cropping point clouds in a dataset."""

    # Required arguments
    dataset_dir: str  # Path to the LeRobot dataset directory

    # Workspace bounds (meters) - defaults from constants/WORKSPACE
    workspace_x_min: float = WORKSPACE["X_BBOX"][0]  # Workspace X minimum bound
    workspace_x_max: float = WORKSPACE["X_BBOX"][1]  # Workspace X maximum bound
    workspace_y_min: float = WORKSPACE["Y_BBOX"][0]  # Workspace Y minimum bound
    workspace_y_max: float = WORKSPACE["Y_BBOX"][1]  # Workspace Y maximum bound
    workspace_z_min: float = WORKSPACE["Z_BBOX"][0]  # Workspace Z minimum bound
    workspace_z_max: float = WORKSPACE["Z_BBOX"][1]  # Workspace Z maximum bound

    # Options
    dry_run: bool = False  # Only show statistics without modifying the dataset


def crop_point_cloud_by_workspace(point_cloud: np.ndarray, workspace: dict) -> np.ndarray:
    """Return points within the workspace bounds."""
    if len(point_cloud) == 0:
        return point_cloud

    point_mask = (
        (point_cloud[:, 0] > workspace["X_BBOX"][0])
        & (point_cloud[:, 0] < workspace["X_BBOX"][1])
        & (point_cloud[:, 1] > workspace["Y_BBOX"][0])
        & (point_cloud[:, 1] < workspace["Y_BBOX"][1])
        & (point_cloud[:, 2] > workspace["Z_BBOX"][0])
        & (point_cloud[:, 2] < workspace["Z_BBOX"][1])
    )
    return point_cloud[point_mask]


def crop_point_clouds_in_dataset(
    dataset_dir: str,
    workspace: dict,
    dry_run: bool = False,
) -> None:
    """Crop all point clouds in a dataset to the specified workspace bounds."""
    dataset_path = Path(dataset_dir)
    lmdb_path = dataset_path / "point_clouds"

    if not lmdb_path.exists():
        raise FileNotFoundError(f"Point cloud LMDB not found at {lmdb_path}")

    logging.info(f"Dataset: {dataset_path}")
    logging.info(f"Workspace bounds: {workspace}")
    logging.info(f"Dry run: {dry_run}")

    # Open LMDB database
    lmdb_env = lmdb.open(str(lmdb_path), map_size=int(1024**4))

    # First pass: read all keys and point clouds
    with lmdb_env.begin() as txn:
        cursor = txn.cursor()
        keys = [key for key, _ in cursor]

    logging.info(f"Found {len(keys)} point clouds")

    # Statistics
    points_before = []
    points_after = []
    cropped_entries = []

    # Process each point cloud
    for key in tqdm(keys, desc="Processing point clouds"):
        with lmdb_env.begin() as txn:
            data = txn.get(key)
            point_cloud = msgpack.unpackb(data)

        n_before = len(point_cloud)
        cropped_pc = crop_point_cloud_by_workspace(point_cloud, workspace)
        n_after = len(cropped_pc)

        points_before.append(n_before)
        points_after.append(n_after)

        if n_before != n_after:
            cropped_entries.append((key, cropped_pc))

    # Print statistics
    logging.info("=" * 60)
    logging.info("STATISTICS")
    logging.info("=" * 60)
    logging.info(f"Total point clouds: {len(keys)}")
    logging.info(f"Point clouds modified: {len(cropped_entries)}")
    logging.info(f"Before cropping:")
    logging.info(f"  Min points: {np.min(points_before)}")
    logging.info(f"  Max points: {np.max(points_before)}")
    logging.info(f"  Mean points: {np.mean(points_before):.1f}")
    logging.info(f"  Total points: {np.sum(points_before)}")
    logging.info(f"After cropping:")
    logging.info(f"  Min points: {np.min(points_after)}")
    logging.info(f"  Max points: {np.max(points_after)}")
    logging.info(f"  Mean points: {np.mean(points_after):.1f}")
    logging.info(f"  Total points: {np.sum(points_after)}")
    logging.info(f"Points removed: {np.sum(points_before) - np.sum(points_after)}")

    if dry_run:
        logging.info("Dry run - no changes made")
        lmdb_env.close()
        return

    # Write back cropped point clouds
    if cropped_entries:
        logging.info(f"Writing {len(cropped_entries)} modified point clouds...")
        with lmdb_env.begin(write=True) as txn:
            for key, cropped_pc in tqdm(cropped_entries, desc="Writing point clouds"):
                txn.put(key, msgpack.packb(cropped_pc))

    lmdb_env.close()

    # Update info.json with new workspace bounds
    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)

        # Update workspace in point cloud feature info
        for key, feature in info.get("features", {}).items():
            if feature.get("dtype") == "point_cloud" and "info" in feature:
                feature["info"]["workspace"] = workspace
                logging.info(f"Updated workspace bounds in info.json for {key}")

        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)

    logging.info("Done!")


if __name__ == "__main__":
    args = Args().parse_args()

    workspace = {
        "X_BBOX": [args.workspace_x_min, args.workspace_x_max],
        "Y_BBOX": [args.workspace_y_min, args.workspace_y_max],
        "Z_BBOX": [args.workspace_z_min, args.workspace_z_max],
    }

    crop_point_clouds_in_dataset(
        dataset_dir=args.dataset_dir,
        workspace=workspace,
        dry_run=args.dry_run,
    )
