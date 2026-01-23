#!/usr/bin/env python

"""
This script adds processed point clouds to a LeRobot dataset (v2.1 or v3.0).

It takes RGB + depth data from the dataset, computes 3D point clouds using
camera intrinsics and extrinsics, then crops and voxelizes them.

The point clouds are stored in a separate LMDB database because they have
variable sizes that don't fit well in parquet files.

Usage:
```bash
# Using default calibration and workspace from constants folder
python examples/post_process_dataset/add_point_cloud_to_dataset.py \
    --dataset_dir=$HOME/lerobot_datasets/depth_test_v2

# Override with custom values
python examples/post_process_dataset/add_point_cloud_to_dataset.py \
    --dataset_dir=$HOME/lerobot_datasets/depth_test_v2 \
    --voxel_size=0.005

# Use multiple workers for parallel processing
python examples/post_process_dataset/add_point_cloud_to_dataset.py \
    --dataset_dir=$HOME/lerobot_datasets/depth_test_v2 \
    --num_workers=4
```
"""

import io
import json
import logging
import multiprocessing as mp
from functools import partial
from pathlib import Path

import cv2
import jsonlines
import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import pandas as pd
from PIL import Image
from tap import Tap
from tqdm import tqdm

from examples.post_process_dataset.constants import INTRINSICS_FILE, EXTRINSICS_FILE, WORKSPACE

msgpack_numpy.patch()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Args(Tap):
    """Arguments for adding point clouds to a LeRobot dataset."""

    # Required arguments
    dataset_dir: str  # Path to the LeRobot dataset directory

    # Calibration files (defaults to constants folder)
    intrinsics_file: str = str(INTRINSICS_FILE)  # Path to camera intrinsics npz file
    extrinsics_file: str = str(EXTRINSICS_FILE)  # Path to camera extrinsics npz file

    # Data keys
    rgb_key: str = "observation.images.front"  # Key for RGB images in the dataset
    depth_key: str = "observation.images.front_depth"  # Key for depth images in the dataset
    output_key: str = "observation.point_cloud"  # Key for the output point cloud

    # Processing parameters
    voxel_size: float = 0.01  # Voxel size for downsampling (meters)
    depth_scale: float = 1000.0  # Scale factor to convert depth values to meters (1000 for mm->m)

    # Workspace bounds (meters) - defaults from constants/WORKSPACE
    workspace_x_min: float = WORKSPACE['X_BBOX'][0]  # Workspace X minimum bound
    workspace_x_max: float = WORKSPACE['X_BBOX'][1]  # Workspace X maximum bound
    workspace_y_min: float = WORKSPACE['Y_BBOX'][0]  # Workspace Y minimum bound
    workspace_y_max: float = WORKSPACE['Y_BBOX'][1]  # Workspace Y maximum bound
    workspace_z_min: float = WORKSPACE['Z_BBOX'][0]  # Workspace Z minimum bound
    workspace_z_max: float = WORKSPACE['Z_BBOX'][1]  # Workspace Z maximum bound

    # Parallelization
    num_workers: int = 1  # Number of parallel workers (1 = single-threaded)


DEFAULT_CHUNK_SIZE = 1000


def load_intrinsics(intrinsics_file: Path) -> np.ndarray:
    """Load camera intrinsics matrix from npz file."""
    data = np.load(intrinsics_file)
    return data['K']


def load_extrinsics(extrinsics_file: Path) -> np.ndarray:
    """Load camera extrinsics and convert to 4x4 camera-to-world matrix."""
    data = np.load(extrinsics_file)
    rvec = data['rvec']
    tvec = data['tvec']
    R, _ = cv2.Rodrigues(rvec)
    T_w2c = np.eye(4)
    T_w2c[:3, :3] = R
    T_w2c[:3, 3] = tvec.flatten()
    T_c2w = np.linalg.inv(T_w2c)
    return T_c2w


def depth_to_point_cloud(
    depth_image: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world_matrix: np.ndarray
) -> np.ndarray:
    """Convert depth image to point cloud in world coordinates."""
    height, width = depth_image.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    z = depth_image
    x_cam = (x - cx) * z / fx
    y_cam = (y - cy) * z / fy
    camera_coords = np.stack((x_cam, y_cam, z), axis=-1)
    points = camera_coords.reshape(-1, 3)
    points_homo = np.column_stack((points, np.ones(len(points))))
    world_points = (camera_to_world_matrix @ points_homo.T).T[:, :3]
    world_points = world_points.reshape(height, width, 3)
    return world_points


def crop_point_cloud_by_workspace(point_cloud: np.ndarray, workspace: dict):
    """Return points within the workspace."""
    point_mask = (
        (point_cloud[..., 0] > workspace['X_BBOX'][0]) &
        (point_cloud[..., 0] < workspace['X_BBOX'][1]) &
        (point_cloud[..., 1] > workspace['Y_BBOX'][0]) &
        (point_cloud[..., 1] < workspace['Y_BBOX'][1]) &
        (point_cloud[..., 2] > workspace['Z_BBOX'][0]) &
        (point_cloud[..., 2] < workspace['Z_BBOX'][1])
    )
    return point_cloud[point_mask]


def voxelize_point_cloud(points: np.ndarray, colors: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxelize point cloud using Open3D."""
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        point_cloud = np.concatenate(
            [np.asarray(pcd.points), np.asarray(pcd.colors)], axis=1
        ).astype(np.float32)
        return point_cloud
    except ImportError:
        logging.warning("Open3D not available, skipping voxelization")
        return np.concatenate([points, colors], axis=1).astype(np.float32)


def get_point_cloud_from_rgb_depth(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    workspace: dict | None = None,
    voxel_size: float = 0.01
) -> np.ndarray:
    """Generate a processed point cloud from RGB and depth images."""
    point_cloud_xyz = depth_to_point_cloud(depth, intrinsics, extrinsics)
    point_cloud_xyz = point_cloud_xyz.astype(np.float32)
    rgb_normalized = rgb.astype(np.float32) / 255.0
    point_cloud = np.concatenate([point_cloud_xyz, rgb_normalized], axis=2)
    point_cloud = point_cloud.reshape(-1, 6)

    # Filter out invalid depth points
    valid_mask = point_cloud[:, 2] > 0
    point_cloud = point_cloud[valid_mask]

    # Crop to workspace
    if workspace is not None:
        point_cloud = crop_point_cloud_by_workspace(point_cloud, workspace)

    # Voxelize
    if voxel_size > 0 and len(point_cloud) > 0:
        point_cloud = voxelize_point_cloud(
            point_cloud[:, :3], point_cloud[:, 3:], voxel_size
        )

    return point_cloud


def load_info(root: Path) -> dict:
    """Load info.json from dataset root."""
    with open(root / "meta" / "info.json") as f:
        return json.load(f)


def get_video_keys(info: dict) -> list[str]:
    """Get video keys from features."""
    return [key for key, ft in info["features"].items() if ft["dtype"] == "video"]


def get_image_keys(info: dict) -> list[str]:
    """Get image keys from features."""
    return [key for key, ft in info["features"].items() if ft["dtype"] == "image"]


def load_episodes_v21(root: Path) -> list[dict]:
    """Load episodes metadata from v2.1 jsonl format."""
    episodes = []
    with jsonlines.open(root / "meta" / "episodes.jsonl") as reader:
        for ep in reader:
            episodes.append(ep)
    return sorted(episodes, key=lambda x: x["episode_index"])


def load_episodes_v3(root: Path) -> list[dict]:
    """Load episodes metadata from v3 parquet format."""
    episodes_dir = root / "meta" / "episodes"
    all_episodes = []
    for parquet_file in sorted(episodes_dir.glob("*/*.parquet")):
        df = pd.read_parquet(parquet_file)
        for _, row in df.iterrows():
            all_episodes.append(row.to_dict())
    return sorted(all_episodes, key=lambda x: x["episode_index"])


def decode_all_video_frames(video_path: Path) -> list[np.ndarray]:
    """Decode all frames from a video file in a single pass (O(n) instead of O(n²))."""
    import av
    frames = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format='rgb24'))
    return frames


def decode_video_frames_in_range(
    video_path: Path,
    start_timestamp: float,
    num_frames: int,
    fps: float
) -> list[np.ndarray]:
    """Decode a range of frames from a video starting at a timestamp."""
    import av
    frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        seek_timestamp = int(start_timestamp / stream.time_base)

        try:
            container.seek(seek_timestamp, stream=stream)
        except av.error.InvalidDataError:
            container.seek(0)

        tolerance = 0.05
        started = False
        for frame in container.decode(video=0):
            current_ts = float(frame.pts * stream.time_base)
            if not started and current_ts >= start_timestamp - tolerance:
                started = True
            if started:
                frames.append(frame.to_ndarray(format='rgb24'))
                if len(frames) >= num_frames:
                    break

    return frames


def process_episode(
    ep: dict,
    dataset_path: Path,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    rgb_key: str,
    depth_key: str,
    video_keys: list[str],
    is_v21: bool,
    fps: float,
    workspace: dict | None,
    voxel_size: float,
    depth_scale: float,
) -> list[tuple[str, np.ndarray, int]]:
    """Process a single episode and return list of (key, point_cloud, npoints) tuples."""
    episode_idx = ep["episode_index"]
    episode_length = ep["length"]

    # Load episode data paths
    if is_v21:
        chunk_idx = episode_idx // DEFAULT_CHUNK_SIZE
        data_path = dataset_path / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{episode_idx:06d}.parquet"
        video_path = dataset_path / "videos" / f"chunk-{chunk_idx:03d}" / rgb_key / f"episode_{episode_idx:06d}.mp4"
    else:
        data_chunk_idx = ep.get("data/chunk_index", 0)
        data_file_idx = ep.get("data/file_index", 0)
        data_path = dataset_path / "data" / f"chunk-{data_chunk_idx:03d}" / f"file-{data_file_idx:03d}.parquet"

        video_chunk_idx = ep.get(f"videos/{rgb_key}/chunk_index", 0)
        video_file_idx = ep.get(f"videos/{rgb_key}/file_index", 0)
        video_path = dataset_path / "videos" / rgb_key / f"chunk-{video_chunk_idx:03d}" / f"file-{video_file_idx:03d}.mp4"

    # Load parquet data
    df = pd.read_parquet(data_path)
    if not is_v21:
        df = df[df["episode_index"] == episode_idx]

    rgb_in_video = rgb_key in video_keys

    # Pre-load all RGB frames at once (batch video decoding - major speedup)
    rgb_frames = None
    if rgb_in_video:
        if is_v21:
            # V2.1: decode entire per-episode video
            rgb_frames = decode_all_video_frames(video_path)
        else:
            # V3: decode range from concatenated video
            from_timestamp = ep.get(f"videos/{rgb_key}/from_timestamp", 0.0)
            rgb_frames = decode_video_frames_in_range(
                video_path, from_timestamp, episode_length, fps
            )

    results = []
    for _, row in df.iterrows():
        frame_idx = int(row["frame_index"])

        # Load depth image
        depth_entry = row[depth_key]
        depth_img = Image.open(io.BytesIO(depth_entry['bytes']))
        depth_arr = np.array(depth_img).astype(np.float32) / depth_scale

        # Get RGB image
        if rgb_in_video:
            rgb_arr = rgb_frames[frame_idx]
        else:
            rgb_entry = row[rgb_key]
            rgb_img = Image.open(io.BytesIO(rgb_entry['bytes']))
            rgb_arr = np.array(rgb_img)

        # Compute point cloud
        point_cloud = get_point_cloud_from_rgb_depth(
            rgb_arr, depth_arr, intrinsics, extrinsics,
            workspace=workspace, voxel_size=voxel_size
        )

        point_key = f"{episode_idx}-{frame_idx}"
        results.append((point_key, point_cloud, len(point_cloud)))

    return results


def process_episode_wrapper(args):
    """Wrapper for multiprocessing - unpacks arguments."""
    return process_episode(*args)


def add_point_clouds_to_dataset(
    dataset_dir: str,
    intrinsics_file: str,
    extrinsics_file: str,
    rgb_key: str = "observation.images.front",
    depth_key: str = "observation.images.front_depth",
    output_key: str = "observation.point_cloud",
    voxel_size: float = 0.01,
    workspace: dict | None = None,
    depth_scale: float = 1000.0,
    num_workers: int = 1,
):
    """Add point clouds to a LeRobot dataset (v2.1 or v3.0)."""
    dataset_path = Path(dataset_dir)
    intrinsics = load_intrinsics(Path(intrinsics_file))
    extrinsics = load_extrinsics(Path(extrinsics_file))

    logging.info(f"Loaded intrinsics:\n{intrinsics}")
    logging.info(f"Loaded extrinsics (camera-to-world):\n{extrinsics}")
    logging.info(f"Workspace bounds: {workspace}")
    logging.info(f"Voxel size: {voxel_size}")
    logging.info(f"Num workers: {num_workers}")

    # Load dataset info and detect version
    info = load_info(dataset_path)
    version = info.get("codebase_version", "v2.1")
    fps = info["fps"]
    video_keys = get_video_keys(info)
    image_keys = get_image_keys(info)

    logging.info(f"Dataset version: {version}")
    logging.info(f"Dataset FPS: {fps}")
    logging.info(f"Video keys: {video_keys}")
    logging.info(f"Image keys: {image_keys}")

    is_v21 = version.startswith("v2")

    # Load episodes metadata
    if is_v21:
        episodes = load_episodes_v21(dataset_path)
    else:
        episodes = load_episodes_v3(dataset_path)

    logging.info(f"Found {len(episodes)} episodes")

    # Validate depth key exists
    if episodes:
        ep = episodes[0]
        if is_v21:
            chunk_idx = ep["episode_index"] // DEFAULT_CHUNK_SIZE
            data_path = dataset_path / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep['episode_index']:06d}.parquet"
        else:
            data_chunk_idx = ep.get("data/chunk_index", 0)
            data_file_idx = ep.get("data/file_index", 0)
            data_path = dataset_path / "data" / f"chunk-{data_chunk_idx:03d}" / f"file-{data_file_idx:03d}.parquet"
        df = pd.read_parquet(data_path)
        if depth_key not in df.columns:
            raise ValueError(f"Depth key '{depth_key}' not found in dataset")

    # Create LMDB database for point clouds
    lmdb_path = dataset_path / "point_clouds"
    if lmdb_path.exists():
        import shutil
        shutil.rmtree(lmdb_path)

    lmdb_env = lmdb.open(str(lmdb_path), map_size=int(1024**4))

    # Prepare arguments for each episode
    episode_args = [
        (ep, dataset_path, intrinsics, extrinsics, rgb_key, depth_key,
         video_keys, is_v21, fps, workspace, voxel_size, depth_scale)
        for ep in episodes
    ]

    # Process episodes
    npoints_list = []

    if num_workers <= 1:
        # Single-threaded processing
        for args in tqdm(episode_args, desc="Processing episodes"):
            results = process_episode(*args)
            # Batch write all point clouds from this episode
            with lmdb_env.begin(write=True) as txn:
                for point_key, point_cloud, npoints in results:
                    txn.put(point_key.encode('ascii'), msgpack.packb(point_cloud))
                    npoints_list.append(npoints)
    else:
        # Multi-process parallel processing
        with mp.Pool(num_workers) as pool:
            for results in tqdm(
                pool.imap(process_episode_wrapper, episode_args),
                total=len(episodes),
                desc="Processing episodes"
            ):
                # Batch write all point clouds from this episode
                with lmdb_env.begin(write=True) as txn:
                    for point_key, point_cloud, npoints in results:
                        txn.put(point_key.encode('ascii'), msgpack.packb(point_cloud))
                        npoints_list.append(npoints)

    lmdb_env.close()

    # Print statistics
    if npoints_list:
        logging.info(f"Point cloud statistics:")
        logging.info(f"  Total frames: {len(npoints_list)}")
        logging.info(f"  Min points: {np.min(npoints_list)}")
        logging.info(f"  Max points: {np.max(npoints_list)}")
        logging.info(f"  Mean points: {np.mean(npoints_list):.1f}")

    # Update info.json
    info["features"][output_key] = {
        "dtype": "point_cloud",
        "shape": [None, 6],
        "names": ["x", "y", "z", "r", "g", "b"],
        "info": {
            "storage": "lmdb",
            "path": "point_clouds",
            "voxel_size": voxel_size,
            "workspace": workspace,
        }
    }

    with open(dataset_path / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    logging.info(f"Successfully added point clouds to dataset at {lmdb_path}")


if __name__ == "__main__":
    args = Args().parse_args()

    workspace = {
        'X_BBOX': [args.workspace_x_min, args.workspace_x_max],
        'Y_BBOX': [args.workspace_y_min, args.workspace_y_max],
        'Z_BBOX': [args.workspace_z_min, args.workspace_z_max],
    }

    add_point_clouds_to_dataset(
        dataset_dir=args.dataset_dir,
        intrinsics_file=args.intrinsics_file,
        extrinsics_file=args.extrinsics_file,
        rgb_key=args.rgb_key,
        depth_key=args.depth_key,
        output_key=args.output_key,
        voxel_size=args.voxel_size,
        workspace=workspace,
        depth_scale=args.depth_scale,
        num_workers=args.num_workers,
    )
