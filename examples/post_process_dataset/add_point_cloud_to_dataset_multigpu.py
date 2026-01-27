#!/usr/bin/env python

"""
Multi-GPU version of point cloud addition to LeRobot datasets.

This script leverages multiple GPUs to accelerate point cloud computation.
Each GPU processes a subset of episodes in parallel using PyTorch for
GPU-accelerated depth-to-point-cloud conversion.

Usage:
```bash
# Use all available GPUs (auto-detected)
python examples/post_process_dataset/add_point_cloud_to_dataset_multigpu.py \
    --dataset_dir=$HOME/lerobot_datasets/depth_test_v2

# Specify specific GPUs
python examples/post_process_dataset/add_point_cloud_to_dataset_multigpu.py \
    --dataset_dir=$HOME/lerobot_datasets/depth_test_v2 \
    --gpu_ids=0,1,2,3

# Custom voxel size
python examples/post_process_dataset/add_point_cloud_to_dataset_multigpu.py \
    --dataset_dir=$HOME/lerobot_datasets/depth_test_v2 \
    --voxel_size=0.005
```
"""

import io
import json
import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty

import cv2
import jsonlines
import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tap import Tap
from tqdm import tqdm

from examples.post_process_dataset.constants import EXTRINSICS_FILE, INTRINSICS_FILE, WORKSPACE

msgpack_numpy.patch()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Args(Tap):
    """Arguments for multi-GPU point cloud processing."""

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

    # GPU configuration
    gpu_ids: str = ""  # Comma-separated GPU IDs (e.g., "0,1,2,3"). Empty = all available GPUs
    cpu_workers_per_gpu: int = 4  # Number of CPU threads per GPU for data loading


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


def depth_to_point_cloud_gpu(
    depth_image: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_to_world_matrix: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """Convert depth image to point cloud in world coordinates using GPU."""
    height, width = depth_image.shape

    # Create pixel coordinate grids
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing='ij'
    )

    # Extract intrinsic parameters
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # Compute camera coordinates
    z = depth_image
    x_cam = (x_coords - cx) * z / fx
    y_cam = (y_coords - cy) * z / fy

    # Stack to get camera coordinates (H, W, 3)
    camera_coords = torch.stack([x_cam, y_cam, z], dim=-1)

    # Reshape to (N, 3) where N = H * W
    points = camera_coords.reshape(-1, 3)

    # Add homogeneous coordinate
    ones = torch.ones(points.shape[0], 1, device=device, dtype=torch.float32)
    points_homo = torch.cat([points, ones], dim=1)

    # Transform to world coordinates
    world_points = (camera_to_world_matrix @ points_homo.T).T[:, :3]

    # Reshape back to (H, W, 3)
    world_points = world_points.reshape(height, width, 3)

    return world_points


def crop_point_cloud_by_workspace_gpu(
    point_cloud: torch.Tensor,
    workspace: dict,
    device: torch.device
) -> torch.Tensor:
    """Return points within the workspace using GPU."""
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
    """Voxelize point cloud using Open3D (CPU)."""
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


def get_point_cloud_from_rgb_depth_gpu(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    device: torch.device,
    workspace: dict | None = None,
    voxel_size: float = 0.01
) -> np.ndarray:
    """Generate a processed point cloud from RGB and depth images using GPU."""
    # Move depth to GPU
    depth_tensor = torch.from_numpy(depth).to(device=device, dtype=torch.float32)

    # Compute point cloud on GPU
    point_cloud_xyz = depth_to_point_cloud_gpu(depth_tensor, intrinsics, extrinsics, device)

    # Normalize RGB and move to GPU
    rgb_tensor = torch.from_numpy(rgb).to(device=device, dtype=torch.float32) / 255.0

    # Concatenate XYZ and RGB (H, W, 6)
    point_cloud = torch.cat([point_cloud_xyz, rgb_tensor], dim=2)

    # Reshape to (N, 6)
    point_cloud = point_cloud.reshape(-1, 6)

    # Filter out invalid depth points (z > 0)
    valid_mask = point_cloud[:, 2] > 0
    point_cloud = point_cloud[valid_mask]

    # Crop to workspace on GPU
    if workspace is not None:
        point_mask = (
            (point_cloud[:, 0] > workspace['X_BBOX'][0]) &
            (point_cloud[:, 0] < workspace['X_BBOX'][1]) &
            (point_cloud[:, 1] > workspace['Y_BBOX'][0]) &
            (point_cloud[:, 1] < workspace['Y_BBOX'][1]) &
            (point_cloud[:, 2] > workspace['Z_BBOX'][0]) &
            (point_cloud[:, 2] < workspace['Z_BBOX'][1])
        )
        point_cloud = point_cloud[point_mask]

    # Move back to CPU for voxelization (Open3D doesn't support GPU)
    point_cloud_np = point_cloud.cpu().numpy()

    # Voxelize on CPU
    if voxel_size > 0 and len(point_cloud_np) > 0:
        point_cloud_np = voxelize_point_cloud(
            point_cloud_np[:, :3], point_cloud_np[:, 3:], voxel_size
        )

    return point_cloud_np


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
    """Decode all frames from a video file in a single pass."""
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


def load_frame_data(
    row,
    depth_key: str,
    rgb_key: str,
    rgb_in_video: bool,
    rgb_frames: list[np.ndarray] | None,
    depth_scale: float,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Load RGB and depth data for a single frame (runs in thread pool)."""
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

    return frame_idx, rgb_arr, depth_arr


def gpu_worker(
    gpu_id: int,
    episode_queue: mp.Queue,
    result_queue: mp.Queue,
    dataset_path: Path,
    intrinsics_np: np.ndarray,
    extrinsics_np: np.ndarray,
    rgb_key: str,
    depth_key: str,
    video_keys: list[str],
    is_v21: bool,
    fps: float,
    workspace: dict | None,
    voxel_size: float,
    depth_scale: float,
    cpu_workers: int,
):
    """Worker process that processes episodes on a specific GPU."""
    # Set CUDA device for this process
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    # Move calibration data to GPU
    intrinsics = torch.from_numpy(intrinsics_np).to(device=device, dtype=torch.float32)
    extrinsics = torch.from_numpy(extrinsics_np).to(device=device, dtype=torch.float32)

    while True:
        try:
            ep = episode_queue.get(timeout=1)
        except Empty:
            # Check if we should exit
            if episode_queue.empty():
                break
            continue

        if ep is None:  # Poison pill
            break

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

        # Pre-load all RGB frames at once (video decoding)
        rgb_frames = None
        if rgb_in_video:
            if is_v21:
                rgb_frames = decode_all_video_frames(video_path)
            else:
                from_timestamp = ep.get(f"videos/{rgb_key}/from_timestamp", 0.0)
                rgb_frames = decode_video_frames_in_range(
                    video_path, from_timestamp, episode_length, fps
                )

        # Use thread pool for parallel I/O (loading depth images from parquet)
        results = []
        with ThreadPoolExecutor(max_workers=cpu_workers) as executor:
            # Submit all frame loading tasks
            futures = [
                executor.submit(
                    load_frame_data,
                    row, depth_key, rgb_key, rgb_in_video, rgb_frames, depth_scale
                )
                for _, row in df.iterrows()
            ]

            # Process results as they complete (GPU computation)
            for future in futures:
                frame_idx, rgb_arr, depth_arr = future.result()

                # Compute point cloud on GPU
                point_cloud = get_point_cloud_from_rgb_depth_gpu(
                    rgb_arr, depth_arr, intrinsics, extrinsics, device,
                    workspace=workspace, voxel_size=voxel_size
                )

                point_key = f"{episode_idx}-{frame_idx}"
                results.append((point_key, point_cloud, len(point_cloud)))

        result_queue.put((episode_idx, results))

    result_queue.put(None)  # Signal worker is done


def add_point_clouds_to_dataset_multigpu(
    dataset_dir: str,
    intrinsics_file: str,
    extrinsics_file: str,
    rgb_key: str = "observation.images.front",
    depth_key: str = "observation.images.front_depth",
    output_key: str = "observation.point_cloud",
    voxel_size: float = 0.01,
    workspace: dict | None = None,
    depth_scale: float = 1000.0,
    gpu_ids: list[int] | None = None,
    cpu_workers_per_gpu: int = 4,
):
    """Add point clouds to a LeRobot dataset using multiple GPUs."""
    dataset_path = Path(dataset_dir)
    intrinsics = load_intrinsics(Path(intrinsics_file))
    extrinsics = load_extrinsics(Path(extrinsics_file))

    # Determine GPUs to use
    if gpu_ids is None or len(gpu_ids) == 0:
        num_gpus = torch.cuda.device_count()
        gpu_ids = list(range(num_gpus))

    num_workers = len(gpu_ids)

    logging.info(f"Loaded intrinsics:\n{intrinsics}")
    logging.info(f"Loaded extrinsics (camera-to-world):\n{extrinsics}")
    logging.info(f"Workspace bounds: {workspace}")
    logging.info(f"Voxel size: {voxel_size}")
    logging.info(f"Using GPUs: {gpu_ids}")
    logging.info(f"CPU workers per GPU: {cpu_workers_per_gpu}")

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

    # Create queues for inter-process communication
    episode_queue = mp.Queue()
    result_queue = mp.Queue()

    # Fill episode queue
    for ep in episodes:
        episode_queue.put(ep)

    # Add poison pills for each worker
    for _ in range(num_workers):
        episode_queue.put(None)

    # Start worker processes
    workers = []
    for gpu_id in gpu_ids:
        p = mp.Process(
            target=gpu_worker,
            args=(
                gpu_id,
                episode_queue,
                result_queue,
                dataset_path,
                intrinsics,
                extrinsics,
                rgb_key,
                depth_key,
                video_keys,
                is_v21,
                fps,
                workspace,
                voxel_size,
                depth_scale,
                cpu_workers_per_gpu,
            )
        )
        p.start()
        workers.append(p)

    # Collect results
    npoints_list = []
    finished_workers = 0

    with tqdm(total=len(episodes), desc="Processing episodes") as pbar:
        while finished_workers < num_workers:
            result = result_queue.get()
            if result is None:
                finished_workers += 1
                continue

            episode_idx, results = result

            # Batch write all point clouds from this episode
            with lmdb_env.begin(write=True) as txn:
                for point_key, point_cloud, npoints in results:
                    txn.put(point_key.encode('ascii'), msgpack.packb(point_cloud))
                    npoints_list.append(npoints)

            pbar.update(1)

    # Wait for all workers to finish
    for p in workers:
        p.join()

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
    mp.set_start_method('spawn', force=True)

    args = Args().parse_args()

    workspace = {
        'X_BBOX': [args.workspace_x_min, args.workspace_x_max],
        'Y_BBOX': [args.workspace_y_min, args.workspace_y_max],
        'Z_BBOX': [args.workspace_z_min, args.workspace_z_max],
    }

    # Parse GPU IDs
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]

    add_point_clouds_to_dataset_multigpu(
        dataset_dir=args.dataset_dir,
        intrinsics_file=args.intrinsics_file,
        extrinsics_file=args.extrinsics_file,
        rgb_key=args.rgb_key,
        depth_key=args.depth_key,
        output_key=args.output_key,
        voxel_size=args.voxel_size,
        workspace=workspace,
        depth_scale=args.depth_scale,
        gpu_ids=gpu_ids,
        cpu_workers_per_gpu=args.cpu_workers_per_gpu,
    )