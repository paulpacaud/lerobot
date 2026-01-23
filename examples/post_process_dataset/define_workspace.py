#!/usr/bin/env python

"""
Interactive tool to define workspace bounds for point cloud cropping.

This script visualizes a point cloud with a wireframe bounding box showing
the current workspace bounds. Run it multiple times with different bounds
to find the optimal workspace for your setup.

Supports both v2.1 and v3.0 LeRobot dataset formats.

Usage:
```bash
# Visualize with default workspace
python examples/post_process_dataset/define_workspace.py \
    --dataset-dir=$HOME/lerobot_datasets/depth_test_v2 \
    --intrinsics-file=examples/camera_calibration/intrinsics.npz \
    --extrinsics-file=examples/camera_calibration/extrinsics.npz \
    --episode-index=0 --frame-index=100

# Adjust bounds and re-run
python examples/post_process_dataset/define_workspace.py \
    --dataset-dir=$HOME/lerobot_datasets/depth_test_v2 \
    --intrinsics-file=examples/camera_calibration/intrinsics.npz \
    --extrinsics-file=examples/camera_calibration/extrinsics.npz \
    --x-min=-0.3 --x-max=0.3 --y-min=-0.4 --y-max=0.3 --z-min=0.0 --z-max=0.4 \
    --show-outside
```
"""

import io
import json
from pathlib import Path

import cv2
import jsonlines
import numpy as np
import open3d as o3d
import pandas as pd
from PIL import Image
from tap import Tap

DEFAULT_CHUNK_SIZE = 1000


class Args(Tap):
    """Arguments for the workspace definition tool."""

    # Required arguments
    dataset_dir: str  # Path to the LeRobot dataset directory
    intrinsics_file: str  # Path to camera intrinsics npz file
    extrinsics_file: str  # Path to camera extrinsics npz file

    # Frame selection
    episode_index: int = 0  # Episode index to visualize
    frame_index: int = 0  # Frame index to visualize

    # Data keys
    rgb_key: str = "observation.images.front"  # Key for RGB images
    depth_key: str = "observation.images.front_depth"  # Key for depth images
    depth_scale: float = 1000.0  # Scale factor for depth (mm->m)

    # Workspace bounds (meters)
    x_min: float = -0.5  # Workspace X minimum
    x_max: float = 0.5  # Workspace X maximum
    y_min: float = -0.5  # Workspace Y minimum
    y_max: float = 0.5  # Workspace Y maximum
    z_min: float = 0.0  # Workspace Z minimum
    z_max: float = 0.5  # Workspace Z maximum

    # Visualization options
    show_outside: bool = False  # Show points outside workspace (dimmed)
    axes_size: float = 0.15  # Size of coordinate axes
    point_size: float = 2.0  # Rendered point size


def load_intrinsics(intrinsics_file: Path) -> np.ndarray:
    data = np.load(intrinsics_file)
    return data['K']


def load_extrinsics(extrinsics_file: Path) -> np.ndarray:
    data = np.load(extrinsics_file)
    rvec = data['rvec']
    tvec = data['tvec']
    R, _ = cv2.Rodrigues(rvec)
    T_w2c = np.eye(4)
    T_w2c[:3, :3] = R
    T_w2c[:3, 3] = tvec.flatten()
    T_c2w = np.linalg.inv(T_w2c)
    return T_c2w


def depth_to_point_cloud(depth_image, intrinsics, camera_to_world_matrix):
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
    return world_points


def create_workspace_bbox(workspace: dict, color=(1, 0, 0)) -> o3d.geometry.LineSet:
    """Create a wireframe bounding box for the workspace."""
    x_min, x_max = workspace['X_BBOX']
    y_min, y_max = workspace['Y_BBOX']
    z_min, z_max = workspace['Z_BBOX']

    points = [
        [x_min, y_min, z_min], [x_max, y_min, z_min],
        [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max],
        [x_max, y_max, z_max], [x_min, y_max, z_max],
    ]
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])
    return line_set


def get_workspace_mask(points: np.ndarray, workspace: dict) -> np.ndarray:
    """Return boolean mask for points inside workspace."""
    return (
        (points[:, 0] > workspace['X_BBOX'][0]) &
        (points[:, 0] < workspace['X_BBOX'][1]) &
        (points[:, 1] > workspace['Y_BBOX'][0]) &
        (points[:, 1] < workspace['Y_BBOX'][1]) &
        (points[:, 2] > workspace['Z_BBOX'][0]) &
        (points[:, 2] < workspace['Z_BBOX'][1])
    )


def load_info(root: Path) -> dict:
    with open(root / "meta" / "info.json") as f:
        return json.load(f)


def load_episodes_v21(root: Path) -> list[dict]:
    episodes = []
    with jsonlines.open(root / "meta" / "episodes.jsonl") as reader:
        for ep in reader:
            episodes.append(ep)
    return sorted(episodes, key=lambda x: x["episode_index"])


def load_episodes_v3(root: Path) -> list[dict]:
    episodes_dir = root / "meta" / "episodes"
    all_episodes = []
    for parquet_file in sorted(episodes_dir.glob("*/*.parquet")):
        df = pd.read_parquet(parquet_file)
        for _, row in df.iterrows():
            all_episodes.append(row.to_dict())
    return sorted(all_episodes, key=lambda x: x["episode_index"])


def decode_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    """Decode a specific frame from a video file by frame index."""
    import av
    container = av.open(str(video_path))
    for i, frame in enumerate(container.decode(video=0)):
        if i == frame_index:
            img = frame.to_ndarray(format='rgb24')
            container.close()
            return img
    container.close()
    raise ValueError(f"Could not decode frame {frame_index} from {video_path}")


def decode_video_frame_at_timestamp(video_path: Path, timestamp: float) -> np.ndarray:
    """Decode a frame from a video file at a specific timestamp."""
    import av
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    seek_timestamp = int(timestamp / stream.time_base)
    try:
        container.seek(seek_timestamp, stream=stream)
    except av.error.InvalidDataError:
        container.seek(0)
    tolerance = 0.05
    for frame in container.decode(video=0):
        current_ts = float(frame.pts * stream.time_base)
        if current_ts >= timestamp - tolerance:
            img = frame.to_ndarray(format='rgb24')
            container.close()
            return img
    container.close()
    raise ValueError(f"Could not decode frame at timestamp {timestamp}")


def main():
    args = Args().parse_args()

    workspace = {
        'X_BBOX': [args.x_min, args.x_max],
        'Y_BBOX': [args.y_min, args.y_max],
        'Z_BBOX': [args.z_min, args.z_max],
    }

    dataset_path = Path(args.dataset_dir)
    intrinsics = load_intrinsics(Path(args.intrinsics_file))
    extrinsics = load_extrinsics(Path(args.extrinsics_file))

    # Load info and detect version
    info = load_info(dataset_path)
    version = info.get("codebase_version", "v2.1")
    fps = info["fps"]
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    is_v21 = version.startswith("v2")

    print(f"\n{'='*60}")
    print("WORKSPACE DEFINITION TOOL")
    print(f"{'='*60}")
    print(f"\nDataset version: {version}")
    print(f"\nCurrent workspace bounds:")
    print(f"  X: [{args.x_min:.3f}, {args.x_max:.3f}] m")
    print(f"  Y: [{args.y_min:.3f}, {args.y_max:.3f}] m")
    print(f"  Z: [{args.z_min:.3f}, {args.z_max:.3f}] m")

    # Load episodes
    if is_v21:
        episodes = load_episodes_v21(dataset_path)
    else:
        episodes = load_episodes_v3(dataset_path)

    ep = episodes[args.episode_index]

    # Load data for the frame
    print(f"\nLoading frame: episode {args.episode_index}, frame {args.frame_index}")

    if is_v21:
        chunk_idx = args.episode_index // DEFAULT_CHUNK_SIZE
        data_path = dataset_path / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{args.episode_index:06d}.parquet"
        video_path = dataset_path / "videos" / f"chunk-{chunk_idx:03d}" / args.rgb_key / f"episode_{args.episode_index:06d}.mp4"
    else:
        data_chunk_idx = ep.get("data/chunk_index", 0)
        data_file_idx = ep.get("data/file_index", 0)
        data_path = dataset_path / "data" / f"chunk-{data_chunk_idx:03d}" / f"file-{data_file_idx:03d}.parquet"
        video_chunk_idx = ep.get(f"videos/{args.rgb_key}/chunk_index", 0)
        video_file_idx = ep.get(f"videos/{args.rgb_key}/file_index", 0)
        video_path = dataset_path / "videos" / args.rgb_key / f"chunk-{video_chunk_idx:03d}" / f"file-{video_file_idx:03d}.mp4"

    df = pd.read_parquet(data_path)
    if not is_v21:
        df = df[df["episode_index"] == args.episode_index]

    row = df[df["frame_index"] == args.frame_index].iloc[0]

    # Load depth
    depth_entry = row[args.depth_key]
    depth_img = Image.open(io.BytesIO(depth_entry['bytes']))
    depth_arr = np.array(depth_img).astype(np.float32) / args.depth_scale

    # Load RGB
    if args.rgb_key in video_keys:
        if is_v21:
            rgb_arr = decode_video_frame(video_path, args.frame_index)
        else:
            from_timestamp = ep.get(f"videos/{args.rgb_key}/from_timestamp", 0.0)
            frame_timestamp = from_timestamp + args.frame_index / fps
            rgb_arr = decode_video_frame_at_timestamp(video_path, frame_timestamp)
    else:
        rgb_entry = row[args.rgb_key]
        rgb_img = Image.open(io.BytesIO(rgb_entry['bytes']))
        rgb_arr = np.array(rgb_img)

    # Compute point cloud
    world_points = depth_to_point_cloud(depth_arr, intrinsics, extrinsics)
    rgb_flat = rgb_arr.reshape(-1, 3).astype(np.float32) / 255.0

    # Filter valid depth
    valid_mask = depth_arr.reshape(-1) > 0
    world_points = world_points[valid_mask]
    rgb_flat = rgb_flat[valid_mask]

    print(f"\nPoint cloud statistics (full):")
    print(f"  Total points: {len(world_points)}")
    print(f"  X range: [{world_points[:, 0].min():.3f}, {world_points[:, 0].max():.3f}] m")
    print(f"  Y range: [{world_points[:, 1].min():.3f}, {world_points[:, 1].max():.3f}] m")
    print(f"  Z range: [{world_points[:, 2].min():.3f}, {world_points[:, 2].max():.3f}] m")

    # Workspace filter
    inside_mask = get_workspace_mask(world_points, workspace)
    n_inside = inside_mask.sum()
    n_outside = len(world_points) - n_inside

    print(f"\nWorkspace filter:")
    print(f"  Points inside:  {n_inside} ({100*n_inside/len(world_points):.1f}%)")
    print(f"  Points outside: {n_outside} ({100*n_outside/len(world_points):.1f}%)")

    # Create visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Workspace Definition", width=1280, height=720)

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.axes_size, origin=(0, 0, 0))
    vis.add_geometry(axes)

    bbox = create_workspace_bbox(workspace, color=(1, 0, 0))
    vis.add_geometry(bbox)

    if args.show_outside:
        colors = rgb_flat.copy()
        colors[~inside_mask] = colors[~inside_mask] * 0.3
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(world_points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(world_points[inside_mask])
        pcd.colors = o3d.utility.Vector3dVector(rgb_flat[inside_mask])
    vis.add_geometry(pcd)

    render_opt = vis.get_render_option()
    render_opt.point_size = args.point_size
    render_opt.background_color = np.array([1.0, 1.0, 1.0])

    view_ctl = vis.get_view_control()
    view_ctl.set_zoom(0.5)
    view_ctl.set_front([0.5, -0.5, 0.7])
    view_ctl.set_lookat([0, 0, 0.1])
    view_ctl.set_up([0, 0, 1])

    print(f"\n{'='*60}")
    print("COMMAND TO USE THESE BOUNDS:")
    print(f"{'='*60}")
    print(f"""
python examples/post_process_dataset/add_point_cloud_to_dataset.py \\
    --dataset-dir={args.dataset_dir} \\
    --intrinsics-file={args.intrinsics_file} \\
    --extrinsics-file={args.extrinsics_file} \\
    --workspace-x-min={args.x_min} --workspace-x-max={args.x_max} \\
    --workspace-y-min={args.y_min} --workspace-y-max={args.y_max} \\
    --workspace-z-min={args.z_min} --workspace-z-max={args.z_max}
""")
    print("Controls: Left-click=rotate, Right-click=pan, Scroll=zoom, Q=quit")
    print(f"{'='*60}\n")

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
