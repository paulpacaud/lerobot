"""
Visualize depth data as 3D point cloud with world origin.

Uses intrinsics.npz and extrinsics.npz from this folder by default.

Usage:
    python visualize_depth.py --dataset depth_test --episode 0 --frame 50
    python visualize_depth.py --dataset depth_test --episode 0 --frame 50 --max-depth 1000

Requirements:
    pip install open3d pyarrow av pillow opencv-python typed-argument-parser
"""

import io
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pyarrow.parquet as pq
from PIL import Image
from tap import Tap


SCRIPT_DIR = Path(__file__).parent


class Args(Tap):
    dataset: Path  # Path to dataset directory
    episode: int = 0  # Episode index
    frame: int = 0  # Frame index within episode
    intrinsics: Path = SCRIPT_DIR / "intrinsics.npz"  # Path to intrinsics.npz
    extrinsics: Path = SCRIPT_DIR / "extrinsics.npz"  # Path to extrinsics.npz
    min_depth: int = 100  # Minimum depth in mm
    max_depth: int = 5000  # Maximum depth in mm
    downsample: int = 1  # Downsample factor
    list_episodes: bool = False  # List available episodes and exit



def get_parquet_files(dataset_path: Path) -> list[Path]:
    """Get sorted list of parquet files in dataset."""
    return sorted((dataset_path / "data").glob("**/*.parquet"))


def find_frame_in_parquet(
    dataset_path: Path, episode_idx: int, frame_idx: int
) -> tuple[int | None, np.ndarray | None]:
    """Find frame in parquet files and return global index and depth data."""
    for pq_file in get_parquet_files(dataset_path):
        table = pq.read_table(pq_file)
        df = table.to_pandas()

        ep_df = df[df['episode_index'] == episode_idx]
        if len(ep_df) == 0:
            continue

        frame_df = ep_df[ep_df['frame_index'] == frame_idx]
        if len(frame_df) == 0:
            continue

        row = frame_df.iloc[0]
        global_frame_idx = row['index']

        # Find and load depth
        depth_cols = [c for c in df.columns if 'depth' in c.lower()]
        depth = None
        if depth_cols:
            depth = load_depth_from_row(row[depth_cols[0]], dataset_path)

        return global_frame_idx, depth

    return None, None


def load_depth_from_row(depth_data: dict, dataset_path: Path) -> np.ndarray | None:
    """Load depth array from parquet row data."""
    if isinstance(depth_data, dict) and 'bytes' in depth_data:
        return np.array(Image.open(io.BytesIO(depth_data['bytes'])))
    elif isinstance(depth_data, dict) and 'path' in depth_data:
        img_path = dataset_path / depth_data['path']
        if img_path.exists():
            return np.array(Image.open(img_path))
    return None


def load_rgb_from_video(dataset_path: Path, global_frame_idx: int) -> np.ndarray | None:
    """Load RGB frame from video file."""
    import av

    video_dir = dataset_path / "videos"
    video_keys = [d.name for d in video_dir.iterdir() if d.is_dir() and "depth" not in d.name]

    if not video_keys:
        return None

    video_files = list((video_dir / video_keys[0]).glob("**/*.mp4"))
    if not video_files:
        return None

    container = av.open(str(video_files[0]))
    rgb = None
    for i, frame in enumerate(container.decode(video=0)):
        if i == global_frame_idx:
            rgb = frame.to_ndarray(format='rgb24')
            break
    container.close()

    return rgb


def load_frame_data(
    dataset_path: Path, episode_idx: int, frame_idx: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load RGB and depth data for a specific frame."""
    global_frame_idx, depth = find_frame_in_parquet(dataset_path, episode_idx, frame_idx)

    if global_frame_idx is None:
        print(f"Frame {frame_idx} not found in episode {episode_idx}")
        return None, None

    rgb = load_rgb_from_video(dataset_path, global_frame_idx)
    return rgb, depth


def get_frame_count(dataset_path: Path, episode_idx: int) -> int:
    """Get number of frames in an episode."""
    for pq_file in get_parquet_files(dataset_path):
        table = pq.read_table(pq_file)
        df = table.to_pandas()
        ep_df = df[df['episode_index'] == episode_idx]
        if len(ep_df) > 0:
            return len(ep_df)
    return 0


def list_episodes(dataset_path: Path) -> None:
    """List all episodes in dataset with frame counts."""
    episodes = set()
    for pq_file in get_parquet_files(dataset_path):
        table = pq.read_table(pq_file)
        df = table.to_pandas()
        episodes.update(df['episode_index'].unique())

    print(f"Available episodes: {sorted(episodes)}")
    for ep in sorted(episodes):
        count = get_frame_count(dataset_path, ep)
        print(f"  Episode {ep}: {count} frames")


def load_intrinsics(intrinsics_path: Path, image_shape: tuple, calib_resolution: tuple = (640, 480)) -> dict:
    """Load camera intrinsics from file and scale to actual image resolution."""
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_path}")

    height, width = image_shape[:2]
    scale_x = width / calib_resolution[0]
    scale_y = height / calib_resolution[1]

    K = np.load(intrinsics_path)['K']

    return {
        'fx': K[0, 0] * scale_x,
        'fy': K[1, 1] * scale_y,
        'cx': K[0, 2] * scale_x,
        'cy': K[1, 2] * scale_y,
    }


def load_extrinsics(extrinsics_path: Path) -> dict:
    """Load camera extrinsics and compute camera-to-world transform."""
    if not extrinsics_path.exists():
        raise FileNotFoundError(f"Extrinsics file not found: {extrinsics_path}")

    data = np.load(extrinsics_path)
    R_world_to_cam, _ = cv2.Rodrigues(data['rvec'])
    t_world_to_cam = data['tvec'].flatten()

    # Invert to get camera-to-world
    R_cam_to_world = R_world_to_cam.T
    t_cam_to_world = -R_cam_to_world @ t_world_to_cam

    return {'R': R_cam_to_world, 't': t_cam_to_world}


def create_pixel_grid(height: int, width: int, downsample: int) -> tuple[np.ndarray, np.ndarray]:
    """Create downsampled pixel coordinate grid."""
    u = np.arange(0, width, downsample)
    v = np.arange(0, height, downsample)
    return np.meshgrid(u, v)


def project_to_camera_frame(
    depth: np.ndarray, intrinsics: dict, min_depth: int, max_depth: int, downsample: int
) -> tuple[np.ndarray, np.ndarray]:
    """Project depth to 3D points in camera frame."""
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    height, width = depth.shape
    u, v = create_pixel_grid(height, width, downsample)

    z = depth[::downsample, ::downsample].astype(np.float32)
    valid = (z > min_depth) & (z < max_depth)

    z_valid = z[valid] / 1000.0  # mm to meters
    u_valid = u[valid]
    v_valid = v[valid]

    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    x_cam = (u_valid - cx) * z_valid / fx
    y_cam = (v_valid - cy) * z_valid / fy

    points_cam = np.stack([x_cam, y_cam, z_valid], axis=-1)
    return points_cam, valid


def transform_to_world_frame(points_cam: np.ndarray, extrinsics: dict) -> np.ndarray:
    """Transform points from camera frame to world frame."""
    R, t = extrinsics['R'], extrinsics['t']
    return (R @ points_cam.T).T + t


def get_point_colors(
    rgb: np.ndarray | None, valid_mask: np.ndarray, depth_values: np.ndarray, downsample: int
) -> np.ndarray:
    """Get colors for points from RGB or depth colormap."""
    import matplotlib.pyplot as plt

    if rgb is not None:
        rgb_downsampled = rgb[::downsample, ::downsample]
        return rgb_downsampled[valid_mask].astype(np.float32) / 255.0

    # Fallback: colormap based on depth
    depth_normalized = (depth_values - depth_values.min()) / (depth_values.max() - depth_values.min() + 1e-6)
    return plt.cm.turbo(depth_normalized)[:, :3]


def depth_to_pointcloud(
    depth: np.ndarray,
    rgb: np.ndarray | None,
    intrinsics: dict,
    extrinsics: dict,
    min_depth: int,
    max_depth: int,
    downsample: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert depth image to 3D point cloud in world frame."""
    points_cam, valid_mask = project_to_camera_frame(depth, intrinsics, min_depth, max_depth, downsample)
    points_world = transform_to_world_frame(points_cam, extrinsics)

    # Get depth values for coloring (z component before transform)
    z_values = points_cam[:, 2]
    colors = get_point_colors(rgb, valid_mask, z_values, downsample)

    return points_world, colors

def print_pointcloud_stats(points: np.ndarray) -> None:
    """Print point cloud statistics."""
    print(f"\nPoint cloud: {len(points):,} points")
    print(f"  X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}] m")
    print(f"  Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}] m")
    print(f"  Z range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}] m")


def visualize_pointcloud(points: np.ndarray, colors: np.ndarray, title: str = "Point Cloud") -> None:
    """Visualize point cloud using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print_pointcloud_stats(points)

    print("\nOpen3D Controls:")
    print("  Mouse drag: Rotate | Scroll: Zoom | Shift+drag: Pan")
    print("  'R': Reset view | 'Q': Quit")

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    o3d.visualization.draw_geometries([pcd, coord_frame], window_name=title, width=1280, height=720)


def main():
    args = Args().parse_args()

    if not args.dataset.exists():
        print(f"Error: Dataset not found: {args.dataset}")
        return

    if args.list_episodes:
        list_episodes(args.dataset)
        return

    # Load frame
    print(f"Loading episode {args.episode}, frame {args.frame}...")
    rgb, depth = load_frame_data(args.dataset, args.episode, args.frame)

    if depth is None:
        print(f"Error: Could not load depth for episode {args.episode}, frame {args.frame}")
        frame_count = get_frame_count(args.dataset, args.episode)
        if frame_count > 0:
            print(f"Available frames: 0 to {frame_count - 1}")
        return

    # Load calibration
    intrinsics = load_intrinsics(args.intrinsics, depth.shape)
    extrinsics = load_extrinsics(args.extrinsics)

    # Create point cloud
    print(f"\nCreating point cloud...")
    print(f"  Depth range filter: {args.min_depth} - {args.max_depth} mm")
    print(f"  Downsample: {args.downsample}x")

    points, colors = depth_to_pointcloud(
        depth, rgb, intrinsics, extrinsics,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        downsample=args.downsample,
    )

    if len(points) == 0:
        print("Error: No valid points in depth range")
        return

    # Visualize
    title = f"Episode {args.episode}, Frame {args.frame}"
    visualize_pointcloud(points, colors, title)


if __name__ == "__main__":
    main()
