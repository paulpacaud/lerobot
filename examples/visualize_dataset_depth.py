#!/usr/bin/env python
"""
Visualize and sanity check a LeRobot dataset with depth data.

This script provides tools to:
- Load and inspect dataset metadata
- Visualize RGB and depth images side by side
- Show depth statistics and histograms
- Visualize 3D point clouds from RGB-D data (in world or camera frame)
- Navigate through episodes and frames
- Export sample visualizations

Usage:
    python examples/visualize_dataset_depth.py --dataset-path /path/to/dataset

    # With specific episode
    python examples/visualize_dataset_depth.py --dataset-path /path/to/dataset --episode 0

    # Save visualization to file
    python examples/visualize_dataset_depth.py --dataset-path /path/to/dataset --save-dir ./visualizations

    # Interactive mode with matplotlib
    python examples/visualize_dataset_depth.py --dataset-path /path/to/dataset --interactive

    # 3D point cloud visualization (uses intrinsics + extrinsics by default)
    python examples/visualize_dataset_depth.py --dataset-path /path/to/dataset --pointcloud

    # Point cloud with custom depth range (in mm)
    python examples/visualize_dataset_depth.py --dataset-path /path/to/dataset --pointcloud --max-depth 800

    # Point cloud in camera frame only (skip extrinsics)
    python examples/visualize_dataset_depth.py --dataset-path /path/to/dataset --pointcloud --camera-frame

    # Point cloud with custom calibration files
    python examples/visualize_dataset_depth.py --dataset-path /path/to/dataset --pointcloud \\
        --intrinsics /path/to/intrinsics.npz --extrinsics /path/to/extrinsics.npz

Calibration files:
    By default, uses calibration from examples/camera_calibration/:
    - intrinsics.npz: Camera matrix K (fx, fy, cx, cy) and distortion coefficients
    - extrinsics.npz: rvec and tvec for world-to-camera transformation
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Optional imports for point cloud visualization
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


def load_dataset_info(dataset_path: Path) -> dict:
    """Load dataset metadata from info.json."""
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset info not found at {info_path}")

    with open(info_path) as f:
        return json.load(f)


def get_image_keys(info: dict) -> tuple[list[str], list[str]]:
    """Extract RGB and depth image keys from dataset info."""
    rgb_keys = []
    depth_keys = []

    for key, feat in info["features"].items():
        if feat["dtype"] in ["video", "image"]:
            if feat.get("info", {}).get("is_depth_map", False):
                depth_keys.append(key)
            elif feat["shape"][-1] == 3:  # RGB has 3 channels
                rgb_keys.append(key)

    return rgb_keys, depth_keys


def load_intrinsics(intrinsics_path: Path | None, image_shape: tuple, calib_resolution: tuple = (640, 480)) -> dict:
    """Load camera intrinsics from file and scale to actual image resolution.

    Args:
        intrinsics_path: Path to intrinsics.npz file
        image_shape: Actual image shape (height, width, ...)
        calib_resolution: Resolution at which calibration was done (width, height).
                         Default is 640x480 for RealSense D400 series.

    Returns:
        dict with fx, fy, cx, cy, K matrix, and dist coefficients,
        all scaled to match the actual image resolution.
    """
    height, width = image_shape[:2]
    calib_width, calib_height = calib_resolution

    # Compute scale factors from calibration resolution to actual image resolution
    scale_x = width / calib_width
    scale_y = height / calib_height

    if intrinsics_path and intrinsics_path.exists():
        data = np.load(intrinsics_path)
        K_calib = data['K']
        dist = data.get('dist', np.zeros(5))

        # Scale intrinsics from calibration resolution to actual image resolution
        fx = K_calib[0, 0] * scale_x
        fy = K_calib[1, 1] * scale_y
        cx = K_calib[0, 2] * scale_x
        cy = K_calib[1, 2] * scale_y

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        return {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'K': K,
            'dist': dist
        }

    # Default intrinsics - estimate based on image size
    # These are approximate values for RealSense D400 series at 640x480
    # fx=601.8, fy=601.1, cx=329.5, cy=242.0 for 640x480
    fx = 601.8 * scale_x
    fy = 601.1 * scale_y
    cx = 329.5 * scale_x
    cy = 242.0 * scale_y

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    return {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'K': K,
        'dist': np.zeros(5)
    }


def load_extrinsics(extrinsics_path: Path | None) -> dict | None:
    """
    Load camera extrinsics from file.

    The extrinsics file contains rvec and tvec that transform from world to camera frame.
    We compute the inverse to get camera-to-world transformation.

    Returns:
        dict with 'R' (3x3 rotation matrix) and 't' (3x1 translation) for camera-to-world,
        or None if file doesn't exist.
    """
    if extrinsics_path is None or not extrinsics_path.exists():
        return None

    data = np.load(extrinsics_path)
    rvec = data['rvec']  # World to camera rotation (Rodrigues)
    tvec = data['tvec']  # World to camera translation

    # Convert Rodrigues vector to rotation matrix using OpenCV (world to camera)
    # This is consistent with how the calibration was done
    R_world_to_cam, _ = cv2.Rodrigues(rvec)
    t_world_to_cam = tvec.flatten()

    # Invert to get camera to world transformation
    # If P_cam = R_w2c @ P_world + t_w2c
    # Then P_world = R_w2c.T @ (P_cam - t_w2c) = R_w2c.T @ P_cam - R_w2c.T @ t_w2c
    R_cam_to_world = R_world_to_cam.T
    t_cam_to_world = -R_cam_to_world @ t_world_to_cam

    return {
        'R': R_cam_to_world,
        't': t_cam_to_world,
        'R_world_to_cam': R_world_to_cam,
        't_world_to_cam': t_world_to_cam
    }


def get_episode_start_index(dataset_path: Path, episode_idx: int) -> int:
    """Get the global starting frame index for an episode.

    Args:
        dataset_path: Path to the dataset
        episode_idx: Episode index

    Returns:
        Global frame index where this episode starts in the video
    """
    import pandas as pd

    # Try loading episode metadata
    episodes_dir = dataset_path / "meta" / "episodes"
    if episodes_dir.exists():
        for chunk_dir in sorted(episodes_dir.glob("chunk-*")):
            for parquet_file in sorted(chunk_dir.glob("file-*.parquet")):
                df = pd.read_parquet(parquet_file)
                if "episode_index" in df.columns and "dataset_from_index" in df.columns:
                    mask = df["episode_index"] == episode_idx
                    if mask.any():
                        return int(df[mask]["dataset_from_index"].iloc[0])

    # Fallback: compute from data parquet files
    data_dir = dataset_path / "data"
    if data_dir.exists():
        for chunk_dir in sorted(data_dir.glob("chunk-*")):
            for parquet_file in sorted(chunk_dir.glob("file-*.parquet")):
                df = pd.read_parquet(parquet_file, columns=["episode_index", "index"])
                mask = df["episode_index"] == episode_idx
                if mask.any():
                    return int(df[mask]["index"].min())

    return 0


def load_rgb_frame(dataset_path: Path, video_key: str, episode_idx: int, frame_idx: int) -> np.ndarray | None:
    """Load an RGB frame from video or image storage."""
    # Try image storage first
    img_path = dataset_path / "images" / video_key / f"episode-{episode_idx:06d}" / f"frame-{frame_idx:06d}.png"
    if img_path.exists():
        return np.array(Image.open(img_path))

    # Try to decode from video (requires additional dependencies)
    video_path = dataset_path / "videos" / video_key / "chunk-000" / "file-000.mp4"
    if video_path.exists():
        try:
            import av
            container = av.open(str(video_path))

            # Get the global frame index by adding episode start offset
            episode_start = get_episode_start_index(dataset_path, episode_idx)
            global_frame_idx = episode_start + frame_idx

            # Decode frames sequentially until we reach the target
            # (seeking in compressed video is approximate, so we count from start)
            for i, frame in enumerate(container.decode(video=0)):
                if i == global_frame_idx:
                    result = frame.to_ndarray(format='rgb24')
                    container.close()
                    return result
            container.close()
        except Exception as e:
            print(f"Warning: Could not decode video frame: {e}")

    return None


def load_depth_frame(dataset_path: Path, depth_key: str, episode_idx: int, frame_idx: int,
                      info: dict | None = None) -> np.ndarray | None:
    """Load a depth frame from parquet files or image storage.

    Args:
        dataset_path: Path to the dataset
        depth_key: Key for the depth feature (e.g., 'observation.images.front_depth')
        episode_idx: Episode index
        frame_idx: Frame index within the episode
        info: Dataset info dict (optional, used for parquet loading)

    Returns:
        Depth image as uint16 numpy array in millimeters, or None if not found
    """
    # Try loading from parquet files first (authoritative source)
    # The images folder may have stale/extra files from recording
    try:
        import pandas as pd
        import io

        # Find parquet files with the depth data
        data_dir = dataset_path / "data"
        if data_dir.exists():
            # Load info if not provided to get data path pattern
            if info is None:
                info = load_dataset_info(dataset_path)

            # Find the correct parquet file by iterating through chunks
            for chunk_dir in sorted(data_dir.glob("chunk-*")):
                for parquet_file in sorted(chunk_dir.glob("file-*.parquet")):
                    df = pd.read_parquet(parquet_file)

                    # Filter for the correct episode and frame
                    mask = (df["episode_index"] == episode_idx) & (df["frame_index"] == frame_idx)
                    if mask.any():
                        row = df[mask].iloc[0]
                        depth_data = row[depth_key]

                        # Handle dict format with 'bytes' key (PNG encoded)
                        if isinstance(depth_data, dict) and "bytes" in depth_data:
                            png_bytes = depth_data["bytes"]
                            img = Image.open(io.BytesIO(png_bytes))
                            return np.array(img, dtype=np.uint16)
                        # Handle direct numpy array
                        elif isinstance(depth_data, np.ndarray):
                            return depth_data.astype(np.uint16)

    except Exception as e:
        print(f"Warning: Could not load depth from parquet: {e}")

    # Fallback: try image storage (images/ folder)
    # Note: images folder may have stale files if recording was interrupted
    img_path = dataset_path / "images" / depth_key / f"episode-{episode_idx:06d}" / f"frame-{frame_idx:06d}.png"
    if img_path.exists():
        img = Image.open(img_path)
        return np.array(img, dtype=np.uint16)

    return None


def get_frame_count(dataset_path: Path, image_key: str, episode_idx: int, info: dict | None = None) -> int:
    """Get the number of frames in an episode for a given image key."""
    # Try to get from parquet files first (authoritative source)
    try:
        import pandas as pd
        data_dir = dataset_path / "data"
        if data_dir.exists():
            for chunk_dir in sorted(data_dir.glob("chunk-*")):
                for parquet_file in sorted(chunk_dir.glob("file-*.parquet")):
                    df = pd.read_parquet(parquet_file, columns=["episode_index"])
                    mask = df["episode_index"] == episode_idx
                    if mask.any():
                        return int(mask.sum())
    except Exception:
        pass

    # Fallback: try images folder (may have stale files)
    img_dir = dataset_path / "images" / image_key / f"episode-{episode_idx:06d}"
    if img_dir.exists():
        return len(list(img_dir.glob("*.png")))

    # Last fallback: use dataset info
    if info is not None:
        return info.get("total_frames", 0)

    return 0


def compute_depth_stats(depth: np.ndarray) -> dict:
    """Compute statistics for a depth image."""
    valid_mask = depth > 0  # 0 usually means invalid/no depth
    valid_depth = depth[valid_mask]

    if len(valid_depth) == 0:
        return {"valid_pixels": 0}

    return {
        "valid_pixels": int(valid_mask.sum()),
        "total_pixels": int(depth.size),
        "valid_ratio": float(valid_mask.sum() / depth.size),
        "min_mm": int(valid_depth.min()),
        "max_mm": int(valid_depth.max()),
        "mean_mm": float(valid_depth.mean()),
        "std_mm": float(valid_depth.std()),
        "median_mm": float(np.median(valid_depth)),
    }


def colorize_depth(depth: np.ndarray, min_depth: int = 0, max_depth: int = 10000) -> np.ndarray:
    """Convert depth image to colorized visualization using turbo colormap."""
    # Normalize depth to 0-1 range
    depth_normalized = np.clip(depth.astype(np.float32), min_depth, max_depth)
    depth_normalized = (depth_normalized - min_depth) / (max_depth - min_depth)

    # Apply colormap (turbo is good for depth visualization)
    cmap = plt.cm.turbo
    depth_colored = cmap(depth_normalized)[:, :, :3]  # Remove alpha channel

    # Mark invalid pixels (depth=0) as black
    invalid_mask = depth == 0
    depth_colored[invalid_mask] = 0

    return (depth_colored * 255).astype(np.uint8)


def depth_to_pointcloud(depth: np.ndarray, rgb: np.ndarray | None, intrinsics: dict,
                        extrinsics: dict | None = None,
                        min_depth: int = 100, max_depth: int = 10000,
                        downsample: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert depth image to 3D point cloud.

    Args:
        depth: Depth image (H, W) in millimeters
        rgb: RGB image (H, W, 3) for coloring points, or None
        intrinsics: Camera intrinsics dict with fx, fy, cx, cy
        extrinsics: Camera extrinsics dict with 'R' and 't' for camera-to-world transform, or None
        min_depth: Minimum valid depth in mm
        max_depth: Maximum valid depth in mm
        downsample: Downsample factor (1 = full resolution)

    Returns:
        points: (N, 3) array of 3D points in meters (world frame if extrinsics provided)
        colors: (N, 3) array of RGB colors normalized to [0, 1]
    """
    height, width = depth.shape

    # Create pixel coordinate grid
    u = np.arange(0, width, downsample)
    v = np.arange(0, height, downsample)
    u, v = np.meshgrid(u, v)

    # Get depth values (downsample if needed)
    z = depth[::downsample, ::downsample].astype(np.float32)

    # Create valid mask
    valid = (z > min_depth) & (z < max_depth)

    # Get intrinsics
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    # NOTE: Do NOT scale intrinsics - pixel coordinates are still in original image space
    # We're just sampling fewer pixels, not resizing the image

    # Convert to 3D points in camera frame (in meters)
    z_valid = z[valid] / 1000.0  # mm to meters
    u_valid = u[valid]
    v_valid = v[valid]

    x_cam = (u_valid - cx) * z_valid / fx
    y_cam = (v_valid - cy) * z_valid / fy
    z_cam = z_valid

    points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

    # Transform to world frame if extrinsics provided
    if extrinsics is not None:
        R = extrinsics['R']  # Camera to world rotation
        t = extrinsics['t']  # Camera to world translation
        # P_world = R @ P_cam + t
        points = (R @ points_cam.T).T + t
    else:
        points = points_cam

    # Get colors
    if rgb is not None:
        rgb_downsampled = rgb[::downsample, ::downsample]
        colors = rgb_downsampled[valid].astype(np.float32) / 255.0
    else:
        # Use depth-based coloring
        depth_normalized = (z_valid - z_valid.min()) / (z_valid.max() - z_valid.min() + 1e-6)
        colors = plt.cm.turbo(depth_normalized)[:, :3]

    return points, colors


def create_world_axes(origin=(0, 0, 0), size=0.1):
    """
    Create coordinate axes visualization at the world origin.

    Args:
        origin: (x, y, z) position of the origin
        size: Length of each axis in meters

    Returns:
        List of Open3D geometries (lines and spheres)
    """
    origin = np.array(origin)

    # Axis endpoints
    x_end = origin + np.array([size, 0, 0])
    y_end = origin + np.array([0, size, 0])
    z_end = origin + np.array([0, 0, size])

    # Create line set for axes
    points = [origin, x_end, y_end, z_end]
    lines = [[0, 1], [0, 2], [0, 3]]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # R, G, B for X, Y, Z

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    geometries = [line_set]

    # Spheres at axis ends for visibility
    sphere_radius = size * 0.1

    x_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    x_sphere.translate(x_end)
    x_sphere.paint_uniform_color([1, 0, 0])

    y_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    y_sphere.translate(y_end)
    y_sphere.paint_uniform_color([0, 1, 0])

    z_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    z_sphere.translate(z_end)
    z_sphere.paint_uniform_color([0, 0, 1])

    # Origin sphere (white)
    origin_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius * 1.2)
    origin_sphere.translate(origin)
    origin_sphere.paint_uniform_color([1, 1, 1])

    geometries.extend([x_sphere, y_sphere, z_sphere, origin_sphere])

    return geometries


def visualize_pointcloud_open3d(points: np.ndarray, colors: np.ndarray,
                                 window_name: str = "Point Cloud",
                                 show_world_frame: bool = True) -> None:
    """Visualize point cloud using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Print some sample points
    print("\nSample points (first 5):")
    for i in range(min(5, len(points))):
        print(f"  Point {i}: x={points[i, 0]:.4f}, y={points[i, 1]:.4f}, z={points[i, 2]:.4f} m")

    print("\nOpen3D Point Cloud Viewer Controls:")
    print("  Mouse drag: Rotate view")
    print("  Scroll: Zoom")
    print("  Shift+drag: Pan")
    print("  'R': Reset view")
    print("  'Q': Quit")
    print()

    geometries = [pcd]

    if show_world_frame:
        print("World frame at origin (0,0,0): X=Red, Y=Green, Z=Blue (50cm axes)")
        print()

        # Create coordinate frame at world origin (50cm axes)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        geometries.append(coord_frame)

    # Use simple draw_geometries for reliability
    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        width=1280,
        height=720
    )


def visualize_pointcloud_matplotlib(points: np.ndarray, colors: np.ndarray,
                                     title: str = "Point Cloud",
                                     save_path: Path | None = None) -> None:
    """Visualize point cloud using matplotlib (fallback)."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Subsample for matplotlib (it's slow with many points)
    max_points = 50000
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        colors = colors[indices]

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=colors, s=0.5, alpha=0.6)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)

    # Set equal aspect ratio
    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]).max() / 2.0

    mid_x = (points[:, 0].max() + points[:, 0].min()) / 2
    mid_y = (points[:, 1].max() + points[:, 1].min()) / 2
    mid_z = (points[:, 2].max() + points[:, 2].min()) / 2

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved point cloud visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_pointcloud(points: np.ndarray, colors: np.ndarray,
                         title: str = "Point Cloud",
                         save_path: Path | None = None,
                         use_open3d: bool = True,
                         show_world_frame: bool = True) -> None:
    """Visualize point cloud using best available method."""
    if use_open3d and HAS_OPEN3D and save_path is None:
        visualize_pointcloud_open3d(points, colors, title, show_world_frame)
    else:
        visualize_pointcloud_matplotlib(points, colors, title, save_path)


def visualize_frame(rgb: np.ndarray | None, depth: np.ndarray | None,
                    episode_idx: int, frame_idx: int,
                    depth_stats: dict | None = None,
                    save_path: Path | None = None) -> None:
    """Visualize RGB and depth side by side."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Episode {episode_idx}, Frame {frame_idx}", fontsize=14)

    # RGB image
    if rgb is not None:
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title("RGB Image")
        axes[0, 0].axis("off")
    else:
        axes[0, 0].text(0.5, 0.5, "RGB not available", ha='center', va='center')
        axes[0, 0].set_title("RGB Image")
        axes[0, 0].axis("off")

    # Depth colorized
    if depth is not None:
        depth_colored = colorize_depth(depth)
        axes[0, 1].imshow(depth_colored)
        axes[0, 1].set_title("Depth (colorized, turbo colormap)")
        axes[0, 1].axis("off")

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.turbo, norm=plt.Normalize(0, 10000))
        cbar = plt.colorbar(sm, ax=axes[0, 1], fraction=0.046, pad=0.04)
        cbar.set_label("Depth (mm)")
    else:
        axes[0, 1].text(0.5, 0.5, "Depth not available", ha='center', va='center')
        axes[0, 1].set_title("Depth (colorized)")
        axes[0, 1].axis("off")

    # Depth raw values visualization
    if depth is not None:
        axes[1, 0].imshow(depth, cmap='gray')
        axes[1, 0].set_title("Depth (raw grayscale)")
        axes[1, 0].axis("off")
    else:
        axes[1, 0].axis("off")

    # Depth histogram and stats
    if depth is not None and depth_stats is not None:
        valid_depth = depth[depth > 0]
        if len(valid_depth) > 0:
            axes[1, 1].hist(valid_depth.flatten(), bins=100, color='steelblue', alpha=0.7)
            axes[1, 1].set_xlabel("Depth (mm)")
            axes[1, 1].set_ylabel("Pixel count")
            axes[1, 1].set_title("Depth Distribution")

            # Add stats text
            stats_text = (
                f"Valid pixels: {depth_stats['valid_ratio']*100:.1f}%\n"
                f"Min: {depth_stats['min_mm']} mm\n"
                f"Max: {depth_stats['max_mm']} mm\n"
                f"Mean: {depth_stats['mean_mm']:.0f} mm\n"
                f"Median: {depth_stats['median_mm']:.0f} mm\n"
                f"Std: {depth_stats['std_mm']:.0f} mm"
            )
            axes[1, 1].text(0.98, 0.98, stats_text, transform=axes[1, 1].transAxes,
                          fontsize=9, verticalalignment='top', horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        axes[1, 1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def print_dataset_summary(dataset_path: Path, info: dict, rgb_keys: list, depth_keys: list) -> None:
    """Print a summary of the dataset."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Path: {dataset_path}")
    print(f"Robot type: {info.get('robot_type', 'unknown')}")
    print(f"Total episodes: {info.get('total_episodes', 0)}")
    print(f"Total frames: {info.get('total_frames', 0)}")
    print(f"FPS: {info.get('fps', 0)}")
    print()

    print("IMAGE FEATURES:")
    print("-" * 40)
    for key in rgb_keys:
        feat = info["features"][key]
        print(f"  {key}:")
        print(f"    Type: {feat['dtype']}")
        print(f"    Shape: {feat['shape']}")

    print()
    print("DEPTH FEATURES:")
    print("-" * 40)
    if depth_keys:
        for key in depth_keys:
            feat = info["features"][key]
            print(f"  {key}:")
            print(f"    Type: {feat['dtype']}")
            print(f"    Shape: {feat['shape']}")
            print(f"    Is depth map: {feat.get('info', {}).get('is_depth_map', False)}")
    else:
        print("  No depth features found!")

    print()
    print("OTHER FEATURES:")
    print("-" * 40)
    for key, feat in info["features"].items():
        if key not in rgb_keys and key not in depth_keys:
            if feat["dtype"] not in ["video", "image"]:
                print(f"  {key}: {feat['dtype']}, shape={feat.get('shape', 'N/A')}")

    print("=" * 60 + "\n")


def analyze_depth_quality(dataset_path: Path, depth_key: str, episode_idx: int,
                          sample_frames: int = 10, info: dict | None = None) -> None:
    """Analyze depth quality across multiple frames."""
    print(f"\nAnalyzing depth quality for episode {episode_idx}...")

    # First try images folder
    depth_dir = dataset_path / "images" / depth_key / f"episode-{episode_idx:06d}"
    use_images_folder = depth_dir.exists()

    if use_images_folder:
        frame_files = sorted(depth_dir.glob("*.png"))
        total_frames = len(frame_files)
    else:
        # Get frame count from parquet
        total_frames = get_frame_count(dataset_path, depth_key, episode_idx, info)

    print(f"Total depth frames: {total_frames}")

    if total_frames == 0:
        print(f"No depth frames found for episode {episode_idx}")
        return

    # Sample frames evenly
    indices = np.linspace(0, total_frames - 1, min(sample_frames, total_frames), dtype=int)

    all_stats = []
    for idx in indices:
        if use_images_folder:
            frame_path = frame_files[idx]
            depth = np.array(Image.open(frame_path), dtype=np.uint16)
        else:
            depth = load_depth_frame(dataset_path, depth_key, episode_idx, int(idx), info)

        if depth is not None:
            stats = compute_depth_stats(depth)
            stats["frame_idx"] = idx
            all_stats.append(stats)

    # Print summary statistics
    print("\nDepth Quality Summary:")
    print("-" * 50)

    valid_ratios = [s["valid_ratio"] for s in all_stats if "valid_ratio" in s]
    min_depths = [s["min_mm"] for s in all_stats if "min_mm" in s]
    max_depths = [s["max_mm"] for s in all_stats if "max_mm" in s]
    mean_depths = [s["mean_mm"] for s in all_stats if "mean_mm" in s]

    if valid_ratios:
        print(f"Valid pixel ratio: {np.mean(valid_ratios)*100:.1f}% (range: {np.min(valid_ratios)*100:.1f}% - {np.max(valid_ratios)*100:.1f}%)")
    if min_depths:
        print(f"Min depth: {np.min(min_depths)} - {np.max(min_depths)} mm")
    if max_depths:
        print(f"Max depth: {np.min(max_depths)} - {np.max(max_depths)} mm")
    if mean_depths:
        print(f"Mean depth: {np.mean(mean_depths):.0f} mm (std across frames: {np.std(mean_depths):.0f} mm)")


def interactive_viewer(dataset_path: Path, info: dict, rgb_keys: list, depth_keys: list,
                       start_episode: int = 0) -> None:
    """Interactive viewer using matplotlib."""
    print("\nInteractive Viewer Controls:")
    print("  'n' or Right Arrow: Next frame")
    print("  'p' or Left Arrow: Previous frame")
    print("  'N': Next episode")
    print("  'P': Previous episode")
    print("  'q': Quit")
    print()

    episode_idx = start_episode
    frame_idx = 0
    total_episodes = info.get("total_episodes", 1)

    rgb_key = rgb_keys[0] if rgb_keys else None
    depth_key = depth_keys[0] if depth_keys else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plt.ion()

    def update_display():
        nonlocal frame_idx

        axes[0].clear()
        axes[1].clear()

        # Load and display RGB
        if rgb_key:
            rgb = load_rgb_frame(dataset_path, rgb_key, episode_idx, frame_idx)
            if rgb is not None:
                axes[0].imshow(rgb)
            axes[0].set_title(f"RGB - Episode {episode_idx}, Frame {frame_idx}")
            axes[0].axis("off")

        # Load and display depth
        if depth_key:
            depth = load_depth_frame(dataset_path, depth_key, episode_idx, frame_idx, info)
            if depth is not None:
                depth_colored = colorize_depth(depth)
                axes[1].imshow(depth_colored)
                stats = compute_depth_stats(depth)
                axes[1].set_title(f"Depth - Mean: {stats.get('mean_mm', 0):.0f}mm, Valid: {stats.get('valid_ratio', 0)*100:.1f}%")
            else:
                axes[1].set_title(f"Depth - Frame {frame_idx} not found")
            axes[1].axis("off")

        fig.canvas.draw()
        fig.canvas.flush_events()

    def on_key(event):
        nonlocal episode_idx, frame_idx

        max_frames = get_frame_count(dataset_path, depth_key or rgb_key, episode_idx, info)

        if event.key in ['n', 'right']:
            frame_idx = min(frame_idx + 1, max_frames - 1)
        elif event.key in ['p', 'left']:
            frame_idx = max(frame_idx - 1, 0)
        elif event.key == 'N':
            episode_idx = min(episode_idx + 1, total_episodes - 1)
            frame_idx = 0
        elif event.key == 'P':
            episode_idx = max(episode_idx - 1, 0)
            frame_idx = 0
        elif event.key == 'q':
            plt.close(fig)
            return

        update_display()

    fig.canvas.mpl_connect('key_press_event', on_key)
    update_display()

    plt.show(block=True)


def main():
    parser = argparse.ArgumentParser(description="Visualize and sanity check a LeRobot dataset with depth data")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to the dataset directory")
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index to visualize (default: 0)")
    parser.add_argument("--frame", type=int, default=0,
                        help="Frame index to visualize (default: 0)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save visualizations (default: display only)")
    parser.add_argument("--interactive", action="store_true",
                        help="Launch interactive viewer")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze depth quality across frames")
    parser.add_argument("--sample-frames", type=int, default=5,
                        help="Number of frames to visualize when saving (default: 5)")

    # Point cloud options
    parser.add_argument("--pointcloud", action="store_true",
                        help="Visualize 3D point cloud")
    parser.add_argument("--intrinsics", type=str, default=None,
                        help="Path to camera intrinsics .npz file (default: examples/camera_calibration/intrinsics.npz)")
    parser.add_argument("--extrinsics", type=str, default=None,
                        help="Path to camera extrinsics .npz file (default: examples/camera_calibration/extrinsics.npz)")
    parser.add_argument("--camera-frame", action="store_true",
                        help="Keep point cloud in camera frame (don't apply extrinsics)")
    parser.add_argument("--min-depth", type=int, default=100,
                        help="Minimum depth in mm for point cloud (default: 100)")
    parser.add_argument("--max-depth", type=int, default=5000,
                        help="Maximum depth in mm for point cloud (default: 5000)")
    parser.add_argument("--downsample", type=int, default=2,
                        help="Downsample factor for point cloud (default: 2)")
    parser.add_argument("--calib-resolution", type=str, default="640x480",
                        help="Resolution at which camera was calibrated, e.g., '640x480' (default: 640x480)")

    args = parser.parse_args()

    # Parse calibration resolution
    calib_parts = args.calib_resolution.lower().split('x')
    if len(calib_parts) != 2:
        print(f"Error: Invalid calibration resolution format '{args.calib_resolution}'. Use 'WIDTHxHEIGHT', e.g., '640x480'")
        return
    args.calib_resolution = (int(calib_parts[0]), int(calib_parts[1]))

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return

    # Load dataset info
    info = load_dataset_info(dataset_path)
    rgb_keys, depth_keys = get_image_keys(info)

    # Print summary
    print_dataset_summary(dataset_path, info, rgb_keys, depth_keys)

    if not depth_keys:
        print("Warning: No depth features found in this dataset!")
        print("Make sure you recorded with 'use_depth: true' in camera config.")
        return

    # Analyze depth quality if requested
    if args.analyze:
        for depth_key in depth_keys:
            analyze_depth_quality(dataset_path, depth_key, args.episode, info=info)

    # Interactive mode
    if args.interactive:
        interactive_viewer(dataset_path, info, rgb_keys, depth_keys, args.episode)
        return

    # Point cloud visualization
    if args.pointcloud:
        rgb_key = rgb_keys[0] if rgb_keys else None
        depth_key = depth_keys[0] if depth_keys else None

        # Load frames
        rgb = load_rgb_frame(dataset_path, rgb_key, args.episode, args.frame) if rgb_key else None
        depth = load_depth_frame(dataset_path, depth_key, args.episode, args.frame, info)

        if depth is None:
            print(f"Error: Could not load depth frame {args.frame} from episode {args.episode}")
            return

        # Load intrinsics (scaled to actual image resolution)
        intrinsics_path = Path(args.intrinsics) if args.intrinsics else Path("examples/camera_calibration/intrinsics.npz")
        intrinsics = load_intrinsics(
            intrinsics_path if intrinsics_path.exists() else None,
            depth.shape,
            calib_resolution=args.calib_resolution
        )

        print(f"\nCamera intrinsics (scaled from {args.calib_resolution[0]}x{args.calib_resolution[1]} to {depth.shape[1]}x{depth.shape[0]}):")
        print(f"  fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}")
        print(f"  cx={intrinsics['cx']:.1f}, cy={intrinsics['cy']:.1f}")

        # Load extrinsics (for camera-to-world transformation)
        extrinsics = None
        if not args.camera_frame:
            extrinsics_path = Path(args.extrinsics) if args.extrinsics else Path("examples/camera_calibration/extrinsics.npz")
            extrinsics = load_extrinsics(extrinsics_path if extrinsics_path.exists() else None)

            if extrinsics is not None:
                print(f"\nCamera extrinsics (camera-to-world):")
                print(f"  Translation: [{extrinsics['t'][0]:.3f}, {extrinsics['t'][1]:.3f}, {extrinsics['t'][2]:.3f}] m")
                print(f"  Rotation matrix:")
                for row in extrinsics['R']:
                    print(f"    [{row[0]:7.4f}, {row[1]:7.4f}, {row[2]:7.4f}]")
                print(f"  Point cloud will be in WORLD frame")
            else:
                print(f"\nNo extrinsics file found. Point cloud will be in CAMERA frame.")
        else:
            print(f"\n--camera-frame specified. Point cloud will be in CAMERA frame.")

        # Create point cloud
        print(f"\nCreating point cloud from episode {args.episode}, frame {args.frame}...")
        print(f"  Depth range: {args.min_depth} - {args.max_depth} mm")
        print(f"  Downsample factor: {args.downsample}")

        points, colors = depth_to_pointcloud(
            depth, rgb, intrinsics,
            extrinsics=extrinsics,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            downsample=args.downsample
        )

        print(f"  Generated {len(points):,} points")

        if len(points) == 0:
            print("Error: No valid points in depth range")
            return

        # Visualize
        frame_type = "World" if extrinsics else "Camera"
        title = f"Point Cloud ({frame_type} Frame) - Episode {args.episode}, Frame {args.frame}"
        save_path = None
        if args.save_dir:
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"pointcloud_ep{args.episode:03d}_frame{args.frame:06d}.png"

        if not HAS_OPEN3D:
            print("\nNote: Open3D not installed. Using matplotlib (slower, limited interaction).")
            print("Install Open3D for better visualization: pip install open3d")

        # Show world frame only if we're in world coordinates (extrinsics applied)
        visualize_pointcloud(points, colors, title, save_path, show_world_frame=(extrinsics is not None))
        return

    # Static visualization
    rgb_key = rgb_keys[0] if rgb_keys else None
    depth_key = depth_keys[0] if depth_keys else None

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Get total frames for this episode
        depth_dir = dataset_path / "images" / depth_key / f"episode-{args.episode:06d}"
        if depth_dir.exists():
            total_frames = len(list(depth_dir.glob("*.png")))
        else:
            total_frames = info.get("total_frames", 0)

        # Sample frames evenly
        frame_indices = np.linspace(0, total_frames - 1, args.sample_frames, dtype=int)

        for frame_idx in frame_indices:
            rgb = load_rgb_frame(dataset_path, rgb_key, args.episode, frame_idx) if rgb_key else None
            depth = load_depth_frame(dataset_path, depth_key, args.episode, frame_idx, info) if depth_key else None
            depth_stats = compute_depth_stats(depth) if depth is not None else None

            save_path = save_dir / f"episode_{args.episode:03d}_frame_{frame_idx:06d}.png"
            visualize_frame(rgb, depth, args.episode, frame_idx, depth_stats, save_path)

        print(f"\nSaved {len(frame_indices)} visualizations to {save_dir}")
    else:
        # Single frame visualization
        rgb = load_rgb_frame(dataset_path, rgb_key, args.episode, args.frame) if rgb_key else None
        depth = load_depth_frame(dataset_path, depth_key, args.episode, args.frame, info) if depth_key else None
        depth_stats = compute_depth_stats(depth) if depth is not None else None

        visualize_frame(rgb, depth, args.episode, args.frame, depth_stats)


if __name__ == "__main__":
    main()
