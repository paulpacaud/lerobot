#!/usr/bin/env python
"""
Visualize point cloud from captured debug camera frames.

Similar to visualize_dataset_depth.py but for the debug camera output.
Shows RGB+depth as point cloud with world origin to check alignment.

Usage:
    python examples/debug_camera_depth/visualize_captured_pointcloud.py debug_camera_output --frame 50
    python examples/debug_camera_depth/visualize_captured_pointcloud.py debug_camera_output --frame 50 --max-depth 800

    # With custom calibration
    python examples/debug_camera_depth/visualize_captured_pointcloud.py debug_camera_output --frame 50 \
        --intrinsics examples/camera_calibration/intrinsics.npz \
        --extrinsics examples/camera_calibration/extrinsics.npz
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: Open3D not installed. Install with: pip install open3d")


def load_intrinsics(intrinsics_path: Path | None, image_shape: tuple,
                    calib_resolution: tuple = (640, 480)) -> dict:
    """Load camera intrinsics from file and scale to actual image resolution."""
    height, width = image_shape[:2]
    calib_width, calib_height = calib_resolution

    scale_x = width / calib_width
    scale_y = height / calib_height

    if intrinsics_path and intrinsics_path.exists():
        data = np.load(intrinsics_path)
        K_calib = data['K']
        dist = data.get('dist', np.zeros(5))

        fx = K_calib[0, 0] * scale_x
        fy = K_calib[1, 1] * scale_y
        cx = K_calib[0, 2] * scale_x
        cy = K_calib[1, 2] * scale_y
    else:
        # Default RealSense D400 intrinsics at 640x480
        fx = 601.8 * scale_x
        fy = 601.1 * scale_y
        cx = 329.5 * scale_x
        cy = 242.0 * scale_y
        dist = np.zeros(5)

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'K': K, 'dist': dist}


def load_extrinsics(extrinsics_path: Path | None) -> dict | None:
    """Load camera extrinsics and compute camera-to-world transform."""
    if extrinsics_path is None or not extrinsics_path.exists():
        return None

    data = np.load(extrinsics_path)
    rvec = data['rvec']
    tvec = data['tvec']

    R_world_to_cam, _ = cv2.Rodrigues(rvec)
    t_world_to_cam = tvec.flatten()

    # Invert to get camera-to-world
    R_cam_to_world = R_world_to_cam.T
    t_cam_to_world = -R_cam_to_world @ t_world_to_cam

    return {
        'R': R_cam_to_world,
        't': t_cam_to_world,
    }


def depth_to_pointcloud(depth: np.ndarray, rgb: np.ndarray | None, intrinsics: dict,
                        extrinsics: dict | None = None,
                        min_depth: int = 100, max_depth: int = 10000,
                        downsample: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Convert depth image to 3D point cloud."""
    height, width = depth.shape

    u = np.arange(0, width, downsample)
    v = np.arange(0, height, downsample)
    u, v = np.meshgrid(u, v)

    z = depth[::downsample, ::downsample].astype(np.float32)
    valid = (z > min_depth) & (z < max_depth)

    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    z_valid = z[valid] / 1000.0  # mm to meters
    u_valid = u[valid]
    v_valid = v[valid]

    x_cam = (u_valid - cx) * z_valid / fx
    y_cam = (v_valid - cy) * z_valid / fy
    z_cam = z_valid

    points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

    if extrinsics is not None:
        R = extrinsics['R']
        t = extrinsics['t']
        points = (R @ points_cam.T).T + t
    else:
        points = points_cam

    if rgb is not None:
        rgb_downsampled = rgb[::downsample, ::downsample]
        colors = rgb_downsampled[valid].astype(np.float32) / 255.0
    else:
        import matplotlib.pyplot as plt
        depth_normalized = (z_valid - z_valid.min()) / (z_valid.max() - z_valid.min() + 1e-6)
        colors = plt.cm.turbo(depth_normalized)[:, :3]

    return points, colors


def visualize_pointcloud_open3d(points: np.ndarray, colors: np.ndarray,
                                 title: str = "Point Cloud",
                                 show_world_frame: bool = True) -> None:
    """Visualize point cloud using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"\nPoint cloud: {len(points):,} points")
    print(f"  X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}] m")
    print(f"  Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}] m")
    print(f"  Z range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}] m")

    print("\nOpen3D Controls:")
    print("  Mouse drag: Rotate | Scroll: Zoom | Shift+drag: Pan")
    print("  'R': Reset view | 'Q': Quit")

    geometries = [pcd]

    if show_world_frame:
        print("\nWorld frame at origin (0,0,0): X=Red, Y=Green, Z=Blue (50cm axes)")
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        geometries.append(coord_frame)

    o3d.visualization.draw_geometries(geometries, window_name=title, width=1280, height=720)


def load_frame(output_dir: Path, frame_idx: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load RGB and depth frame from debug output directory."""
    rgb_path = output_dir / "rgb" / f"frame_{frame_idx:06d}.png"
    depth_path = output_dir / "depth" / f"frame_{frame_idx:06d}.png"

    rgb = None
    depth = None

    if rgb_path.exists():
        rgb = np.array(Image.open(rgb_path))
    else:
        print(f"Warning: RGB frame not found: {rgb_path}")

    if depth_path.exists():
        depth = np.array(Image.open(depth_path), dtype=np.uint16)
    else:
        print(f"Warning: Depth frame not found: {depth_path}")

    return rgb, depth


def get_available_frames(output_dir: Path) -> list[int]:
    """Get list of available frame indices."""
    depth_dir = output_dir / "depth"
    if not depth_dir.exists():
        return []

    frames = []
    for f in sorted(depth_dir.glob("frame_*.png")):
        try:
            idx = int(f.stem.split("_")[1])
            frames.append(idx)
        except (ValueError, IndexError):
            pass
    return frames


def main():
    parser = argparse.ArgumentParser(description="Visualize point cloud from captured debug frames")
    parser.add_argument("output_dir", type=str, help="Path to debug camera output directory")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize")
    parser.add_argument("--intrinsics", type=str, default=None,
                        help="Path to intrinsics.npz (default: examples/camera_calibration/intrinsics.npz)")
    parser.add_argument("--extrinsics", type=str, default=None,
                        help="Path to extrinsics.npz (default: examples/camera_calibration/extrinsics.npz)")
    parser.add_argument("--camera-frame", action="store_true",
                        help="Keep point cloud in camera frame (don't apply extrinsics)")
    parser.add_argument("--min-depth", type=int, default=100, help="Minimum depth in mm (default: 100)")
    parser.add_argument("--max-depth", type=int, default=5000, help="Maximum depth in mm (default: 5000)")
    parser.add_argument("--downsample", type=int, default=2, help="Downsample factor (default: 2)")
    parser.add_argument("--list-frames", action="store_true", help="List available frames and exit")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        return

    # List frames mode
    if args.list_frames:
        frames = get_available_frames(output_dir)
        if frames:
            print(f"Available frames: {frames[0]} to {frames[-1]} ({len(frames)} total)")
        else:
            print("No frames found")
        return

    if not HAS_OPEN3D:
        print("Error: Open3D is required. Install with: pip install open3d")
        return

    # Load frame
    rgb, depth = load_frame(output_dir, args.frame)
    if depth is None:
        print(f"Error: Could not load depth frame {args.frame}")
        available = get_available_frames(output_dir)
        if available:
            print(f"Available frames: {available[0]} to {available[-1]}")
        return

    print(f"\nLoaded frame {args.frame}:")
    print(f"  RGB shape: {rgb.shape if rgb is not None else 'N/A'}")
    print(f"  Depth shape: {depth.shape}")

    # Depth stats
    valid_depth = depth[depth > 0]
    print(f"  Depth range: {valid_depth.min()} - {valid_depth.max()} mm")
    print(f"  Depth mean: {valid_depth.mean():.0f} mm")
    print(f"  Valid pixels: {len(valid_depth)} / {depth.size} ({100*len(valid_depth)/depth.size:.1f}%)")

    # Load intrinsics
    intrinsics_path = Path(args.intrinsics) if args.intrinsics else Path("examples/camera_calibration/intrinsics.npz")
    intrinsics = load_intrinsics(intrinsics_path if intrinsics_path.exists() else None, depth.shape)

    print(f"\nCamera intrinsics:")
    print(f"  fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}")
    print(f"  cx={intrinsics['cx']:.1f}, cy={intrinsics['cy']:.1f}")

    # Load extrinsics
    extrinsics = None
    if not args.camera_frame:
        extrinsics_path = Path(args.extrinsics) if args.extrinsics else Path("examples/camera_calibration/extrinsics.npz")
        extrinsics = load_extrinsics(extrinsics_path if extrinsics_path.exists() else None)

        if extrinsics is not None:
            print(f"\nCamera extrinsics (camera-to-world):")
            print(f"  Translation: [{extrinsics['t'][0]:.3f}, {extrinsics['t'][1]:.3f}, {extrinsics['t'][2]:.3f}] m")
            print(f"  Point cloud will be in WORLD frame")
        else:
            print(f"\nNo extrinsics found. Point cloud will be in CAMERA frame.")
    else:
        print(f"\n--camera-frame specified. Point cloud will be in CAMERA frame.")

    # Create point cloud
    print(f"\nCreating point cloud...")
    print(f"  Depth range filter: {args.min_depth} - {args.max_depth} mm")
    print(f"  Downsample: {args.downsample}x")

    points, colors = depth_to_pointcloud(
        depth, rgb, intrinsics,
        extrinsics=extrinsics,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        downsample=args.downsample
    )

    if len(points) == 0:
        print("Error: No valid points in depth range")
        return

    # Visualize
    frame_type = "World" if extrinsics else "Camera"
    title = f"Debug Capture - Frame {args.frame} ({frame_type} Frame)"
    visualize_pointcloud_open3d(points, colors, title, show_world_frame=(extrinsics is not None))


if __name__ == "__main__":
    main()
