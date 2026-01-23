"""
Visualize post-processed point clouds from a LeRobot dataset.

This script reads point clouds from the LMDB database and displays them
in an interactive 3D viewer with coordinate axes at the origin.

Usage:
```bash
python examples/post_process_dataset/visualize_postprocessed_pcd.py \
    --dataset-dir=$HOME/lerobot_datasets/depth_test \
    --episode-index=0 \
    --frame-index=100
```
"""

from pathlib import Path

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import open3d as o3d
from tap import Tap

msgpack_numpy.patch()


class Args(Tap):
    dataset_dir: str  # Path to the LeRobot dataset directory

    # Frame selection
    episode_index: int = 0  # Episode index to visualize
    frame_index: int = 0  # Frame index to visualize

    # Rendering options
    axes_size: float = 0.1  # Size of the coordinate axes (meters)
    point_size: float = 4.0  # Rendered point size


def create_coordinate_axes(size: float = 0.1, origin: tuple = (0, 0, 0)) -> o3d.geometry.TriangleMesh:
    """
    Create a coordinate frame (XYZ axes) at the specified origin.

    Args:
        size: Length of each axis
        origin: Position of the coordinate frame origin

    Returns:
        Open3D TriangleMesh representing the coordinate frame
    """
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
    return axes


def load_point_cloud_from_lmdb(
    lmdb_path: Path,
    episode_index: int,
    frame_index: int
) -> np.ndarray:
    """
    Load a point cloud from LMDB database.

    Args:
        lmdb_path: Path to the LMDB database directory
        episode_index: Episode index
        frame_index: Frame index within the episode

    Returns:
        Point cloud array of shape (N, 6) with [x, y, z, r, g, b]
    """
    lmdb_env = lmdb.open(str(lmdb_path), readonly=True)

    with lmdb_env.begin() as txn:
        key = f"{episode_index}-{frame_index}"
        pc_bytes = txn.get(key.encode('ascii'))

        if pc_bytes is None:
            lmdb_env.close()
            raise ValueError(f"Point cloud not found for key '{key}'")

        point_cloud = msgpack.unpackb(pc_bytes)

    lmdb_env.close()
    return point_cloud


def visualize_point_cloud(
    point_cloud: np.ndarray,
    window_name: str = "Point Cloud Viewer",
    axes_size: float = 0.1,
    point_size: float = 2.0,
):
    """
    Visualize a point cloud in 3D with coordinate axes.

    Args:
        point_cloud: Array of shape (N, 6) with [x, y, z, r, g, b]
        window_name: Title of the visualization window
        axes_size: Size of the coordinate axes
        point_size: Size of rendered points
    """
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6])

    # Create coordinate axes at origin
    axes = create_coordinate_axes(size=axes_size, origin=(0, 0, 0))

    # Print point cloud statistics
    print(f"\nPoint cloud statistics:")
    print(f"  Number of points: {len(point_cloud)}")
    print(f"  X range: [{point_cloud[:, 0].min():.3f}, {point_cloud[:, 0].max():.3f}]")
    print(f"  Y range: [{point_cloud[:, 1].min():.3f}, {point_cloud[:, 1].max():.3f}]")
    print(f"  Z range: [{point_cloud[:, 2].min():.3f}, {point_cloud[:, 2].max():.3f}]")

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)

    # Add geometries
    vis.add_geometry(pcd)
    vis.add_geometry(axes)

    # Set render options
    render_opt = vis.get_render_option()
    render_opt.point_size = point_size
    render_opt.background_color = np.array([1.0, 1.0, 1.0])  # White background

    # Set initial viewpoint (looking at the scene from above and front)
    view_ctl = vis.get_view_control()
    view_ctl.set_zoom(0.5)
    view_ctl.set_front([0.5, -0.5, 0.7])
    view_ctl.set_lookat([0, 0, 0])
    view_ctl.set_up([0, 0, 1])

    print(f"\nVisualization controls:")
    print(f"  Left mouse: Rotate view")
    print(f"  Right mouse: Pan view")
    print(f"  Scroll: Zoom in/out")
    print(f"  Q: Quit")

    # Run visualizer
    vis.run()
    vis.destroy_window()


def main():
    args = Args().parse_args()

    dataset_path = Path(args.dataset_dir)
    lmdb_path = dataset_path / "point_clouds"

    if not lmdb_path.exists():
        raise FileNotFoundError(
            f"Point cloud LMDB not found at {lmdb_path}. "
            "Run add_point_cloud_to_dataset.py first."
        )

    print(f"Loading point cloud: episode {args.episode_index}, frame {args.frame_index}")
    point_cloud = load_point_cloud_from_lmdb(
        lmdb_path, args.episode_index, args.frame_index
    )

    visualize_point_cloud(
        point_cloud=point_cloud,
        window_name=f"Episode {args.episode_index}, Frame {args.frame_index}",
        axes_size=args.axes_size,
        point_size=args.point_size,
    )


if __name__ == "__main__":
    main()
