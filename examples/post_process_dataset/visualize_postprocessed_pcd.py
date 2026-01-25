"""
Visualize post-processed point clouds from a LeRobot dataset.

This script reads point clouds from the LMDB database and displays them
in an interactive 3D viewer with coordinate axes at the origin.

Optionally shows the end-effector position as a blue sphere if EE data
is available (either from converted EE-space dataset or computed via FK).

Usage:
```bash
# Basic point cloud visualization
python examples/post_process_dataset/visualize_postprocessed_pcd.py \
    --dataset_dir=$HOME/lerobot_datasets/depth_test \
    --episode_index=0 \
    --frame_index=100

# With end-effector visualization (EE-space dataset)
python examples/post_process_dataset/visualize_postprocessed_pcd.py \
    --dataset_dir=$HOME/lerobot_datasets/depth_test_ee \
    --episode_index=0 \
    --frame_index=100 \
    --show_ee

# With end-effector computed from joint-space dataset
python examples/post_process_dataset/visualize_postprocessed_pcd.py \
    --dataset_dir=$HOME/lerobot_datasets/depth_test_v2 \
    --episode_index=0 \
    --frame_index=100 \
    --show_ee \
    --urdf_path=./examples/post_process_dataset/constants/SO100/so100.urdf
```
"""

from pathlib import Path

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import open3d as o3d
import pandas as pd
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

    # End-effector visualization
    show_ee: bool = False  # Show end-effector position as blue sphere
    urdf_path: str | None = None  # Path to URDF (for FK if dataset is joint-space)
    ee_radius: float = 0.015  # Radius of the EE sphere (meters)


def load_ee_position(
    dataset_path: Path,
    episode_index: int,
    frame_index: int,
    urdf_path: str | None = None,
) -> np.ndarray | None:
    """
    Load or compute the end-effector position for a given frame.

    Args:
        dataset_path: Path to the dataset
        episode_index: Episode index
        frame_index: Frame index
        urdf_path: Path to URDF file (required if dataset is joint-space)

    Returns:
        EE position [x, y, z] in meters, or None if not available
    """
    # Try to find parquet file (v2.1 format)
    parquet_path = dataset_path / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    if not parquet_path.exists():
        # Try v3.0 format
        parquet_path = dataset_path / "data" / "chunk-000" / "file-000.parquet"

    if not parquet_path.exists():
        print(f"Warning: Could not find parquet file for episode {episode_index}")
        return None

    df = pd.read_parquet(parquet_path)

    # Filter by episode if v3.0 format
    if "episode_index" in df.columns and df["episode_index"].nunique() > 1:
        df = df[df["episode_index"] == episode_index]

    if frame_index >= len(df):
        print(f"Warning: Frame index {frame_index} out of range (max {len(df)-1})")
        return None

    state = np.array(df.iloc[frame_index]["observation.state"])

    # Check if state is already EE-space (7D) or joint-space (6D)
    if len(state) == 7:
        # EE-space: [x, y, z, rx, ry, rz, gripper]
        return state[:3]
    elif len(state) == 6 and urdf_path is not None:
        # Joint-space: compute FK
        try:
            from lerobot.model.kinematics import RobotKinematics

            # Convert to absolute path for placo
            urdf_abs_path = str(Path(urdf_path).resolve())

            kinematics = RobotKinematics(
                urdf_path=urdf_abs_path,
                target_frame_name="jaw",
                joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            )
            T = kinematics.forward_kinematics(state[:5].astype(np.float64))
            return T[:3, 3]
        except Exception as e:
            print(f"Warning: Could not compute FK: {e}")
            return None
    else:
        print(f"Warning: State has {len(state)} dims. Provide --urdf-path for joint-space data.")
        return None


def create_ee_sphere(position: np.ndarray, radius: float = 0.015) -> o3d.geometry.TriangleMesh:
    """
    Create a blue sphere at the end-effector position.

    Args:
        position: [x, y, z] position in meters
        radius: Sphere radius in meters

    Returns:
        Open3D TriangleMesh sphere
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(position)
    sphere.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
    sphere.compute_vertex_normals()
    return sphere


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
    ee_position: np.ndarray | None = None,
    ee_radius: float = 0.015,
):
    """
    Visualize a point cloud in 3D with coordinate axes and optional EE marker.

    Args:
        point_cloud: Array of shape (N, 6) with [x, y, z, r, g, b]
        window_name: Title of the visualization window
        axes_size: Size of the coordinate axes
        point_size: Size of rendered points
        ee_position: Optional [x, y, z] position to show as blue sphere
        ee_radius: Radius of the EE sphere
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

    if ee_position is not None:
        print(f"\nEnd-effector position:")
        print(f"  X: {ee_position[0]:.4f} m")
        print(f"  Y: {ee_position[1]:.4f} m")
        print(f"  Z: {ee_position[2]:.4f} m")

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)

    # Add geometries
    vis.add_geometry(pcd)
    vis.add_geometry(axes)

    # Add EE sphere if position provided
    if ee_position is not None:
        ee_sphere = create_ee_sphere(ee_position, radius=ee_radius)
        vis.add_geometry(ee_sphere)

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

    # Load EE position if requested
    ee_position = None
    if args.show_ee:
        print("Loading end-effector position...")
        ee_position = load_ee_position(
            dataset_path,
            args.episode_index,
            args.frame_index,
            urdf_path=args.urdf_path,
        )

    visualize_point_cloud(
        point_cloud=point_cloud,
        window_name=f"Episode {args.episode_index}, Frame {args.frame_index}",
        axes_size=args.axes_size,
        point_size=args.point_size,
        ee_position=ee_position,
        ee_radius=args.ee_radius,
    )


if __name__ == "__main__":
    main()
