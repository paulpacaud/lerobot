#!/usr/bin/env python
"""
Visualize EE trajectory overlaid on point cloud with adjustable transform.

Shows the full EE trajectory with an applied x, y, z translation offset.
Useful for manually finding the robot-to-world transform.

Usage:
```bash
# No offset (original)
python examples/post_process_dataset/visualize_ee_trajectory_with_transform.py \
    --dataset_dir=/home/ppacaud/lerobot_datasets/depth_test_ee \
    --episode_index=0

# With translation offset
python examples/post_process_dataset/visualize_ee_trajectory_with_transform.py \
    --dataset_dir=/home/ppacaud/lerobot_datasets/depth_test_ee \
    --episode_index=0 \
    --tx=-0.10 --ty=-0.05 --tz=0.05
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
    dataset_dir: str  # Path to the EE-space dataset
    episode_index: int = 0  # Episode to visualize
    pcd_frame: int = 0  # Which frame's point cloud to show as background

    # Translation offset (robot frame to world frame)
    tx: float = 0.0  # X translation (meters)
    ty: float = 0.0  # Y translation (meters)
    tz: float = 0.0  # Z translation (meters)


def main():
    args = Args().parse_args()
    dataset_path = Path(args.dataset_dir)

    # Load EE positions
    parquet_path = dataset_path / "data" / "chunk-000" / f"episode_{args.episode_index:06d}.parquet"
    if not parquet_path.exists():
        parquet_path = dataset_path / "data" / "chunk-000" / "file-000.parquet"

    df = pd.read_parquet(parquet_path)
    if "episode_index" in df.columns:
        df = df[df["episode_index"] == args.episode_index]

    ee_all = np.stack(df["observation.state"].values)[:, :3]
    print(f"Loaded {len(ee_all)} EE positions")

    # Apply translation offset
    offset = np.array([args.tx, args.ty, args.tz])
    ee_transformed = ee_all + offset

    print(f"\nTranslation offset: [{args.tx:.4f}, {args.ty:.4f}, {args.tz:.4f}]")
    print(f"Original EE range:")
    print(f"  X: [{ee_all[:, 0].min():.4f}, {ee_all[:, 0].max():.4f}]")
    print(f"  Y: [{ee_all[:, 1].min():.4f}, {ee_all[:, 1].max():.4f}]")
    print(f"  Z: [{ee_all[:, 2].min():.4f}, {ee_all[:, 2].max():.4f}]")
    print(f"Transformed EE range:")
    print(f"  X: [{ee_transformed[:, 0].min():.4f}, {ee_transformed[:, 0].max():.4f}]")
    print(f"  Y: [{ee_transformed[:, 1].min():.4f}, {ee_transformed[:, 1].max():.4f}]")
    print(f"  Z: [{ee_transformed[:, 2].min():.4f}, {ee_transformed[:, 2].max():.4f}]")

    # Load point cloud
    lmdb_path = dataset_path / "point_clouds"
    env = lmdb.open(str(lmdb_path), readonly=True)
    with env.begin() as txn:
        key = f"{args.episode_index}-{args.pcd_frame}"
        pc = np.array(msgpack.unpackb(txn.get(key.encode("ascii"))))
    env.close()
    print(f"\nLoaded point cloud: {len(pc)} points")
    print(f"Point cloud range:")
    print(f"  X: [{pc[:, 0].min():.4f}, {pc[:, 0].max():.4f}]")
    print(f"  Y: [{pc[:, 1].min():.4f}, {pc[:, 1].max():.4f}]")
    print(f"  Z: [{pc[:, 2].min():.4f}, {pc[:, 2].max():.4f}]")

    # Create point cloud geometry
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:6])

    # Create trajectory as line set (subsample for performance)
    trajectory_points = ee_transformed[::3]
    lines = [[i, i + 1] for i in range(len(trajectory_points) - 1)]
    colors = [[1, 0, 0] for _ in lines]  # red

    trajectory = o3d.geometry.LineSet()
    trajectory.points = o3d.utility.Vector3dVector(trajectory_points)
    trajectory.lines = o3d.utility.Vector2iVector(lines)
    trajectory.colors = o3d.utility.Vector3dVector(colors)

    # Create spheres at key frames
    n_frames = len(ee_transformed)
    key_frames = [i * n_frames // 9 for i in range(10)]
    key_frames[-1] -= 1
    sphere_colors = [
        [0, 1, 0],      # Green
        [0, 0.5, 1],    # Cyan
        [0, 0, 1],      # Blue
        [0.5, 0, 1],    # Purple
        [1, 0, 1],      # Magenta
        [1, 0, 0.5],    # Pink
        [1, 0, 0],      # Red
        [1, 0.5, 0],    # Orange
        [1, 1, 0],      # Yellow
        [0, 1, 1],      # Teal
    ]

    spheres = []
    print("\nKey frames (colored spheres):")
    color_names = ["Green", "Cyan", "Blue", "Purple", "Magenta", "Pink", "Red", "Orange", "Yellow", "Teal"]
    for i, (f, color) in enumerate(zip(key_frames, sphere_colors)):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        sphere.translate(ee_transformed[f])
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        spheres.append(sphere)
        print(f"  {color_names[i]} (Frame {f}): [{ee_transformed[f, 0]:.3f}, {ee_transformed[f, 1]:.3f}, {ee_transformed[f, 2]:.3f}]")

    # Coordinate axes at origin
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # Visualize
    print("\nVisualization controls:")
    print("  Left mouse: Rotate")
    print("  Right mouse: Pan")
    print("  Scroll: Zoom")
    print("  Q: Quit")
    print(f"\nAdjust --tx, --ty, --tz to align trajectory with gripper in point cloud")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"EE Trajectory + Transform - Episode {args.episode_index}", width=1280, height=720)
    vis.add_geometry(pcd)
    vis.add_geometry(trajectory)
    vis.add_geometry(axes)
    for s in spheres:
        vis.add_geometry(s)

    opt = vis.get_render_option()
    opt.point_size = 4.0
    opt.background_color = np.array([1, 1, 1])

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
