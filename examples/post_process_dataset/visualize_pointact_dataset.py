"""
Visualize a PointAct format dataset.

Shows the EE trajectory overlaid on the point cloud.

Usage:
```bash
python examples/post_process_dataset/visualize_pointact_dataset.py --dataset_dir=$HOME/lerobot_datasets/depth_test_pointact --episode_index=0 --pcd_frame=0
```
"""

import json
from pathlib import Path

import av
import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import open3d as o3d
import pandas as pd
from tap import Tap

msgpack_numpy.patch()

# Visualization constants
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
POINT_SIZE = 4.0
BACKGROUND_COLOR = np.array([1, 1, 1])
SPHERE_RADIUS = 0.015
AXES_SIZE = 0.1
TRAJECTORY_COLOR = [1, 0, 0]  # Red
KEY_FRAME_COLORS = [
    [0, 1, 0],      # Green (start)
    [0, 0.5, 1],    # Cyan
    [0, 0, 1],      # Blue
    [0.5, 0, 1],    # Purple
    [1, 0, 1],      # Magenta
    [1, 0, 0.5],    # Pink
    [1, 0, 0],      # Red
    [1, 0.5, 0],    # Orange
    [1, 1, 0],      # Yellow
    [0, 1, 1],      # Teal (end)
]
KEY_FRAME_COLOR_NAMES = [
    "Green", "Cyan", "Blue", "Purple", "Magenta",
    "Pink", "Red", "Orange", "Yellow", "Teal",
]
NUM_KEY_FRAMES = 10


class Args(Tap):
    """Arguments for visualizing PointAct dataset."""

    dataset_dir: str  # Path to the PointAct dataset
    episode_index: int = 0  # Episode to visualize
    pcd_frame: int = 0  # Which frame's point cloud to show as background

    # Data keys (PointAct format)
    state_key: str = "observation.state"  # Key for state with EE position
    ee_state_key: str = "observation.states.ee_state"  # Key for EE-only state
    joint_state_key: str = "observation.states.joint_state"  # Key for joint state
    gripper_state_key: str = "observation.states.gripper_state"  # Key for gripper state
    point_cloud_key: str = "observation.points.frontview"  # Key for point cloud
    image_key: str = "observation.images.front_image"  # Key for images

    # Visualization options
    show_joint_info: bool = False  # Print joint state info for each key frame
    trajectory_subsample: int = 3  # Subsample trajectory for performance
    interactive: bool = False  # Enable interactive mode to click and see point coordinates


def load_info(dataset_path: Path) -> dict:
    """Load info.json from dataset root."""
    with open(dataset_path / "meta" / "info.json") as f:
        return json.load(f)


def decode_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    """Decode a single frame from a video file."""
    with av.open(str(video_path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i == frame_index:
                return frame.to_ndarray(format="rgb24")
    return None


def load_episode_data(dataset_path: Path, episode_index: int) -> pd.DataFrame:
    """Load parquet data for a specific episode."""
    parquet_path = dataset_path / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    if not parquet_path.exists():
        parquet_path = dataset_path / "data" / "chunk-000" / "file-000.parquet"

    df = pd.read_parquet(parquet_path)
    if "episode_index" in df.columns:
        df = df[df["episode_index"] == episode_index]
    return df


def extract_states(df: pd.DataFrame, args: Args) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Extract state arrays from the dataframe.

    Returns:
        Tuple of (states, ee_positions, ee_states, joint_states, gripper_states)
    """
    states = np.stack(df[args.state_key].values)
    ee_positions = states[:, :3]  # x, y, z

    ee_states = np.stack(df[args.ee_state_key].values) if args.ee_state_key in df.columns else None
    joint_states = np.stack(df[args.joint_state_key].values) if args.joint_state_key in df.columns else None
    gripper_states = np.stack(df[args.gripper_state_key].values) if args.gripper_state_key in df.columns else None

    return states, ee_positions, ee_states, joint_states, gripper_states


def print_state_info(states: np.ndarray, ee_positions: np.ndarray, ee_states: np.ndarray | None, joint_states: np.ndarray | None, gripper_states: np.ndarray | None) -> None:
    """Print state shape and range information."""
    print(f"\nState shape: {states.shape}")
    print(f"EE position range:")
    print(f"  X: [{ee_positions[:, 0].min():.4f}, {ee_positions[:, 0].max():.4f}]")
    print(f"  Y: [{ee_positions[:, 1].min():.4f}, {ee_positions[:, 1].max():.4f}]")
    print(f"  Z: [{ee_positions[:, 2].min():.4f}, {ee_positions[:, 2].max():.4f}]")

    if ee_states is not None:
        print(f"\nEE state shape: {ee_states.shape}")
    if joint_states is not None:
        print(f"Joint state shape: {joint_states.shape}")
    if gripper_states is not None:
        print(f"Gripper state shape: {gripper_states.shape}")


def load_point_cloud(dataset_path: Path, info: dict, point_cloud_key: str, episode_index: int, frame_index: int) -> np.ndarray:
    """Load point cloud from LMDB storage."""
    pcd_feature = info["features"].get(point_cloud_key, {})
    pcd_info = pcd_feature.get("info", {})
    lmdb_subpath = pcd_info.get("path", "point_clouds")

    lmdb_path = dataset_path / lmdb_subpath
    if not lmdb_path.exists():
        lmdb_path = dataset_path / "point_clouds"

    env = lmdb.open(str(lmdb_path), readonly=True)
    with env.begin() as txn:
        key = f"{episode_index}-{frame_index}"
        pc_data = txn.get(key.encode("ascii"))
        if pc_data is None:
            print(f"Warning: Point cloud not found for key '{key}'")
            pc = np.zeros((0, 6))
        else:
            pc = np.array(msgpack.unpackb(pc_data))
    env.close()
    return pc


def print_point_cloud_info(pc: np.ndarray, frame_index: int) -> None:
    """Print point cloud statistics."""
    print(f"\nPoint cloud (frame {frame_index}): {len(pc)} points")
    if len(pc) > 0:
        print(f"Point cloud range:")
        print(f"  X: [{pc[:, 0].min():.4f}, {pc[:, 0].max():.4f}]")
        print(f"  Y: [{pc[:, 1].min():.4f}, {pc[:, 1].max():.4f}]")
        print(f"  Z: [{pc[:, 2].min():.4f}, {pc[:, 2].max():.4f}]")


def create_point_cloud_geometry(pc: np.ndarray) -> o3d.geometry.PointCloud:
    """Create Open3D point cloud geometry from numpy array."""
    pcd = o3d.geometry.PointCloud()
    if len(pc) > 0:
        pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:6])
    return pcd


def create_trajectory_geometry(ee_positions: np.ndarray, subsample: int) -> o3d.geometry.LineSet:
    """Create trajectory line set from EE positions."""
    trajectory_points = ee_positions[::subsample]
    lines = [[i, i + 1] for i in range(len(trajectory_points) - 1)]
    colors = [TRAJECTORY_COLOR for _ in lines]

    trajectory = o3d.geometry.LineSet()
    trajectory.points = o3d.utility.Vector3dVector(trajectory_points)
    trajectory.lines = o3d.utility.Vector2iVector(lines)
    trajectory.colors = o3d.utility.Vector3dVector(colors)
    return trajectory


def compute_key_frame_indices(n_frames: int) -> list[int]:
    """Compute indices for key frames evenly distributed across the episode."""
    key_frames = [i * n_frames // (NUM_KEY_FRAMES - 1) for i in range(NUM_KEY_FRAMES)]
    key_frames[-1] = min(key_frames[-1], n_frames - 1)
    return key_frames


def create_key_frame_spheres(ee_positions: np.ndarray, key_frames: list[int]) -> list[o3d.geometry.TriangleMesh]:
    """Create colored sphere markers at key frame positions."""
    spheres = []
    for frame_idx, color in zip(key_frames, KEY_FRAME_COLORS):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=SPHERE_RADIUS)
        sphere.translate(ee_positions[frame_idx])
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        spheres.append(sphere)
    return spheres


def print_key_frame_info(ee_positions: np.ndarray, states: np.ndarray, joint_states: np.ndarray | None, key_frames: list[int], show_joint_info: bool) -> None:
    """Print information about key frames."""
    print("\nKey frames (colored spheres):")
    for i, frame_idx in enumerate(key_frames):
        pos = ee_positions[frame_idx]
        pos_str = f"[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
        color_name = KEY_FRAME_COLOR_NAMES[i]

        if show_joint_info and joint_states is not None:
            joints = joint_states[frame_idx]
            joint_str = ", ".join([f"{j:.2f}" for j in joints])
            print(f"  {color_name:8s} (Frame {frame_idx:4d}): pos={pos_str}, joints=[{joint_str}]")
        else:
            gripper_val = states[frame_idx, -1]
            print(f"  {color_name:8s} (Frame {frame_idx:4d}): pos={pos_str}, gripper={gripper_val:.2f}")


def print_visualization_controls(interactive: bool = False) -> None:
    """Print visualization control instructions."""
    print("\nVisualization controls:")
    print("  Left mouse: Rotate")
    print("  Right mouse: Pan")
    print("  Scroll: Zoom")
    if interactive:
        print("  Shift + Left click: Select point (coordinates printed to console)")
        print("  K: Keep current selection")
        print("  C: Clear selection")
    print("  Q: Quit")


def run_visualizer(pcd: o3d.geometry.PointCloud, trajectory: o3d.geometry.LineSet, spheres: list[o3d.geometry.TriangleMesh], episode_index: int) -> None:
    """Set up and run the Open3D visualizer."""
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=AXES_SIZE)

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"PointAct Dataset - Episode {episode_index}",
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT
    )
    vis.add_geometry(pcd)
    vis.add_geometry(trajectory)
    vis.add_geometry(axes)
    for s in spheres:
        vis.add_geometry(s)

    opt = vis.get_render_option()
    opt.point_size = POINT_SIZE
    opt.background_color = BACKGROUND_COLOR

    vis.run()
    vis.destroy_window()


def run_interactive_visualizer(pcd: o3d.geometry.PointCloud, trajectory: o3d.geometry.LineSet, spheres: list[o3d.geometry.TriangleMesh], episode_index: int) -> None:
    """Set up and run the Open3D interactive visualizer with point picking."""
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=AXES_SIZE)

    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(
        window_name=f"PointAct Dataset - Episode {episode_index} (Interactive)",
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT
    )
    vis.add_geometry(pcd)
    vis.add_geometry(trajectory)
    vis.add_geometry(axes)
    for s in spheres:
        vis.add_geometry(s)

    opt = vis.get_render_option()
    opt.point_size = POINT_SIZE
    opt.background_color = BACKGROUND_COLOR

    print("\nInteractive mode: Shift+Click on points to select them.")
    print("Press 'K' to keep selection, then close window to see coordinates.\n")

    vis.run()

    # Get selected points after visualization closes
    picked_points = vis.get_picked_points()
    if picked_points:
        print("\n" + "=" * 50)
        print("SELECTED POINTS:")
        print("=" * 50)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        for i, picked in enumerate(picked_points):
            idx = picked.index
            coord = points[idx]  # Use original point cloud coordinates for full precision
            print(f"\nPoint {i + 1} (index {idx}):")
            print(f"  Coordinates: X={coord[0]:.3f}, Y={coord[1]:.3f}, Z={coord[2]:.3f}")
            if colors is not None and idx < len(colors):
                c = colors[idx]
                print(f"  Color (RGB): [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}]")
        print("=" * 50)
    else:
        print("\nNo points were selected.")

    vis.destroy_window()


def main():
    args = Args().parse_args()
    dataset_path = Path(args.dataset_dir)

    # Load dataset info
    info = load_info(dataset_path)
    print(f"Dataset: {dataset_path}")
    print(f"Features: {list(info['features'].keys())}")

    # Load episode data
    df = load_episode_data(dataset_path, args.episode_index)
    print(f"\nEpisode {args.episode_index}: {len(df)} frames")

    # Extract states
    states, ee_positions, ee_states, joint_states, gripper_states = extract_states(df, args)
    print_state_info(states, ee_positions, ee_states, joint_states, gripper_states)

    # Load and display point cloud
    pc = load_point_cloud(dataset_path, info, args.point_cloud_key, args.episode_index, args.pcd_frame)
    print_point_cloud_info(pc, args.pcd_frame)

    # Create geometries
    pcd = create_point_cloud_geometry(pc)
    trajectory = create_trajectory_geometry(ee_positions, args.trajectory_subsample)

    # Create key frame markers
    key_frames = compute_key_frame_indices(len(ee_positions))
    spheres = create_key_frame_spheres(ee_positions, key_frames)
    print_key_frame_info(ee_positions, states, joint_states, key_frames, args.show_joint_info)

    # Run visualization
    print_visualization_controls(interactive=args.interactive)
    if args.interactive:
        run_interactive_visualizer(pcd, trajectory, spheres, args.episode_index)
    else:
        run_visualizer(pcd, trajectory, spheres, args.episode_index)


if __name__ == "__main__":
    main()
