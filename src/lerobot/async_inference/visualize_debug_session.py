# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Visualize a PointAct debug session.

Shows the point cloud, EE trajectory, and action predictions from a saved debug session.

Usage:
```bash
python -m lerobot.async_inference.visualize_debug_session --session_dir=./debug_logs/session_20260128_210000 --frame=0

# Interactive mode: click on points to see their coordinates
python -m lerobot.async_inference.visualize_debug_session --session_dir=./debug_logs/session_20260128_210000 --frame=0 --interactive

# Show all frames with their state info
python -m lerobot.async_inference.visualize_debug_session --session_dir=./debug_logs/session_20260128_210000 --list_frames
```
"""

import json
from pathlib import Path

import cv2
import msgpack
import msgpack_numpy
import numpy as np
import open3d as o3d
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
ACTION_COLOR = [0, 0, 1]  # Blue for predicted actions

# Color scheme for key frame markers
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
    """Arguments for visualizing PointAct debug session."""

    session_dir: str  # Path to the debug session directory
    frame: int = 0  # Which frame to visualize (uses its point cloud)
    list_frames: bool = False  # List all available frames and exit
    interactive: bool = False  # Enable interactive mode for point picking
    show_action: bool = True  # Show predicted action trajectory
    trajectory_subsample: int = 1  # Subsample trajectory for performance
    save_image: str = ""  # Save RGB image to this path instead of displaying


def load_metadata(session_dir: Path) -> dict:
    """Load session metadata from metadata.json."""
    metadata_path = session_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with open(metadata_path) as f:
        return json.load(f)


def list_debug_files(session_dir: Path) -> tuple[list[Path], list[Path]]:
    """List all input and output debug files in the session directory.

    Returns:
        Tuple of (input_files, output_files) sorted by step number
    """
    input_files = sorted(session_dir.glob("step_*_input.msgpack"))
    output_files = sorted(session_dir.glob("step_*_output.msgpack"))
    return input_files, output_files


def load_debug_file(filepath: Path) -> dict:
    """Load a msgpack debug file."""
    with open(filepath, "rb") as f:
        return msgpack.unpackb(f.read(), raw=False)


def load_all_frames(session_dir: Path) -> list[dict]:
    """Load all input/output pairs from a debug session.

    Returns:
        List of dicts with 'input' and 'output' keys for each frame
    """
    input_files, output_files = list_debug_files(session_dir)

    frames = []
    for inp_file in input_files:
        # Extract step number from filename
        step_str = inp_file.stem.split("_")[1]  # "step_00000_input" -> "00000"
        step_num = int(step_str)

        # Find matching output file
        out_file = session_dir / f"step_{step_str}_output.msgpack"

        input_data = load_debug_file(inp_file)
        output_data = load_debug_file(out_file) if out_file.exists() else None

        frames.append({
            "step": step_num,
            "input": input_data,
            "output": output_data,
        })

    return frames


def print_frame_list(frames: list[dict], metadata: dict) -> None:
    """Print a summary of all frames in the session."""
    print(f"\nDebug Session: {metadata.get('session_start', 'Unknown')}")
    print(f"Task: {metadata.get('task', 'Unknown')}")
    print(f"Server: {metadata.get('server_address', 'Unknown')}")
    print(f"FPS: {metadata.get('fps', 'Unknown')}")
    print(f"\nTotal frames: {len(frames)}")
    print("-" * 80)
    print(f"{'Step':>6} | {'State (joint positions)':^50} | {'Req Time (ms)':>12}")
    print("-" * 80)

    for frame in frames:
        step = frame["step"]
        inp = frame["input"]
        out = frame["output"]

        state = inp.get("observation.state")
        if state is not None:
            state_str = ", ".join(f"{v:.2f}" for v in state[:6])  # First 6 values
        else:
            joint_state = inp.get("observation.states.joint_state")
            if joint_state is not None:
                state_str = ", ".join(f"{v:.2f}" for v in joint_state)
            else:
                state_str = "N/A"

        req_time = out.get("request_time_ms", 0) if out else 0
        print(f"{step:>6} | [{state_str:^48}] | {req_time:>10.2f}")

    print("-" * 80)


def print_frame_info(frame: dict, metadata: dict) -> None:
    """Print detailed information about a specific frame."""
    inp = frame["input"]
    out = frame["output"]

    print(f"\n{'=' * 60}")
    print(f"Frame {frame['step']}")
    print(f"{'=' * 60}")
    print(f"Timestamp: {inp.get('timestamp', 'N/A')}")
    print(f"Task: {inp.get('task', metadata.get('task', 'N/A'))}")

    # State info
    state = inp.get("observation.state")
    if state is not None:
        print(f"\nState (EE): shape={np.array(state).shape}")
        print(f"  Values: [{', '.join(f'{v:.4f}' for v in state)}]")

    joint_state = inp.get("observation.states.joint_state")
    if joint_state is not None:
        joint_arr = np.array(joint_state)
        print(f"\nJoint State: shape={joint_arr.shape}")
        print(f"  Values: [{', '.join(f'{v:.4f}' for v in joint_arr)}]")

    ee_state = inp.get("observation.states.ee_state")
    if ee_state is not None:
        ee_arr = np.array(ee_state)
        print(f"\nEE State: shape={ee_arr.shape}")
        print(f"  Position: [{', '.join(f'{v:.4f}' for v in ee_arr[:3])}]")
        print(f"  Rotation: [{', '.join(f'{v:.4f}' for v in ee_arr[3:6])}]")

    # Image info
    image = inp.get("observation.images.front_image")
    if image is not None:
        img_arr = np.array(image)
        print(f"\nImage: shape={img_arr.shape}, dtype={img_arr.dtype}")
        print(f"  Range: [{img_arr.min()}, {img_arr.max()}]")

    # Point cloud info
    points = inp.get("observation.points")
    if points is not None:
        pts_arr = np.array(points)
        print(f"\nPoint Cloud: shape={pts_arr.shape}")
        if len(pts_arr) > 0:
            print(f"  X range: [{pts_arr[:, 0].min():.4f}, {pts_arr[:, 0].max():.4f}]")
            print(f"  Y range: [{pts_arr[:, 1].min():.4f}, {pts_arr[:, 1].max():.4f}]")
            print(f"  Z range: [{pts_arr[:, 2].min():.4f}, {pts_arr[:, 2].max():.4f}]")

    # Action info
    if out:
        print(f"\n--- Server Response ---")
        print(f"Request time: {out.get('request_time_ms', 'N/A'):.2f} ms")
        action = out.get("action")
        if action is not None:
            action_arr = np.array(action)
            # Handle nested batch dimensions
            while action_arr.ndim > 2:
                action_arr = action_arr[0]
            print(f"Action chunk: shape={action_arr.shape}")
            if action_arr.ndim == 2 and len(action_arr) > 0:
                print(f"  First action: [{', '.join(f'{v:.4f}' for v in action_arr[0])}]")
                print(f"  Last action:  [{', '.join(f'{v:.4f}' for v in action_arr[-1])}]")
    print(f"{'=' * 60}")


def create_point_cloud_geometry(points: np.ndarray) -> o3d.geometry.PointCloud:
    """Create Open3D point cloud geometry from numpy array."""
    pcd = o3d.geometry.PointCloud()
    if len(points) > 0:
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        if points.shape[1] >= 6:
            pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6])
    return pcd


def create_trajectory_geometry(ee_positions: np.ndarray, subsample: int, color: list) -> o3d.geometry.LineSet:
    """Create trajectory line set from EE positions."""
    trajectory_points = ee_positions[::subsample]
    if len(trajectory_points) < 2:
        trajectory_points = ee_positions

    lines = [[i, i + 1] for i in range(len(trajectory_points) - 1)]
    colors = [color for _ in lines]

    trajectory = o3d.geometry.LineSet()
    trajectory.points = o3d.utility.Vector3dVector(trajectory_points)
    trajectory.lines = o3d.utility.Vector2iVector(lines)
    trajectory.colors = o3d.utility.Vector3dVector(colors)
    return trajectory


def compute_key_frame_indices(n_frames: int) -> list[int]:
    """Compute indices for key frames evenly distributed across frames."""
    if n_frames <= NUM_KEY_FRAMES:
        return list(range(n_frames))
    key_frames = [i * n_frames // (NUM_KEY_FRAMES - 1) for i in range(NUM_KEY_FRAMES)]
    key_frames[-1] = min(key_frames[-1], n_frames - 1)
    return key_frames


def create_key_frame_spheres(ee_positions: np.ndarray, key_frames: list[int]) -> list[o3d.geometry.TriangleMesh]:
    """Create colored sphere markers at key frame positions."""
    spheres = []
    for i, frame_idx in enumerate(key_frames):
        if frame_idx >= len(ee_positions):
            continue
        color_idx = i % len(KEY_FRAME_COLORS)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=SPHERE_RADIUS)
        sphere.translate(ee_positions[frame_idx])
        sphere.paint_uniform_color(KEY_FRAME_COLORS[color_idx])
        sphere.compute_vertex_normals()
        spheres.append(sphere)
    return spheres


def extract_ee_positions_from_frames(frames: list[dict]) -> np.ndarray:
    """Extract EE positions from all frames.

    Tries to use observation.states.ee_state first, falls back to observation.state.
    """
    positions = []
    for frame in frames:
        inp = frame["input"]
        ee_state = inp.get("observation.states.ee_state")
        if ee_state is not None:
            positions.append(np.array(ee_state)[:3])
        else:
            state = inp.get("observation.state")
            if state is not None:
                positions.append(np.array(state)[:3])

    if not positions:
        return np.zeros((0, 3))
    return np.array(positions)


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


def run_visualizer(geometries: list, title: str) -> None:
    """Set up and run the Open3D visualizer."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

    for geom in geometries:
        vis.add_geometry(geom)

    opt = vis.get_render_option()
    opt.point_size = POINT_SIZE
    opt.background_color = BACKGROUND_COLOR

    vis.run()
    vis.destroy_window()


def run_interactive_visualizer(pcd: o3d.geometry.PointCloud, geometries: list, title: str) -> None:
    """Set up and run the Open3D interactive visualizer with point picking."""
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(window_name=title, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

    vis.add_geometry(pcd)
    for geom in geometries:
        vis.add_geometry(geom)

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
            coord = points[idx]
            print(f"\nPoint {i + 1} (index {idx}):")
            print(f"  Coordinates: X={coord[0]:.4f}, Y={coord[1]:.4f}, Z={coord[2]:.4f}")
            if colors is not None and idx < len(colors):
                c = colors[idx]
                print(f"  Color (RGB): [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}]")
        print("=" * 50)
    else:
        print("\nNo points were selected.")

    vis.destroy_window()


def save_image(image: np.ndarray, filepath: str) -> None:
    """Save an image to disk.

    Args:
        image: RGB image as numpy array
        filepath: Path to save the image
    """
    if image is None:
        return
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, img_bgr)
    print(f"Image saved to: {filepath}")


def main():
    args = Args().parse_args()
    session_dir = Path(args.session_dir)

    if not session_dir.exists():
        print(f"Error: Session directory not found: {session_dir}")
        return

    # Load metadata
    try:
        metadata = load_metadata(session_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Load all frames
    frames = load_all_frames(session_dir)
    if not frames:
        print(f"Error: No debug frames found in {session_dir}")
        return

    # List frames mode
    if args.list_frames:
        print_frame_list(frames, metadata)
        return

    # Check frame index
    if args.frame >= len(frames):
        print(f"Error: Frame {args.frame} not found. Available frames: 0-{len(frames) - 1}")
        return

    # Get selected frame
    selected_frame = frames[args.frame]
    print_frame_info(selected_frame, metadata)

    # Extract point cloud from selected frame
    points = selected_frame["input"].get("observation.points")
    if points is not None:
        points = np.array(points)
    else:
        points = np.zeros((0, 6))

    print(f"\nPoint cloud (frame {args.frame}): {len(points)} points")

    # Create point cloud geometry
    pcd = create_point_cloud_geometry(points)

    # Extract EE positions from all frames for trajectory
    ee_positions = extract_ee_positions_from_frames(frames)
    print(f"Trajectory: {len(ee_positions)} positions")

    # Create geometries list
    geometries = []

    # Add trajectory if we have EE positions
    if len(ee_positions) > 1:
        trajectory = create_trajectory_geometry(ee_positions, args.trajectory_subsample, TRAJECTORY_COLOR)
        geometries.append(trajectory)

        # Add key frame spheres
        key_frames = compute_key_frame_indices(len(ee_positions))
        spheres = create_key_frame_spheres(ee_positions, key_frames)
        geometries.extend(spheres)

        # Print key frame info
        print("\nKey frames (colored spheres):")
        for i, frame_idx in enumerate(key_frames):
            if frame_idx < len(ee_positions):
                pos = ee_positions[frame_idx]
                color_name = KEY_FRAME_COLOR_NAMES[i % len(KEY_FRAME_COLOR_NAMES)]
                print(f"  {color_name:8s} (Frame {frame_idx:4d}): [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    # Add action trajectory if available and requested
    if args.show_action and selected_frame["output"]:
        action = selected_frame["output"].get("action")
        if action is not None:
            action_arr = np.array(action)
            # Handle batch dimensions
            while action_arr.ndim > 2:
                action_arr = action_arr[0]
            # Extract positions (first 3 values of each action)
            if action_arr.ndim == 2 and action_arr.shape[1] >= 3:
                action_positions = action_arr[:, :3]
                action_traj = create_trajectory_geometry(action_positions, 1, ACTION_COLOR)
                geometries.append(action_traj)
                print(f"\nAction prediction: {len(action_positions)} steps (blue trajectory)")

    # Add coordinate axes
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=AXES_SIZE)
    geometries.append(axes)

    # Save image to disk (default: in session directory)
    image = selected_frame["input"].get("observation.images.front_image")
    if image is not None:
        if args.save_image:
            image_path = args.save_image
        else:
            image_path = str(session_dir / f"frame_{args.frame:05d}_rgb.png")
        save_image(np.array(image), image_path)

    # Run visualization
    title = f"Debug Session - Frame {args.frame}"
    print_visualization_controls(interactive=args.interactive)

    if args.interactive:
        run_interactive_visualizer(pcd, geometries, title)
    else:
        all_geometries = [pcd] + geometries
        run_visualizer(all_geometries, title)


if __name__ == "__main__":
    main()
