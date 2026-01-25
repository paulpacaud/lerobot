#!/usr/bin/env python
"""
Visualize the robot FK alongside the RGB image from the dataset.

Shows:
- Left: RGB image from the dataset at the specified frame
- Right: 3D robot visualization with FK-computed EE position

This helps verify that the FK matches what's in the actual image.

Usage:
```bash
python examples/post_process_dataset/visualize_robot_fk_with_rgb.py \
    --urdf_path=./examples/post_process_dataset/constants/SO101/so101_new_calib.urdf \
    --dataset_dir=/home/ppacaud/lerobot_datasets/depth_test_v2 \
    --episode_index=0 \
    --frame_index=100
```
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tap import Tap


class Args(Tap):
    urdf_path: str  # Path to the robot URDF file
    dataset_dir: str  # Path to dataset (joint-space)
    episode_index: int = 0
    frame_index: int = 0
    video_key: str = "observation.images.front"  # Video key in dataset


def load_joints_from_dataset(dataset_dir: str, episode_index: int, frame_index: int) -> np.ndarray:
    """Load joint values from dataset."""
    dataset_path = Path(dataset_dir)
    parquet_path = dataset_path / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    if not parquet_path.exists():
        parquet_path = dataset_path / "data" / "chunk-000" / "file-000.parquet"

    df = pd.read_parquet(parquet_path)
    if "episode_index" in df.columns:
        df = df[df["episode_index"] == episode_index]

    state = np.array(df.iloc[frame_index]["observation.state"], dtype=np.float64)
    return state[:5]  # First 5 are arm joints


def extract_frame_from_video(video_path: Path, frame_index: int) -> np.ndarray:
    """Extract a specific frame from a video file."""
    try:
        # Try torchvision first (handles AV1 well)
        import torchvision

        video_reader = torchvision.io.VideoReader(str(video_path), "video")
        # Seek to approximate position
        for i, frame_data in enumerate(video_reader):
            if i == frame_index:
                frame = frame_data["data"].numpy()
                # torchvision returns (C, H, W), convert to (H, W, C)
                return np.transpose(frame, (1, 2, 0))
        raise ValueError(f"Frame {frame_index} not found in video")
    except Exception as e:
        print(f"torchvision failed: {e}, trying imageio...")

        # Fallback to imageio
        import imageio.v3 as iio

        frames = iio.imread(str(video_path), plugin="pyav")
        if frame_index >= len(frames):
            raise ValueError(f"Frame {frame_index} out of range (max {len(frames) - 1})")
        return frames[frame_index]


def render_robot_to_image(urdf_path: str, joints_deg: np.ndarray) -> np.ndarray:
    """Render the robot to an image using Open3D offscreen rendering."""
    import open3d as o3d
    import yourdfpy

    from lerobot.model.kinematics import RobotKinematics

    # Load URDF
    urdf = yourdfpy.URDF.load(urdf_path)

    # Set joint values
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    joints_rad = np.deg2rad(joints_deg)
    cfg = {name: val for name, val in zip(joint_names, joints_rad)}
    urdf.update_cfg(cfg)

    # Get robot mesh
    robot_mesh = urdf.scene.to_geometry()

    # Convert to Open3D
    vertices = np.asarray(robot_mesh.vertices)
    faces = np.asarray(robot_mesh.faces)

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    if robot_mesh.visual.vertex_colors is not None:
        colors = np.asarray(robot_mesh.visual.vertex_colors)[:, :3] / 255.0
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    else:
        o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])

    o3d_mesh.compute_vertex_normals()

    # Compute FK for EE position
    kin = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name="gripper_frame_link",
        joint_names=joint_names,
    )
    T = kin.forward_kinematics(joints_deg)
    ee_pos = T[:3, 3]

    # Create EE sphere
    ee_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
    ee_sphere.translate(ee_pos)
    ee_sphere.paint_uniform_color([0.0, 0.0, 1.0])
    ee_sphere.compute_vertex_normals()

    # Create axes
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # Render to image using offscreen renderer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=640, height=480)
    vis.add_geometry(o3d_mesh)
    vis.add_geometry(ee_sphere)
    vis.add_geometry(axes)

    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])
    opt.mesh_show_back_face = True

    ctr = vis.get_view_control()
    ctr.set_zoom(1)
    ctr.set_front([1, 0, 0.4])
    ctr.set_lookat([0, 0, 0.1])  # Look at origin (slightly elevated)
    ctr.set_up([0, 0, 1])

    vis.poll_events()
    vis.update_renderer()

    # Capture image
    img = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    return (np.asarray(img) * 255).astype(np.uint8)


def main():
    args = Args().parse_args()

    dataset_path = Path(args.dataset_dir)
    urdf_path = str(Path(args.urdf_path).resolve())

    # Load joints
    joints_deg = load_joints_from_dataset(args.dataset_dir, args.episode_index, args.frame_index)
    print(f"Loaded joints (deg): {joints_deg}")

    # Load RGB frame from video
    video_path = (
        dataset_path
        / "videos"
        / "chunk-000"
        / args.video_key
        / f"episode_{args.episode_index:06d}.mp4"
    )
    print(f"Loading RGB frame from: {video_path}")
    rgb_frame = extract_frame_from_video(video_path, args.frame_index)

    # Render robot
    print("Rendering robot...")
    robot_img = render_robot_to_image(urdf_path, joints_deg)

    # Compute FK for display
    from lerobot.model.kinematics import RobotKinematics

    kin = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name="gripper_frame_link",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
    )
    T = kin.forward_kinematics(joints_deg)
    ee_pos = T[:3, 3]

    # Display side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(rgb_frame)
    axes[0].set_title(f"RGB Image (Episode {args.episode_index}, Frame {args.frame_index})")
    axes[0].axis("off")

    axes[1].imshow(robot_img)
    axes[1].set_title(
        f"Robot FK Visualization\n"
        f"Joints: [{joints_deg[0]:.1f}, {joints_deg[1]:.1f}, {joints_deg[2]:.1f}, {joints_deg[3]:.1f}, {joints_deg[4]:.1f}] deg\n"
        f"EE: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}] m"
    )
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(f"fk_comparison_ep{args.episode_index}_frame{args.frame_index}.png", dpi=150)
    print(f"Saved: fk_comparison_ep{args.episode_index}_frame{args.frame_index}.png")
    plt.show()


if __name__ == "__main__":
    main()
