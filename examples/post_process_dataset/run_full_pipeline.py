#!/usr/bin/env python3
"""
Unified pipeline to convert a LeRobot v3 dataset to PointAct format.

Pipeline steps:
1. Convert v3 → v2.1 format
2. Add point clouds to dataset
3. Convert to PointAct format (joint-space → EE-space, resize images, trim idle frames)
"""

import logging
import shutil
import tempfile
from pathlib import Path

from tap import Tap

# Import main functions from each pipeline step
from examples.post_process_dataset.convert_lerobot_dataset_v3_to_v2 import convert_dataset
from examples.post_process_dataset.add_point_cloud_to_dataset import add_point_clouds_to_dataset
from examples.post_process_dataset.convert_to_pointact_format import convert_to_pointact_format

# Import constants
from examples.post_process_dataset.constants.constants import (
    INTRINSICS_FILE,
    EXTRINSICS_FILE,
    WORKSPACE,
    CONSTANTS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Args(Tap):
    """Arguments for the full dataset conversion pipeline."""

    # Required arguments
    input_dir: str  # Path to input LeRobot v3 dataset
    output_dir: str  # Path for output PointAct dataset

    # URDF configuration
    urdf_path: str = str(CONSTANTS_DIR / "SO101" / "so101_new_calib.urdf")  # Path to robot URDF

    # Calibration files
    intrinsics_file: str = str(INTRINSICS_FILE)  # Path to camera intrinsics npz
    extrinsics_file: str = str(EXTRINSICS_FILE)  # Path to camera extrinsics npz

    # Point cloud processing
    voxel_size: float = 0.01  # Voxel size for downsampling (meters)
    depth_scale: float = 1000.0  # Scale factor for depth values (1000 for mm->m)
    num_workers: int = 8  # Number of parallel workers for point cloud processing

    # Workspace bounds (meters)
    workspace_x_min: float = WORKSPACE["X_BBOX"][0]
    workspace_x_max: float = WORKSPACE["X_BBOX"][1]
    workspace_y_min: float = WORKSPACE["Y_BBOX"][0]
    workspace_y_max: float = WORKSPACE["Y_BBOX"][1]
    workspace_z_min: float = WORKSPACE["Z_BBOX"][0]
    workspace_z_max: float = WORKSPACE["Z_BBOX"][1]

    # Image resize
    image_size: int = 256  # Target image size (square)

    # Idle frame trimming
    no_trim_idle_frames: bool = False  # Disable trimming of idle frames
    trim_threshold_factor: float = 0.05  # Threshold for idle detection
    min_idle_segment: int = 5  # Minimum consecutive idle frames
    keep_frames_per_idle: int = 1  # Frames to keep from each idle segment

    # FK configuration
    target_frame: str = "gripper_frame_link"  # EE frame in URDF
    joint_names: list[str] = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ]

    # Data keys
    rgb_key: str = "observation.images.front"
    depth_key: str = "observation.images.front_depth"

    # Keep intermediate v2 directory
    keep_intermediate: bool = False  # Keep the intermediate v2 directory


def run_pipeline(args: Args) -> None:
    """Run the full dataset conversion pipeline."""
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    # Build workspace dict
    workspace = {
        "X_BBOX": [args.workspace_x_min, args.workspace_x_max],
        "Y_BBOX": [args.workspace_y_min, args.workspace_y_max],
        "Z_BBOX": [args.workspace_z_min, args.workspace_z_max],
    }

    # Create intermediate v2 directory
    if args.keep_intermediate:
        intermediate_dir = output_path.parent / f"{output_path.name}_v2_intermediate"
        intermediate_dir_str = str(intermediate_dir)
    else:
        temp_dir = tempfile.mkdtemp(prefix="lerobot_v2_")
        intermediate_dir_str = temp_dir

    try:
        # Step 1: Convert v3 → v2.1
        logging.info("=" * 60)
        logging.info("STEP 1/3: Converting v3 to v2.1 format")
        logging.info("=" * 60)
        convert_dataset(
            input_dir=str(input_path),
            output_dir=intermediate_dir_str,
        )

        # Step 2: Add point clouds
        logging.info("=" * 60)
        logging.info("STEP 2/3: Adding point clouds to dataset")
        logging.info("=" * 60)
        add_point_clouds_to_dataset(
            dataset_dir=intermediate_dir_str,
            intrinsics_file=args.intrinsics_file,
            extrinsics_file=args.extrinsics_file,
            rgb_key=args.rgb_key,
            depth_key=args.depth_key,
            output_key="observation.point_cloud",
            voxel_size=args.voxel_size,
            workspace=workspace,
            depth_scale=args.depth_scale,
            num_workers=args.num_workers,
        )

        # Step 3: Convert to PointAct format
        logging.info("=" * 60)
        logging.info("STEP 3/3: Converting to PointAct format")
        logging.info("=" * 60)
        convert_to_pointact_format(
            dataset_dir=intermediate_dir_str,
            urdf_path=args.urdf_path,
            output_dir=str(output_path),
            target_frame=args.target_frame,
            joint_names=args.joint_names,
            image_size=args.image_size,
            trim_idle_frames=not args.no_trim_idle_frames,
            trim_threshold_factor=args.trim_threshold_factor,
            min_idle_segment=args.min_idle_segment,
            keep_frames_per_idle=args.keep_frames_per_idle,
            state_key="observation.state",
            action_key="action",
            rgb_key=args.rgb_key,
            depth_key=args.depth_key,
            point_cloud_key="observation.point_cloud",
        )

        logging.info("=" * 60)
        logging.info("PIPELINE COMPLETE")
        logging.info(f"Output dataset: {output_path}")
        logging.info("=" * 60)

    finally:
        # Clean up intermediate directory if not keeping it
        if not args.keep_intermediate and Path(intermediate_dir_str).exists():
            logging.info(f"Cleaning up intermediate directory: {intermediate_dir_str}")
            shutil.rmtree(intermediate_dir_str)


if __name__ == "__main__":
    args = Args().parse_args()
    run_pipeline(args)