#!/usr/bin/env python
"""
Minimal script to test RealSense camera RGB+depth synchronization.

This script isolates the camera from the robot to debug color/depth desync issues.
It captures frames directly from the RealSense pipeline and logs timing information.

Usage:
    python examples/debug_camera_depth/test_camera_sync.py --serial <SERIAL_NUMBER>
    python examples/debug_camera_depth/test_camera_sync.py --serial <SERIAL_NUMBER> --duration 10 --save
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print("Error: pyrealsense2 not installed. Run: pip install pyrealsense2")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_cameras():
    """List available RealSense cameras."""
    ctx = rs.context()
    devices = ctx.query_devices()
    cameras = []
    for dev in devices:
        info = {
            "name": dev.get_info(rs.camera_info.name),
            "serial": dev.get_info(rs.camera_info.serial_number),
        }
        cameras.append(info)
    return cameras


def test_camera_sync(
    serial_number: str,
    duration_s: float = 5.0,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    save_data: bool = False,
    output_dir: Path | None = None,
):
    """
    Test camera RGB+depth synchronization.

    Args:
        serial_number: Camera serial number
        duration_s: Recording duration in seconds
        fps: Target frame rate
        width: Frame width
        height: Frame height
        save_data: Whether to save captured frames
        output_dir: Directory to save frames (if save_data=True)
    """
    logger.info(f"Testing camera {serial_number} for {duration_s}s at {fps}fps")

    # Configure pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    # Start pipeline
    profile = pipeline.start(config)

    # Create aligner to align depth to color
    align = rs.align(rs.stream.color)

    # Warmup
    logger.info("Warming up camera...")
    for _ in range(30):
        pipeline.wait_for_frames()

    # Prepare output
    if save_data and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        rgb_dir = output_dir / "rgb"
        depth_dir = output_dir / "depth"
        rgb_dir.mkdir(exist_ok=True)
        depth_dir.mkdir(exist_ok=True)

    # Capture frames
    logger.info(f"Capturing frames for {duration_s}s...")

    frames_data = []
    start_time = time.perf_counter()
    frame_count = 0
    desync_count = 0

    while (time.perf_counter() - start_time) < duration_s:
        # Get frameset
        frameset = pipeline.wait_for_frames()
        capture_time = time.perf_counter() - start_time

        # Get raw frames before alignment (for timestamp check)
        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()

        if not color_frame or not depth_frame:
            logger.warning(f"Frame {frame_count}: Missing color or depth frame")
            continue

        # Get timestamps and frame numbers
        color_ts = color_frame.get_timestamp()
        depth_ts = depth_frame.get_timestamp()
        color_frame_num = color_frame.get_frame_number()
        depth_frame_num = depth_frame.get_frame_number()
        ts_diff_ms = abs(color_ts - depth_ts)

        # Check for desync
        is_desync = ts_diff_ms > 33.0  # More than 1 frame at 30fps
        if is_desync:
            desync_count += 1
            logger.warning(
                f"Frame {frame_count}: DESYNC! ts_diff={ts_diff_ms:.1f}ms, "
                f"color_ts={color_ts:.1f}, depth_ts={depth_ts:.1f}, "
                f"color_frame={color_frame_num}, depth_frame={depth_frame_num}"
            )

        # Align depth to color
        aligned_frameset = align.process(frameset)
        aligned_depth_frame = aligned_frameset.get_depth_frame()

        # Get image data
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(aligned_depth_frame.get_data())

        # Compute depth statistics
        valid_depth = depth_image[depth_image > 0]
        depth_mean = valid_depth.mean() if len(valid_depth) > 0 else 0
        depth_std = valid_depth.std() if len(valid_depth) > 0 else 0

        # Store frame data
        frame_info = {
            "frame_idx": frame_count,
            "capture_time": capture_time,
            "color_ts": color_ts,
            "depth_ts": depth_ts,
            "ts_diff_ms": ts_diff_ms,
            "color_frame_num": color_frame_num,
            "depth_frame_num": depth_frame_num,
            "is_desync": is_desync,
            "depth_mean": depth_mean,
            "depth_std": depth_std,
            "depth_valid_pct": 100 * len(valid_depth) / depth_image.size,
        }
        frames_data.append(frame_info)

        # Save frames if requested
        if save_data and output_dir:
            from PIL import Image
            rgb_img = Image.fromarray(color_image)
            rgb_img.save(rgb_dir / f"frame_{frame_count:06d}.png")

            depth_img = Image.fromarray(depth_image)
            depth_img.save(depth_dir / f"frame_{frame_count:06d}.png")

        frame_count += 1

    # Stop pipeline
    pipeline.stop()

    # Analyze results
    elapsed = time.perf_counter() - start_time
    actual_fps = frame_count / elapsed

    logger.info("=" * 60)
    logger.info("CAPTURE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total frames: {frame_count}")
    logger.info(f"Duration: {elapsed:.2f}s")
    logger.info(f"Actual FPS: {actual_fps:.1f}")
    logger.info(f"Desync frames: {desync_count} ({100*desync_count/frame_count:.1f}%)")

    # Analyze timestamp differences
    ts_diffs = [f["ts_diff_ms"] for f in frames_data]
    logger.info(f"Timestamp diff: mean={np.mean(ts_diffs):.2f}ms, max={np.max(ts_diffs):.2f}ms, std={np.std(ts_diffs):.2f}ms")

    # Analyze depth variations
    depth_means = [f["depth_mean"] for f in frames_data]
    logger.info(f"Depth mean: min={np.min(depth_means):.0f}mm, max={np.max(depth_means):.0f}mm")

    # Detect sudden depth changes (potential desync indicators)
    depth_changes = np.abs(np.diff(depth_means))
    large_changes = np.where(depth_changes > 100)[0]  # >100mm change between frames
    if len(large_changes) > 0:
        logger.warning(f"Large depth changes (>100mm) at frames: {large_changes.tolist()}")
        for idx in large_changes[:10]:  # Show first 10
            f1, f2 = frames_data[idx], frames_data[idx + 1]
            logger.warning(
                f"  Frame {idx}->{idx+1}: depth {f1['depth_mean']:.0f}mm -> {f2['depth_mean']:.0f}mm, "
                f"ts_diff={f1['ts_diff_ms']:.1f}ms -> {f2['ts_diff_ms']:.1f}ms"
            )

    # Save analysis to CSV
    if save_data and output_dir:
        import csv
        csv_path = output_dir / "frame_data.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=frames_data[0].keys())
            writer.writeheader()
            writer.writerows(frames_data)
        logger.info(f"Saved frame data to {csv_path}")

    return frames_data


def main():
    parser = argparse.ArgumentParser(description="Test RealSense camera RGB+depth sync")
    parser.add_argument("--serial", type=str, help="Camera serial number")
    parser.add_argument("--list", action="store_true", help="List available cameras")
    parser.add_argument("--duration", type=float, default=5.0, help="Recording duration in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--save", action="store_true", help="Save captured frames")
    parser.add_argument("--output-dir", type=str, default="./debug_camera_output", help="Output directory")

    args = parser.parse_args()

    if args.list:
        cameras = find_cameras()
        if cameras:
            print("Available RealSense cameras:")
            for cam in cameras:
                print(f"  - {cam['name']}: {cam['serial']}")
        else:
            print("No RealSense cameras found")
        return

    if not args.serial:
        cameras = find_cameras()
        if not cameras:
            print("No RealSense cameras found")
            return
        args.serial = cameras[0]["serial"]
        print(f"Using first available camera: {args.serial}")

    test_camera_sync(
        serial_number=args.serial,
        duration_s=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
        save_data=args.save,
        output_dir=Path(args.output_dir) if args.save else None,
    )


if __name__ == "__main__":
    main()
