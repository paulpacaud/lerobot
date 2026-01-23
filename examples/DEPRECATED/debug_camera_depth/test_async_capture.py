#!/usr/bin/env python
"""
Test the async capture pattern used by the robot code.

This script simulates how the lerobot recording loop interacts with the camera,
using background thread capture + main thread reading pattern.

Usage:
    python examples/debug_camera_depth/test_async_capture.py --serial <SERIAL_NUMBER>
    python examples/debug_camera_depth/test_async_capture.py --serial <SERIAL_NUMBER> --duration 10 --save
"""

import argparse
import logging
import time
from pathlib import Path
from threading import Thread, Event, Lock

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print("Error: pyrealsense2 not installed. Run: pip install pyrealsense2")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AsyncRealSenseCapture:
    """
    Simulates the async capture pattern used in camera_realsense.py.
    """

    def __init__(self, serial_number: str, fps: int = 30, width: int = 640, height: int = 480):
        self.serial_number = serial_number
        self.fps = fps
        self.width = width
        self.height = height

        self.pipeline = None
        self.align = None

        # Threading
        self.thread = None
        self.stop_event = Event()
        self.frame_lock = Lock()
        self.new_frame_event = Event()

        # Frame storage
        self.latest_color = None
        self.latest_depth = None
        self.latest_color_ts = 0
        self.latest_depth_ts = 0
        self.latest_color_frame_num = 0
        self.latest_depth_frame_num = 0
        self.capture_count = 0

    def connect(self):
        """Connect to camera and start pipeline."""
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial_number)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # Warmup
        for _ in range(30):
            self.pipeline.wait_for_frames()

        logger.info(f"Camera {self.serial_number} connected")

    def start_capture_thread(self):
        """Start background capture thread."""
        self.stop_event.clear()
        self.thread = Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info("Background capture thread started")

    def _capture_loop(self):
        """Background thread that continuously captures frames."""
        while not self.stop_event.is_set():
            try:
                frameset = self.pipeline.wait_for_frames(timeout_ms=500)

                color_frame = frameset.get_color_frame()
                depth_frame = frameset.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                # Get timestamps before alignment
                color_ts = color_frame.get_timestamp()
                depth_ts = depth_frame.get_timestamp()
                color_frame_num = color_frame.get_frame_number()
                depth_frame_num = depth_frame.get_frame_number()

                # Align depth to color
                aligned = self.align.process(frameset)
                aligned_depth = aligned.get_depth_frame()

                # Get image arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(aligned_depth.get_data())

                # Store with lock
                with self.frame_lock:
                    self.latest_color = color_image.copy()
                    self.latest_depth = depth_image.copy()
                    self.latest_color_ts = color_ts
                    self.latest_depth_ts = depth_ts
                    self.latest_color_frame_num = color_frame_num
                    self.latest_depth_frame_num = depth_frame_num
                    self.capture_count += 1

                self.new_frame_event.set()

            except Exception as e:
                logger.warning(f"Capture error: {e}")
                time.sleep(0.01)

    def read_color_and_depth(self, timeout_ms: float = 200):
        """
        Read latest color and depth frames (simulates async_read_color_and_depth).
        Returns: (color, depth, color_ts, depth_ts, color_frame_num, depth_frame_num)
        """
        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError("Timed out waiting for frame")

        with self.frame_lock:
            color = self.latest_color
            depth = self.latest_depth
            color_ts = self.latest_color_ts
            depth_ts = self.latest_depth_ts
            color_frame_num = self.latest_color_frame_num
            depth_frame_num = self.latest_depth_frame_num
            self.new_frame_event.clear()

        return color, depth, color_ts, depth_ts, color_frame_num, depth_frame_num

    def stop(self):
        """Stop capture thread and pipeline."""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.pipeline:
            self.pipeline.stop()
        logger.info("Camera stopped")


def test_async_capture(
    serial_number: str,
    duration_s: float = 5.0,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    save_data: bool = False,
    output_dir: Path | None = None,
):
    """
    Test async capture pattern with simulated recording loop.
    """
    logger.info(f"Testing async capture for {duration_s}s at {fps}fps")

    camera = AsyncRealSenseCapture(serial_number, fps, width, height)
    camera.connect()
    camera.start_capture_thread()

    # Prepare output
    if save_data and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        rgb_dir = output_dir / "rgb"
        depth_dir = output_dir / "depth"
        rgb_dir.mkdir(exist_ok=True)
        depth_dir.mkdir(exist_ok=True)

    # Simulate recording loop at target FPS
    frame_interval = 1.0 / fps
    frames_data = []
    start_time = time.perf_counter()
    frame_count = 0
    desync_count = 0
    last_read_time = start_time

    logger.info(f"Starting recording loop at {fps}fps...")

    while (time.perf_counter() - start_time) < duration_s:
        loop_start = time.perf_counter()

        try:
            # Read frames (simulates get_observation)
            color, depth, color_ts, depth_ts, color_fn, depth_fn = camera.read_color_and_depth()
            read_time = time.perf_counter() - start_time

            ts_diff_ms = abs(color_ts - depth_ts)
            is_desync = ts_diff_ms > 33.0

            if is_desync:
                desync_count += 1
                logger.warning(
                    f"Frame {frame_count}: DESYNC ts_diff={ts_diff_ms:.1f}ms "
                    f"(color={color_fn}, depth={depth_fn})"
                )

            # Compute depth stats
            valid_depth = depth[depth > 0]
            depth_mean = valid_depth.mean() if len(valid_depth) > 0 else 0

            frame_info = {
                "frame_idx": frame_count,
                "read_time": read_time,
                "color_ts": color_ts,
                "depth_ts": depth_ts,
                "ts_diff_ms": ts_diff_ms,
                "color_frame_num": color_fn,
                "depth_frame_num": depth_fn,
                "is_desync": is_desync,
                "depth_mean": depth_mean,
                "loop_time_ms": (time.perf_counter() - loop_start) * 1000,
            }
            frames_data.append(frame_info)

            # Save frames
            if save_data and output_dir:
                from PIL import Image
                Image.fromarray(color).save(rgb_dir / f"frame_{frame_count:06d}.png")
                Image.fromarray(depth).save(depth_dir / f"frame_{frame_count:06d}.png")

            frame_count += 1

        except TimeoutError:
            logger.warning(f"Frame {frame_count}: Timeout waiting for frame")

        # Sleep to maintain target FPS
        elapsed = time.perf_counter() - loop_start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    camera.stop()

    # Analyze results
    elapsed = time.perf_counter() - start_time
    actual_fps = frame_count / elapsed

    logger.info("=" * 60)
    logger.info("ASYNC CAPTURE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total frames: {frame_count}")
    logger.info(f"Duration: {elapsed:.2f}s")
    logger.info(f"Actual FPS: {actual_fps:.1f}")
    logger.info(f"Background captures: {camera.capture_count}")
    logger.info(f"Desync frames: {desync_count} ({100*desync_count/max(frame_count,1):.1f}%)")

    # Analyze timestamp differences
    if frames_data:
        ts_diffs = [f["ts_diff_ms"] for f in frames_data]
        logger.info(f"Timestamp diff: mean={np.mean(ts_diffs):.2f}ms, max={np.max(ts_diffs):.2f}ms")

        # Analyze depth variations
        depth_means = [f["depth_mean"] for f in frames_data]
        logger.info(f"Depth mean: min={np.min(depth_means):.0f}mm, max={np.max(depth_means):.0f}mm")

        # Detect sudden depth changes
        depth_changes = np.abs(np.diff(depth_means))
        large_changes = np.where(depth_changes > 100)[0]
        if len(large_changes) > 0:
            logger.warning(f"Large depth changes (>100mm) at {len(large_changes)} frames")
            for idx in large_changes[:5]:
                f1, f2 = frames_data[idx], frames_data[idx + 1]
                logger.warning(
                    f"  Frame {idx}: {f1['depth_mean']:.0f}mm -> {f2['depth_mean']:.0f}mm"
                )

        # Analyze loop timing
        loop_times = [f["loop_time_ms"] for f in frames_data]
        logger.info(f"Loop time: mean={np.mean(loop_times):.1f}ms, max={np.max(loop_times):.1f}ms")

    # Save analysis
    if save_data and output_dir:
        import csv
        csv_path = output_dir / "frame_data.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=frames_data[0].keys())
            writer.writeheader()
            writer.writerows(frames_data)
        logger.info(f"Saved frame data to {csv_path}")

    return frames_data


def find_cameras():
    """List available RealSense cameras."""
    ctx = rs.context()
    devices = ctx.query_devices()
    return [
        {"name": dev.get_info(rs.camera_info.name), "serial": dev.get_info(rs.camera_info.serial_number)}
        for dev in devices
    ]


def main():
    parser = argparse.ArgumentParser(description="Test async capture pattern")
    parser.add_argument("--serial", type=str, help="Camera serial number")
    parser.add_argument("--list", action="store_true", help="List available cameras")
    parser.add_argument("--duration", type=float, default=5.0, help="Recording duration in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--save", action="store_true", help="Save captured frames")
    parser.add_argument("--output-dir", type=str, default="./debug_async_output", help="Output directory")

    args = parser.parse_args()

    if args.list:
        cameras = find_cameras()
        if cameras:
            print("Available cameras:")
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

    test_async_capture(
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
