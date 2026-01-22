#!/usr/bin/env python3

"""
Standalone motion script for LeRobot follower arms.
This script creates a slow, smooth motion sequence for a follower arm without
requiring a leader arm.
"""

import argparse
import math
import time

from lerobot.robots.so100_follower.config_so100_follower import (
    SO100FollowerConfig,
)
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.utils.robot_utils import precise_sleep


def execute_smooth_motion(
    robot,
    amplitude: float,
    duration: float,
    fps: float,
    loops: int,
):
    """
    Execute a smooth sinusoidal motion sequence on the robot.

    Args:
        robot: The robot instance to control
        duration: Duration of each motion loop in seconds
        amplitude: Motion amplitude from 0 (stay at mid-range configuration) to
            100 (full amplitude of each joint).
        fps: Frames per second for motion playback
        loops: Number of times to loop the motion
    """
    num_steps = int(duration * fps)
    print(
        f"Starting motion playback at {fps} FPS "
        f"for {loops} loops ({duration:.1f}s each)..."
    )

    for loop in range(loops):
        print(f"Loop {loop + 1}/{loops}")

        for i in range(num_steps):
            start_time = time.perf_counter()

            t = i / num_steps * 2 * math.pi

            base_positions = {
                "shoulder_pan.pos": 0.0,
                "shoulder_lift.pos": 0.0,
                "elbow_flex.pos": 0.0,
                "wrist_flex.pos": 0.0,
                "wrist_roll.pos": 0.0,
                "gripper.pos": 50.0,  # Middle position for gripper
            }

            # Add smooth sinusoidal variations
            action = {
                "shoulder_pan.pos": base_positions["shoulder_pan.pos"]
                + amplitude * math.sin(t),
                "shoulder_lift.pos": base_positions["shoulder_lift.pos"]
                + amplitude * math.sin(t * 0.7),
                "elbow_flex.pos": base_positions["elbow_flex.pos"]
                + amplitude * math.sin(t * 1.3),
                "wrist_flex.pos": base_positions["wrist_flex.pos"]
                + amplitude * math.sin(t * 0.9),
                "wrist_roll.pos": base_positions["wrist_roll.pos"]
                + amplitude * math.sin(t * 1.5),
                "gripper.pos": base_positions["gripper.pos"]
                + 20 * math.sin(t * 0.5),  # Gentle gripper motion
            }

            robot.send_action(action)
            current_positions = robot.get_observation()

            elapsed = time.perf_counter() - start_time
            target_dt = 1.0 / fps
            precise_sleep(max(target_dt - elapsed, 0.0))

            if (i + 1) % 10 == 0:
                print(f"  Step {i + 1}/{num_steps}")
                print(f"    Target action: {action}")
                print(f"    Current positions: {current_positions}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Play a slow standalone motion on an SO-100 follower arm",
    )
    parser.add_argument(
        "port",
        help="Serial port for the arm, e.g. /dev/ttyACM0",
    )
    parser.add_argument(
        "id",
        help="Robot ID used to locate the calibration file, "
        'e.g. "follower_arm"',
    )
    parser.add_argument(
        "amplitude",
        type=float,
        help="Motion amplitude from 0 (stay at mid-range configuration) "
        "to 100 (full amplitude of each joint).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=40.0,
        help="Frames per second for motion playback",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration of each motion loop in seconds",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=3,
        help="Number of times to loop the motion",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.amplitude > 50.0:
        print(
            f"{args.amplitude = }% > 50% is disabled as it explores the regime"
            "where the arm pushes harder and harder against the table, if "
            "there is one. Try it if there is no table under the arm."
        )
        return

    print(f"Creating an SO-100 follower arm on port {args.port}")
    config = SO100FollowerConfig(
        id=args.id,
        port=args.port,
        disable_torque_on_disconnect=True,
        max_relative_target=None,
        use_degrees=False,  # use RANGE_M100_100 rather than DEGREES
    )

    try:
        robot = SO100Follower(config)
    except Exception as e:
        print(f"Error creating robot: {e}")
        return

    try:
        print("Connecting to robot...")
        robot.connect(calibrate=False)
        if not robot.is_connected:
            print("Failed to connect to robot!")
            return
        print("Robot connected successfully.")

        print("Starting motion sequence...")
        execute_smooth_motion(
            robot,
            duration=args.duration,
            amplitude=args.amplitude,
            fps=args.fps,
            loops=args.loops,
        )
        print("Motion sequence completed.")

    except KeyboardInterrupt:
        print("\nMotion interrupted by user")
    except Exception as e:
        print(f"Error during motion execution: {e}")
    finally:
        if robot.is_connected:
            print("Disconnecting robot...")
            robot.disconnect()
            print("Robot disconnected")


if __name__ == "__main__":
    main()
