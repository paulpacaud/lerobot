#!/usr/bin/env python

"""
Interactive script to move the robot to a specified EE position using inverse kinematics.

Uses the same approach as examples/so100_to_so100_EE/teleoperate.py.

Usage:
```bash
python examples/post_process_dataset/goto_ee_position.py --robot_port=/dev/ttyACM0
```
"""

import logging
import time

import numpy as np
from tap import Tap

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.robot_kinematic_processor import InverseKinematicsEEToJoints
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.utils.rotation import Rotation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_URDF_PATH = "./examples/post_process_dataset/constants/SO101/so101_new_calib.urdf"


class Args(Tap):
    """Arguments for interactive EE position control."""

    robot_port: str = "/dev/ttyACM0"  # USB port for the robot
    robot_id: str = "follower_arm"  # Robot identifier
    urdf_path: str = DEFAULT_URDF_PATH  # Path to the robot URDF file
    target_frame: str = "gripper_frame_link"  # End-effector frame in URDF
    use_degrees: bool = False  # Use degrees for motor control (default matches SO100FollowerConfig)
    duration: float = 5.0  # Duration in seconds to reach target
    steps: int = 100  # Number of interpolation steps


def main():
    args = Args().parse_args()

    # Initialize robot first to get motor names
    logging.info(f"Connecting to robot on port: {args.robot_port}")
    robot_config = SO100FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        use_degrees=args.use_degrees,
    )
    robot = SO100Follower(robot_config)
    robot.connect()

    if not robot.is_connected:
        raise RuntimeError("Failed to connect to robot!")

    # Get motor names from robot (same as examples)
    motor_names = list(robot.bus.motors.keys())
    logging.info(f"Motor names: {motor_names}")

    # Initialize kinematics (same as examples)
    logging.info(f"Loading URDF from: {args.urdf_path}")
    kinematics = RobotKinematics(
        urdf_path=args.urdf_path,
        target_frame_name=args.target_frame,
        joint_names=motor_names,
    )

    # Build pipeline to convert EE action to joints (same as replay.py)
    ee_to_joints_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            InverseKinematicsEEToJoints(
                kinematics=kinematics,
                motor_names=motor_names,
                initial_guess_current_joints=False,  # Same as replay.py
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    print("\nEnter EE position in ROBOT frame: x y z rx ry rz gripper")
    print("  - x, y, z: position in meters")
    print("  - rx, ry, rz: rotation vector (axis-angle)")
    print("  - gripper: 0-100 (0=closed, 100=open)")
    print("Type 'q' to quit, 'r' to read current position\n")

    try:
        while True:
            user_input = input("EE (x y z rx ry rz gripper): ").strip()

            if user_input.lower() == 'q':
                break

            if user_input.lower() == 'r':
                # Read and display current position
                robot_obs = robot.get_observation()
                current_joints = np.array([robot_obs[f"{m}.pos"] for m in motor_names])
                print(f"Current joints: {current_joints}")

                # Compute FK to get current EE position
                T_current = kinematics.forward_kinematics(current_joints)
                position = T_current[:3, 3]
                rot_matrix = T_current[:3, :3]
                rot_vec = Rotation.from_matrix(rot_matrix).as_rotvec()

                print(f"Current EE (robot frame): T={position} R={rot_vec}")
                print(f"Gripper: {current_joints[-1]}")
                print(f"Copy-paste: {position[0]:.6f} {position[1]:.6f} {position[2]:.6f} {rot_vec[0]:.6f} {rot_vec[1]:.6f} {rot_vec[2]:.6f} {current_joints[-1]:.1f}")
                continue

            try:
                values = [float(v) for v in user_input.split()]
                if len(values) != 7:
                    print("Error: Need 7 values (x y z rx ry rz gripper)")
                    continue

                x, y, z, rx, ry, rz, gripper = values

                # Build EE action dict (same format as examples)
                ee_action: RobotAction = {
                    "ee.x": x,
                    "ee.y": y,
                    "ee.z": z,
                    "ee.wx": rx,
                    "ee.wy": ry,
                    "ee.wz": rz,
                    "ee.gripper_pos": gripper,
                }

                print(f"EE target: T=[{x}, {y}, {z}] R=[{rx}, {ry}, {rz}]")

                # Get current robot observation
                robot_obs = robot.get_observation()
                current_joints = np.array([robot_obs[f"{m}.pos"] for m in motor_names])
                print(f"Current joints: {current_joints}")

                # Convert EE action to joint action using the processor (same as examples)
                joint_action = ee_to_joints_processor((ee_action, robot_obs))
                target_joints = np.array([joint_action[f"{m}.pos"] for m in motor_names])
                print(f"IK result (joints): {target_joints}")

                # Verify IK by running FK on the result
                T_verify = kinematics.forward_kinematics(target_joints)
                pos_verify = T_verify[:3, 3]
                rot_verify = Rotation.from_matrix(T_verify[:3, :3]).as_rotvec()
                print(f"FK verify: T={pos_verify} R={rot_verify}")
                print(f"Position error: {np.linalg.norm(pos_verify - np.array([x, y, z])):.6f} m")
                print(f"Rotation error: {np.linalg.norm(rot_verify - np.array([rx, ry, rz])):.4f} rad")

                # Send to robot with slow interpolation
                confirm = input("Send to robot? (y/n): ").strip().lower()
                if confirm == 'y':
                    start_joints = current_joints.copy()

                    print(f"Moving over {args.duration}s in {args.steps} steps...")
                    dt = args.duration / args.steps

                    for step in range(args.steps + 1):
                        t = step / args.steps
                        interp_joints = start_joints + t * (target_joints - start_joints)

                        action = {f"{m}.pos": float(interp_joints[i]) for i, m in enumerate(motor_names)}
                        robot.send_action(action)
                        time.sleep(dt)

                    print("Done!")
                else:
                    print("Cancelled.")

            except ValueError as e:
                print(f"Error parsing input: {e}")

    finally:
        robot.disconnect()
        logging.info("Robot disconnected.")


if __name__ == "__main__":
    main()
