#!/usr/bin/env python

"""
Read the current end-effector pose of the robot using forward kinematics.

Usage:
```bash
python examples/post_process_dataset/read_ee_position.py --robot_port=/dev/ttyACM0
```
"""

import numpy as np
from tap import Tap

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.utils.rotation import Rotation

DEFAULT_URDF_PATH = "./examples/post_process_dataset/constants/SO101/so101_new_calib.urdf"

ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ALL_MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class Args(Tap):
    """Arguments for reading EE position."""

    robot_port: str = "/dev/ttyACM0"  # USB port for the robot
    robot_id: str = "follower_arm"  # Robot identifier
    urdf_path: str = DEFAULT_URDF_PATH  # Path to the robot URDF file
    target_frame: str = "gripper_frame_link"  # End-effector frame in URDF
    use_degrees: bool = True  # Use degrees for motor control


def get_current_ee_pose(robot, kinematics) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Get the current EE pose in robot frame.

    Returns:
        position: (3,) array [x, y, z] in meters
        rotation_vec: (3,) array [rx, ry, rz] axis-angle
        gripper: float 0-100
    """
    robot_obs = robot.get_observation()
    current_joints = np.array([robot_obs[f"{m}.pos"] for m in ALL_MOTOR_NAMES])

    T_current = kinematics.forward_kinematics(current_joints[:5])
    position = T_current[:3, 3]
    rot_matrix = T_current[:3, :3]
    rotation_vec = Rotation.from_matrix(rot_matrix).as_rotvec()
    gripper = current_joints[5]

    return position, rotation_vec, gripper


def main():
    args = Args().parse_args()

    # Initialize kinematics
    kinematics = RobotKinematics(
        urdf_path=args.urdf_path,
        target_frame_name=args.target_frame,
        joint_names=ARM_JOINT_NAMES,
    )

    # Initialize robot
    robot_config = SO100FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        use_degrees=args.use_degrees,
    )
    robot = SO100Follower(robot_config)
    robot.connect()

    try:
        position, rotation_vec, gripper = get_current_ee_pose(robot, kinematics)

        print(f"EE pose (robot frame):")
        print(f"  Position: x={position[0]:.4f} y={position[1]:.4f} z={position[2]:.4f}")
        print(f"  Rotation: rx={rotation_vec[0]:.4f} ry={rotation_vec[1]:.4f} rz={rotation_vec[2]:.4f}")
        print(f"  Gripper:  {gripper:.1f}")
        print(f"\nAs single line: {position[0]:.4f} {position[1]:.4f} {position[2]:.4f} {rotation_vec[0]:.4f} {rotation_vec[1]:.4f} {rotation_vec[2]:.4f} {gripper:.1f}")

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
