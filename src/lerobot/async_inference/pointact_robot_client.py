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
PointAct Robot Client

A standalone robot client that preprocesses observations to PointAct format
and sends them via HTTP POST with msgpack serialization to a PointAct policy server.

PointAct format requirements:
- State in EE-space: [x, y, z, axis_angle1, axis_angle2, axis_angle3, gripper]
- Images resized to 256x256
- Point clouds from depth images
- msgpack serialization (not pickle)
- HTTP/REST protocol (not gRPC)

Example command:
```shell
python -m lerobot.async_inference.pointact_robot_client \
    --server_address=127.0.0.1:8080 \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm \
    --robot.cameras="{ front: {type: intelrealsense, serial_number_or_name: 147122078460, width: 640, height: 480, fps: 30, use_depth: true}}" \
    --task="put the cube in the green square spot" \
    --urdf_path=./URDF/SO101/so101_new_calib.urdf \
    --translation_offset="[-0.2755, -0.0599, 0.0257]" \
    --intrinsics_file=./examples/post_process_dataset/constants/intrinsics.npz \
    --extrinsics_file=./examples/post_process_dataset/constants/extrinsics.npz \
    --repo_id=user/pointact_model \
    --fps=30
```
"""

import logging
import time
from dataclasses import asdict
from pprint import pformat
from typing import Any

import draccus
import numpy as np
import zmq

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import (
    RealSenseCameraConfig,  # noqa: F401
)
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so100_follower,
    so101_follower,
)
from lerobot.robots.so100_follower.robot_kinematic_processor import InverseKinematicsEEToJoints

from .constants import SUPPORTED_ROBOTS
from .helpers import FPSTracker, get_logger
from .pointact_configs import PointActClientConfig
from .pointact_utils import (
    deserialize_response,
    ee_action_to_transform,
    get_point_cloud_from_rgb_depth,
    joints_to_ee,
    load_extrinsics,
    load_intrinsics,
    pack_pointact_batch,
    resize_image,
    serialize_batch,
)


class PointActRobotClient:
    """Robot client for PointAct policy inference.

    Collects observations from robot, preprocesses to PointAct format,
    sends via HTTP POST to policy server, and executes returned actions.
    """

    PREFIX = "pointact_client"

    def __init__(self, config: PointActClientConfig):
        """Initialize PointAct robot client.

        Args:
            config: PointActClientConfig containing all configuration parameters
        """
        self.logger = get_logger(self.PREFIX)
        self.config = config

        # Initialize robot
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()
        self.logger.info("Robot connected")

        # Initialize kinematics for FK
        motor_names = list(self.robot.bus.motors.keys())
        self.motor_names = motor_names
        self.kinematics = RobotKinematics(
            urdf_path=config.urdf_path,
            target_frame_name=config.target_frame,
            joint_names=config.joint_names,
        )
        self.translation_offset = np.array(config.translation_offset, dtype=np.float64)
        self.logger.info(f"FK initialized with URDF: {config.urdf_path}")
        self.logger.info(f"Target frame: {config.target_frame}")
        self.logger.info(f"Translation offset: {config.translation_offset}")

        # Initialize IK processor for converting EE actions to joints
        self.ik_processor = RobotProcessorPipeline(
            steps=[
                InverseKinematicsEEToJoints(
                    kinematics=self.kinematics,
                    motor_names=motor_names,
                    initial_guess_current_joints=False,
                ),
            ],
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )
        self.logger.info("IK processor initialized")

        # Initialize point cloud processor if calibration files are provided
        self.intrinsics = None
        self.extrinsics = None
        if config.intrinsics_file and config.extrinsics_file:
            self.intrinsics = load_intrinsics(config.intrinsics_file)
            self.extrinsics = load_extrinsics(config.extrinsics_file)
            self.logger.info(f"Point cloud processor initialized")
            self.logger.info(f"  Intrinsics: {config.intrinsics_file}")
            self.logger.info(f"  Extrinsics: {config.extrinsics_file}")
            self.logger.info(f"  Voxel size: {config.voxel_size}")
            self.logger.info(f"  Workspace: {config.workspace}")
        else:
            self.logger.warning(
                "Point cloud processing disabled (no intrinsics/extrinsics files)"
            )

        # ZeroMQ socket for server communication
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.REQ)
        server_addr = f"tcp://{config.server_address}"
        self.socket.connect(server_addr)
        self.logger.info(f"Connected to ZMQ server: {server_addr}")

        # FPS tracking
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        # Control state
        self.running = False
        self.timestep = 0

        # Action queue for chunked actions
        self.action_queue: list[np.ndarray] = []
        self.action_chunk_index = 0

    def preprocess_observation(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        """Preprocess robot observation to PointAct format.

        Args:
            raw_obs: Raw observation from robot containing:
                - Joint positions as {motor_name}.pos keys
                - RGB image as observation.images.{camera_key}
                - Depth image as observation.images.{camera_key}_depth (optional)

        Returns:
            Preprocessed observation dict with PointAct format keys
        """
        # Extract joint positions
        joint_positions = np.array(
            [
                raw_obs[f"{name}.pos"]
                for name in self.motor_names
            ],
            dtype=np.float32,
        )

        # Convert joints to EE pose using FK
        ee_pose, ee_pose_with_gripper = joints_to_ee(
            joint_positions,
            self.kinematics,
            self.translation_offset,
        )

        # Build preprocessed observation
        processed_obs = {
            "observation.state": ee_pose_with_gripper,
            "observation.states.ee_state": ee_pose,
            "observation.states.joint_state": joint_positions,
            "observation.states.gripper_state": np.array(
                [joint_positions[-1]], dtype=np.float32
            ),
        }

        # Process RGB image
        # Robot returns camera images directly as {camera_name}, not observation.images.{camera_name}
        image_key = self.config.source_image_key
        if image_key in raw_obs:
            rgb_image = raw_obs[image_key]
            # Resize to target size
            resized_image = resize_image(rgb_image, self.config.image_size)
            processed_obs["observation.images.front_image"] = resized_image
        else:
            self.logger.warning(f"RGB image key '{image_key}' not found in observation")

        # Process depth image to point cloud
        depth_key = f"{self.config.source_image_key}_depth"
        if (
            depth_key in raw_obs
            and self.intrinsics is not None
            and self.extrinsics is not None
        ):
            depth_image = raw_obs[depth_key]
            # Handle depth image format: may be (H, W, 1) or (H, W), float or uint16
            if depth_image.ndim == 3:
                depth_image = depth_image[:, :, 0]  # Remove channel dimension
            # Convert depth to meters if needed
            if depth_image.dtype in (np.uint16, np.int16, np.int32):
                depth_meters = depth_image.astype(np.float32) / self.config.depth_scale
            else:
                # Already in float (likely meters from RealSense)
                depth_meters = depth_image.astype(np.float32)
            # Get RGB for coloring
            rgb_image = raw_obs.get(image_key)
            if rgb_image is not None:
                point_cloud = get_point_cloud_from_rgb_depth(
                    rgb_image,
                    depth_meters,
                    self.intrinsics,
                    self.extrinsics,
                    workspace=self.config.workspace,
                    voxel_size=self.config.voxel_size,
                )
                processed_obs["observation.points.frontview"] = point_cloud

        # Add task
        processed_obs["task"] = raw_obs.get("task", self.config.task)

        return processed_obs

    def send_observation(self, processed_obs: dict[str, Any]) -> dict[str, Any] | None:
        """Send preprocessed observation to policy server via HTTP POST.

        Args:
            processed_obs: Preprocessed observation dict

        Returns:
            Response dict with action or None if failed
        """
        # Pack into batch format - use joint_state (6,)
        state = processed_obs["observation.states.joint_state"]
        image = processed_obs.get("observation.images.front_image", np.zeros((256, 256, 3), dtype=np.uint8))
        point_cloud = processed_obs.get("observation.points.frontview", np.zeros((0, 6), dtype=np.float32))
        task = processed_obs.get("task", self.config.task)

        # Debug: print batch contents
        self.logger.info("=" * 50)
        self.logger.info("BATCH CONTENTS:")
        self.logger.info(f"  observation.state:")
        self.logger.info(f"    shape: {state.shape}, dtype: {state.dtype}")
        self.logger.info(f"    values: [{', '.join(f'{v:.2f}' for v in state)}]")
        self.logger.info(f"  observation.images.front_image:")
        self.logger.info(f"    shape: {image.shape}, dtype: {image.dtype}")
        self.logger.info(f"    min: {image.min()}, max: {image.max()}")
        self.logger.info(f"  observation.points:")
        self.logger.info(f"    shape: {point_cloud.shape}, dtype: {point_cloud.dtype}")
        if len(point_cloud) > 0:
            self.logger.info(f"    xyz range: x=[{point_cloud[:,0].min():.3f}, {point_cloud[:,0].max():.3f}], "
                           f"y=[{point_cloud[:,1].min():.3f}, {point_cloud[:,1].max():.3f}], "
                           f"z=[{point_cloud[:,2].min():.3f}, {point_cloud[:,2].max():.3f}]")
        self.logger.info(f"  task: '{task}'")
        self.logger.info(f"  repo_id: '{self.config.repo_id}'")
        self.logger.info("=" * 50)

        batch = pack_pointact_batch(
            state=state,
            image=image,
            point_cloud=point_cloud,
            task=task,
            repo_id=self.config.repo_id,
        )

        # Build ZMQ request in the format expected by PolicyServer
        request = {
            "endpoint": "get_action",
            "data": {"batch": batch, "options": {'pred_rot_type': None}},
        }

        try:
            # Send ZMQ request
            start_time = time.perf_counter()
            request_bytes = serialize_batch(request)
            self.socket.send(request_bytes)

            # Receive response
            response_bytes = self.socket.recv()
            request_time = (time.perf_counter() - start_time) * 1000

            # Deserialize response
            result = deserialize_response(response_bytes)

            # Check for server error
            if isinstance(result, dict) and "error" in result:
                self.logger.error(f"Server error: {result['error']}")
                return None

            self.logger.debug(
                f"Sent observation #{self.timestep}, received action | "
                f"Request time: {request_time:.2f}ms"
            )
            return result

        except zmq.ZMQError as e:
            self.logger.error(f"ZMQ request failed: {e}")
            return None

    def ee_action_to_joint_action(
        self,
        ee_action: np.ndarray,
        robot_obs: dict[str, Any],
    ) -> dict[str, float]:
        """Convert EE-space action to joint-space action using IK.

        Args:
            ee_action: Array of shape (7,) with [x, y, z, rx, ry, rz, gripper]
            robot_obs: Current robot observation for IK initial guess

        Returns:
            Joint action dict with {motor_name}.pos keys
        """
        # Build EE action dict
        ee_action_dict = {
            "ee.x": float(ee_action[0]),
            "ee.y": float(ee_action[1]),
            "ee.z": float(ee_action[2]),
            "ee.wx": float(ee_action[3]),
            "ee.wy": float(ee_action[4]),
            "ee.wz": float(ee_action[5]),
            "ee.gripper_pos": float(ee_action[6]),
        }

        # Apply IK processor
        joint_action = self.ik_processor((ee_action_dict, robot_obs))
        return joint_action

    def store_action_chunk(self, action_response: dict[str, Any]) -> None:
        """Store action chunk from server response in queue.

        Args:
            action_response: Server response containing action chunk
        """
        action_data = action_response.get("action")
        if action_data is None:
            self.logger.error("No action in server response")
            return

        # Convert to numpy array
        action_arr = np.array(action_data)
        self.logger.info(f"Raw action chunk: shape={action_arr.shape}")

        # Remove batch dimension if present (shape: [1, chunk_size, action_dim] -> [chunk_size, action_dim])
        while action_arr.ndim > 2:
            action_arr = action_arr[0]

        # Handle single action case (shape: [action_dim] -> [1, action_dim])
        if action_arr.ndim == 1:
            action_arr = action_arr.reshape(1, -1)

        # Now shape should be [chunk_size, action_dim]
        chunk_size = action_arr.shape[0]
        self.logger.info(f"Action chunk: {chunk_size} actions, each with {action_arr.shape[1]} dims")

        # Store each action in queue
        self.action_queue = [action_arr[i].astype(np.float32) for i in range(chunk_size)]
        self.logger.info(f"Stored {len(self.action_queue)} actions in queue")

    def execute_single_action(self, joint_values: np.ndarray) -> dict[str, float]:
        """Execute a single action (joint values).

        Args:
            joint_values: Array of joint positions

        Returns:
            Performed action dict
        """
        # Build joint action dict
        joint_action = {
            f"{name}.pos": float(joint_values[i])
            for i, name in enumerate(self.motor_names)
        }

        # Send action to robot
        performed_action = self.robot.send_action(joint_action)

        self.logger.info(
            f"Executed action #{self.timestep} (queue: {len(self.action_queue)} left) | "
            f"Joints: [{', '.join(f'{v:.1f}' for v in joint_values)}]"
        )

        return performed_action

    def control_loop(self, verbose: bool = False) -> None:
        """Main control loop: observe, send, receive, execute.

        Args:
            verbose: Enable verbose logging
        """
        self.running = True
        self.logger.info("Control loop starting")
        self.logger.info(f"Task: {self.config.task}")
        self.logger.info(f"Target FPS: {self.config.fps}")

        try:
            while self.running:
                loop_start = time.perf_counter()

                # Get raw observation from robot
                raw_obs = self.robot.get_observation()
                raw_obs["task"] = self.config.task

                # Only request new actions if queue is empty
                if len(self.action_queue) == 0:
                    # Preprocess to PointAct format
                    processed_obs = self.preprocess_observation(raw_obs)

                    # Send to server and get action chunk
                    action_response = self.send_observation(processed_obs)

                    if action_response is not None:
                        # Store action chunk in queue
                        self.store_action_chunk(action_response)

                # Execute next action from queue
                if len(self.action_queue) > 0:
                    action = self.action_queue.pop(0)
                    self.execute_single_action(action)

                self.timestep += 1

                # FPS tracking
                if verbose:
                    fps_metrics = self.fps_tracker.calculate_fps_metrics(time.time())
                    self.logger.info(
                        f"Step #{self.timestep} | "
                        f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                        f"Target: {fps_metrics['target_fps']:.2f}"
                    )

                # Sleep to maintain target FPS
                loop_time = time.perf_counter() - loop_start
                sleep_time = max(0, self.config.environment_dt - loop_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            self.logger.info("Control loop interrupted by user")
        finally:
            self.running = False

    def stop(self) -> None:
        """Stop the client and disconnect robot."""
        self.running = False
        self.robot.disconnect()
        self.socket.close()
        self.zmq_context.term()
        self.logger.info("Client stopped")


@draccus.wrap()
def pointact_client(cfg: PointActClientConfig):
    """Main entry point for PointAct robot client."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.info(pformat(asdict(cfg)))

    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = PointActRobotClient(cfg)

    try:
        client.control_loop(verbose=True)
    finally:
        client.stop()


if __name__ == "__main__":
    pointact_client()
