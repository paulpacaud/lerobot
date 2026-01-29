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

"""Utility functions for PointAct preprocessing.

This module provides functions for:
- Forward kinematics: converting joint positions to EE poses
- Point cloud generation from depth images
- Image resizing
- Batch packing for msgpack serialization
"""

import logging
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.model.kinematics import RobotKinematics

logger = logging.getLogger(__name__)


def joints_to_ee(
    joint_values: np.ndarray,
    kinematics: RobotKinematics,
    translation_offset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert joint positions to end-effector pose using forward kinematics.

    Args:
        joint_values: Array of shape (6,) with joint positions in degrees
                     [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        kinematics: RobotKinematics instance for FK computation
        translation_offset: Optional (3,) array with [tx, ty, tz] offset

    Returns:
        Tuple of:
            - ee_pose: Array of shape (6,) with [x, y, z, rx, ry, rz] (axis-angle)
            - ee_pose_with_gripper: Array of shape (7,) with [x, y, z, rx, ry, rz, gripper]
    """
    arm_joints = joint_values[:5].astype(np.float64)
    gripper_pos = float(joint_values[5])

    # Compute forward kinematics
    T = kinematics.forward_kinematics(arm_joints)
    position = T[:3, 3]

    # Apply translation offset if provided
    if translation_offset is not None:
        position = position + translation_offset

    # Convert rotation matrix to axis-angle (rotation vector)
    rotation_vec = Rotation.from_matrix(T[:3, :3]).as_rotvec()

    ee_pose = np.concatenate([position, rotation_vec]).astype(np.float32)
    ee_pose_with_gripper = np.concatenate([ee_pose, [gripper_pos]]).astype(np.float32)

    return ee_pose, ee_pose_with_gripper


def load_intrinsics(intrinsics_file: str) -> np.ndarray:
    """Load camera intrinsics matrix from npz file.

    Args:
        intrinsics_file: Path to the intrinsics .npz file

    Returns:
        3x3 camera intrinsic matrix K
    """
    data = np.load(intrinsics_file)
    return data["K"]


def load_extrinsics(extrinsics_file: str) -> np.ndarray:
    """Load camera extrinsics and convert to 4x4 camera-to-world matrix.

    Args:
        extrinsics_file: Path to the extrinsics .npz file

    Returns:
        4x4 camera-to-world transformation matrix
    """
    data = np.load(extrinsics_file)
    rvec = data["rvec"]
    tvec = data["tvec"]
    R, _ = cv2.Rodrigues(rvec)
    T_w2c = np.eye(4)
    T_w2c[:3, :3] = R
    T_w2c[:3, 3] = tvec.flatten()
    T_c2w = np.linalg.inv(T_w2c)
    return T_c2w


def depth_to_point_cloud(
    depth_image: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world_matrix: np.ndarray,
) -> np.ndarray:
    """Convert depth image to point cloud in world coordinates.

    Args:
        depth_image: 2D depth image (H, W) in meters
        intrinsics: 3x3 camera intrinsic matrix
        camera_to_world_matrix: 4x4 camera-to-world transformation matrix

    Returns:
        Point cloud of shape (H, W, 3) in world coordinates
    """
    height, width = depth_image.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    z = depth_image
    x_cam = (x - cx) * z / fx
    y_cam = (y - cy) * z / fy
    camera_coords = np.stack((x_cam, y_cam, z), axis=-1)
    points = camera_coords.reshape(-1, 3)
    points_homo = np.column_stack((points, np.ones(len(points))))
    world_points = (camera_to_world_matrix @ points_homo.T).T[:, :3]
    world_points = world_points.reshape(height, width, 3)
    return world_points


def crop_point_cloud_by_workspace(point_cloud: np.ndarray, workspace: dict) -> np.ndarray:
    """Filter point cloud to only include points within workspace bounds.

    Args:
        point_cloud: Point cloud of shape (N, 6) with [x, y, z, r, g, b]
        workspace: Dict with 'X_BBOX', 'Y_BBOX', 'Z_BBOX' keys

    Returns:
        Filtered point cloud within workspace bounds
    """
    point_mask = (
        (point_cloud[..., 0] > workspace["X_BBOX"][0])
        & (point_cloud[..., 0] < workspace["X_BBOX"][1])
        & (point_cloud[..., 1] > workspace["Y_BBOX"][0])
        & (point_cloud[..., 1] < workspace["Y_BBOX"][1])
        & (point_cloud[..., 2] > workspace["Z_BBOX"][0])
        & (point_cloud[..., 2] < workspace["Z_BBOX"][1])
    )
    return point_cloud[point_mask]


def voxelize_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """Voxelize point cloud using Open3D.

    Args:
        points: Point cloud of shape (N, 3) with [x, y, z]
        colors: Colors of shape (N, 3) with [r, g, b] normalized to 0-1
        voxel_size: Voxel size for downsampling in meters

    Returns:
        Voxelized point cloud of shape (M, 6) with [x, y, z, r, g, b]
    """
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        point_cloud = np.concatenate(
            [np.asarray(pcd.points), np.asarray(pcd.colors)], axis=1
        ).astype(np.float32)
        return point_cloud
    except ImportError:
        logger.warning("Open3D not available, skipping voxelization")
        return np.concatenate([points, colors], axis=1).astype(np.float32)


def get_point_cloud_from_rgb_depth(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    workspace: dict | None = None,
    voxel_size: float = 0.01,
) -> np.ndarray:
    """Generate a processed point cloud from RGB and depth images.

    Args:
        rgb: RGB image of shape (H, W, 3) with values 0-255
        depth: Depth image of shape (H, W) in meters
        intrinsics: 3x3 camera intrinsic matrix
        extrinsics: 4x4 camera-to-world transformation matrix
        workspace: Optional workspace bounds for cropping
        voxel_size: Voxel size for downsampling (0 to skip)

    Returns:
        Point cloud of shape (N, 6) with [x, y, z, r, g, b]
    """
    # Convert depth to 3D points in world coordinates
    point_cloud_xyz = depth_to_point_cloud(depth, intrinsics, extrinsics)
    point_cloud_xyz = point_cloud_xyz.astype(np.float32)

    # Normalize RGB to 0-1
    rgb_normalized = rgb.astype(np.float32) / 255.0

    # Combine xyz and rgb
    point_cloud = np.concatenate([point_cloud_xyz, rgb_normalized], axis=2)
    point_cloud = point_cloud.reshape(-1, 6)

    # Filter out invalid depth points (z > 0)
    valid_mask = point_cloud[:, 2] > 0
    point_cloud = point_cloud[valid_mask]

    # Crop to workspace
    if workspace is not None:
        point_cloud = crop_point_cloud_by_workspace(point_cloud, workspace)

    # Voxelize
    if voxel_size > 0 and len(point_cloud) > 0:
        point_cloud = voxelize_point_cloud(
            point_cloud[:, :3], point_cloud[:, 3:], voxel_size
        )

    return point_cloud


def resize_image(image: np.ndarray, target_size: int) -> np.ndarray:
    """Resize image to target square size.

    Args:
        image: Input image of shape (H, W, 3)
        target_size: Target size for both width and height

    Returns:
        Resized image of shape (target_size, target_size, 3)
    """
    return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def pack_pointact_batch(
    state: np.ndarray,
    image: np.ndarray,
    point_cloud: np.ndarray,
    task: str,
    repo_id: str,
) -> dict[str, list[Any]]:
    """Pack observation data into PointAct batch format.

    The batch format uses lists for batching (even for single observations).

    Args:
        state: EE state of shape (7,) with [x, y, z, ax1, ax2, ax3, gripper]
        image: RGB image of shape (256, 256, 3)
        point_cloud: Point cloud of shape (N, 6) with [x, y, z, r, g, b]
        task: Task instruction string
        repo_id: Model repository ID

    Returns:
        Batch dict ready for msgpack serialization
    """
    batch = {
        "observation.images.front_image": [image],
        "observation.points": [point_cloud],
        "observation.state": [state],
        "task": [task],
        "repo_id": [repo_id],
    }
    return batch


def serialize_batch(batch: dict[str, list[Any]]) -> bytes:
    """Serialize batch dict with msgpack.

    Args:
        batch: Batch dict from pack_pointact_batch

    Returns:
        Serialized bytes
    """
    import msgpack
    import msgpack_numpy

    msgpack_numpy.patch()
    return msgpack.packb(batch)


def deserialize_response(data: bytes) -> dict[str, Any]:
    """Deserialize msgpack response from server.

    Args:
        data: Serialized response bytes

    Returns:
        Deserialized response dict
    """
    import msgpack
    import msgpack_numpy

    msgpack_numpy.patch()
    return msgpack.unpackb(data, raw=False)


def ee_action_to_transform(ee_action: np.ndarray) -> np.ndarray:
    """Convert EE action to 4x4 transformation matrix.

    Args:
        ee_action: Array of shape (7,) with [x, y, z, rx, ry, rz, gripper]

    Returns:
        4x4 transformation matrix
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = ee_action[:3]
    T[:3, :3] = Rotation.from_rotvec(ee_action[3:6]).as_matrix()
    return T
