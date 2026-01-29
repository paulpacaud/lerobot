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

"""Configuration for the PointAct robot client."""

from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig

from .constants import DEFAULT_FPS


# Default workspace bounds (in meters, world coordinates)
DEFAULT_WORKSPACE = {
    "X_BBOX": [-0.21, 0.23],
    "Y_BBOX": [-0.35, 0.3],
    "Z_BBOX": [0.025, 0.4],
}

# Default translation offset from robot base to world frame origin
DEFAULT_TRANSLATION_OFFSET = [-0.2755, -0.0599, 0.0257]

# Default joint names for SO100/SO101 robots
DEFAULT_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


@dataclass
class PointActClientConfig:
    """Configuration for PointAct robot client.

    This client preprocesses observations to PointAct format and sends them
    via HTTP POST with msgpack serialization to a PointAct policy server.
    """

    # Robot configuration (required)
    robot: RobotConfig = field(metadata={"help": "Robot configuration"})

    # URDF path (required)
    urdf_path: str = field(metadata={"help": "Path to URDF file for FK/IK"})

    # Network configuration
    server_address: str = field(
        default="localhost:8080",
        metadata={"help": "Policy server address (host:port)"},
    )
    target_frame: str = field(
        default="gripper_frame_link",
        metadata={"help": "End-effector frame name in URDF"},
    )
    translation_offset: list[float] = field(
        default_factory=lambda: DEFAULT_TRANSLATION_OFFSET.copy(),
        metadata={"help": "Translation offset [tx, ty, tz] from robot base to world frame"},
    )
    joint_names: list[str] = field(
        default_factory=lambda: DEFAULT_JOINT_NAMES.copy(),
        metadata={"help": "Joint names for FK (excluding gripper)"},
    )

    # Image configuration
    image_size: int = field(
        default=256,
        metadata={"help": "Target image size (square)"},
    )
    source_image_key: str = field(
        default="front",
        metadata={"help": "Camera key for RGB image"},
    )

    # Point cloud configuration
    intrinsics_file: str = field(
        default="",
        metadata={"help": "Path to camera intrinsics .npz file"},
    )
    extrinsics_file: str = field(
        default="",
        metadata={"help": "Path to camera extrinsics .npz file"},
    )
    voxel_size: float = field(
        default=0.01,
        metadata={"help": "Voxel size for point cloud downsampling (meters)"},
    )
    depth_scale: float = field(
        default=1000.0,
        metadata={"help": "Scale factor to convert depth values to meters (1000 for mm->m)"},
    )
    workspace: dict = field(
        default_factory=lambda: DEFAULT_WORKSPACE.copy(),
        metadata={"help": "Workspace bounds for point cloud cropping"},
    )

    # Model configuration
    repo_id: str = field(
        default="",
        metadata={"help": "Model repository ID for PointAct policy"},
    )

    # Task configuration
    task: str = field(
        default="",
        metadata={"help": "Task instruction for the robot to execute"},
    )

    # Control configuration
    fps: int = field(
        default=DEFAULT_FPS,
        metadata={"help": "Control frequency (frames per second)"},
    )

    @property
    def environment_dt(self) -> float:
        """Environment time step, in seconds."""
        return 1 / self.fps

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.server_address:
            raise ValueError("server_address cannot be empty")

        if not self.urdf_path:
            raise ValueError("urdf_path is required")

        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")

        if self.voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive, got {self.voxel_size}")

        if len(self.translation_offset) != 3:
            raise ValueError(
                f"translation_offset must have 3 elements, got {len(self.translation_offset)}"
            )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PointActClientConfig":
        """Create a PointActClientConfig from a dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {
            "server_address": self.server_address,
            "urdf_path": self.urdf_path,
            "target_frame": self.target_frame,
            "translation_offset": self.translation_offset,
            "joint_names": self.joint_names,
            "image_size": self.image_size,
            "source_image_key": self.source_image_key,
            "intrinsics_file": self.intrinsics_file,
            "extrinsics_file": self.extrinsics_file,
            "voxel_size": self.voxel_size,
            "depth_scale": self.depth_scale,
            "workspace": self.workspace,
            "repo_id": self.repo_id,
            "task": self.task,
            "fps": self.fps,
        }
