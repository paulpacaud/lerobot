POINTACT_FEATURES = {
    # RGB image (resized)
    "observation.images.front_image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    # Depth image
    "observation.images.front_depth": {
        "dtype": "image",
        "shape": (480, 640, 1),
        "names": ["height", "width", "channels"],
        "is_depth_map": True,
    },
    # Point cloud
    "observation.points.frontview": {
        "dtype": "point_cloud",
        "shape": (None, 6),
        "names": ["x", "y", "z", "r", "g", "b"],
        "storage": "lmdb",
        "path": "point_clouds",
        "voxel_size": 0.01,
        "workspace": {"X_BBOX": [-0.23, 0.23], "Y_BBOX": [-0.35, 0.3], "Z_BBOX": [0.0, 0.4]},
    },
    # State (EE pose + gripper) - actual measured/observed state at time t (what the robot is doing)
    "observation.state": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper_openness"]},
    },
    # EE state (position + orientation)
    "observation.states.ee_state": {
        "dtype": "float32",
        "shape": (6,),
        "names": {"motors": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3"]},
    },
    # Joint state (all motors including gripper)
    "observation.states.joint_state": {
        "dtype": "float32",
        "shape": (6,),
        "names": {"motors": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]},
    },
    # Gripper state
    "observation.states.gripper_state": {
        "dtype": "float32",
        "shape": (1,),
        "names": {"motors": ["gripper_openness"]},
    },
    # Action - commanded/target action at time t (what the robot should do)
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper_openness"]},
    },
    # Metadata
    "timestamp": {
        "dtype": "float32",
        "shape": (1,),
    },
    "frame_index": {
        "dtype": "int64",
        "shape": (1,),
    },
    "episode_index": {
        "dtype": "int64",
        "shape": (1,),
    },
    "index": {
        "dtype": "int64",
        "shape": (1,),
    },
    "task_index": {
        "dtype": "int64",
        "shape": (1,),
    },
}