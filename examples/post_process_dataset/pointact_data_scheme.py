POINTACT_FEATURES = {
    # RGB image (resized)
    "observation.images.front_image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    # Depth image (resized)
    "observation.images.front_depth": {
        "dtype": "image",
        "shape": (256, 256, 1),
        "names": ["height", "width", "channel"],
        "storage": "parquet",  # stored as struct<bytes, path> in parquet
        "optional": True,
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
        "shape": (8,),
        "names": {"motors": ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper_openness"]},
    },
    # EE state (position + orientation)
    "observation.states.ee_state": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": ["x", "y", "z", "qw", "qx", "qy", "qz"]},
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
    # Action (EE space) - commanded/target action at time t (what the robot should do)
    "action": {
        "dtype": "float32",
        "shape": (8,),
        "names": {"motors": ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper_openness"]},
    },
    # Action (joint space) - original joint commands for replay
    "action.joints": { # the joint poses in the _pointact datasets are absolute values (in degrees). 
        "dtype": "float32",
        "shape": (6,),
        "names": {"motors": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]},
    },
    # Metadata (stored as scalars in parquet)
    "timestamp": {
        "dtype": "float64",
        "shape": (),
    },
    "frame_index": {
        "dtype": "int64",
        "shape": (),
    },
    "episode_index": {
        "dtype": "int64",
        "shape": (),
    },
    "index": {
        "dtype": "int64",
        "shape": (),
    },
    "task_index": {
        "dtype": "int64",
        "shape": (),
    },
}