from pathlib import Path

# Path to this constants directory
CONSTANTS_DIR = Path(__file__).parent

# Camera calibration file paths
INTRINSICS_FILE = CONSTANTS_DIR / "intrinsics.npz"
EXTRINSICS_FILE = CONSTANTS_DIR / "extrinsics.npz"

TABLE_HEIGHT=0.018

# Workspace bounds  (in meters, world coordinates), we crop outside
WORKSPACE = {
    'X_BBOX': [-0.21, 0.23],   # Forward-backward
    'Y_BBOX': [-0.35, 0.3],   # Left-right
    'Z_BBOX': [TABLE_HEIGHT, 0.4],    # Up-down (height from table), we crop the table
}

# Robot frame position in world coordinates (meters)
# Translation offset from robot base to world frame origin
ROBOT_FRAME = {
    'tx': -0.2755,
    'ty': -0.0599,
    'tz': 0.0257,
}