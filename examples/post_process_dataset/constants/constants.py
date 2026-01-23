from pathlib import Path

# Path to this constants directory
CONSTANTS_DIR = Path(__file__).parent

# Camera calibration file paths
INTRINSICS_FILE = CONSTANTS_DIR / "intrinsics.npz"
EXTRINSICS_FILE = CONSTANTS_DIR / "extrinsics.npz"

# Workspace bounds (in meters, world coordinates)
WORKSPACE = {
    'X_BBOX': [-0.21, 0.23],   # Forward-backward
    'Y_BBOX': [-0.35, 0.3],   # Left-right
    'Z_BBOX': [0.0, 0.4],    # Up-down (height from table)
}