import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rotation

# ===============================
# Load Intrinsics
# ===============================
intrinsics_path = "examples/camera_calibration/intrinsics.npz"
intrinsics_data = np.load(intrinsics_path)
K = intrinsics_data['K']
dist = intrinsics_data['dist']
print(f"Loaded intrinsics from {intrinsics_path}")

# ===============================
# Chessboard definition
# ===============================
CHECKERBOARD = (6, 9)      # (Nx, Ny) inner corners
SQUARE_SIZE = 0.024        # meters

# World (object) points in chessboard frame
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), dtype=np.float64)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE  # scale to meters
objp[:, 1] *= -1.0  # flip world Y to make world Z "up"

# ===============================
# Load image
# ===============================
img = cv2.imread("examples/camera_calibration/chessboard.jpg")  # <-- replace with your image path
if img is None:
    raise RuntimeError("Failed to load image")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ===============================
# Detect chessboard
# ===============================
flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)

if not ret:
    raise RuntimeError("Chessboard not detected")

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    1e-3
)
corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
imgp = corners.reshape(-1, 2).astype(np.float64)

# ===============================
# PnP: world(board) -> camera
# ===============================
ok, rvec, tvec = cv2.solvePnP(
    objp,
    imgp,
    K,
    dist,
    flags=cv2.SOLVEPNP_ITERATIVE
)

if not ok:
    raise RuntimeError("solvePnP failed")

R_cam_world, _ = cv2.Rodrigues(rvec)
t_cam_world = tvec.reshape(3, 1)

# T_cam_world
T_cam_world = np.eye(4)
T_cam_world[:3, :3] = R_cam_world
T_cam_world[:3, 3:] = t_cam_world

# ===============================
# Invert to get world_T_cam
# ===============================
R_world_cam = R_cam_world.T
t_world_cam = -R_world_cam @ t_cam_world

T_world_cam = np.eye(4)
T_world_cam[:3, :3] = R_world_cam
T_world_cam[:3, 3:] = t_world_cam

# ===============================
# Extract position and Euler angles
# ===============================
pos = T_world_cam[:3, 3].tolist()

rot = Rotation.from_matrix(T_world_cam[:3, :3])
euler = rot.as_euler("xyz", degrees=False).tolist()  # set degrees=True if needed

# ===============================
# Output
# ===============================
output = {
    "extrinsics": {
        "world_T_cam": T_world_cam.tolist()
    },
    "info_cam": {
        "pos": pos,
        "euler": euler
    }
}

print(output)

# Save extrinsics and data for sanity check
extrinsics_path = "examples/camera_calibration/extrinsics.npz"
image_path = "examples/camera_calibration/chessboard.jpg"
np.savez(
    extrinsics_path,
    rvec=rvec,
    tvec=tvec,
    objp=objp,
    imgp=imgp,
    image_path=image_path
)
print(f"\nExtrinsics and calibration data saved to {extrinsics_path}")