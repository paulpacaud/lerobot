import numpy as np
import cv2

def sanity_check_pose(
    img,
    objp,
    imgp,
    rvec,
    tvec,
    K,
    dist,
    axis_length=0.10,
    output_path="examples/camera_calibration/sanity_check.jpg"
):
    """
    Sanity checks for camera extrinsics estimated with solvePnP.

    img  : BGR image
    objp : (N,3) world points
    imgp : (N,2) image points
    rvec : (3,1) rotation vector (world -> camera)
    tvec : (3,1) translation vector (world -> camera)
    K    : (3,3) intrinsic matrix
    dist : distortion coefficients
    """

    print("\n=== SANITY CHECK ===")

    # ------------------------------------------------------------
    # 1) Rotation matrix validity
    # ------------------------------------------------------------
    R, _ = cv2.Rodrigues(rvec)

    ortho_error = np.linalg.norm(R.T @ R - np.eye(3))
    det_R = np.linalg.det(R)

    print("\n[Rotation matrix]")
    print("Orthonormality error ||RᵀR − I|| =", ortho_error)
    print("det(R) =", det_R)

    if ortho_error > 1e-3 or abs(det_R - 1.0) > 1e-3:
        print("WARNING: rotation matrix is not valid")

    # ------------------------------------------------------------
    # 2) Camera position in world frame
    # ------------------------------------------------------------
    R_world_cam = R.T
    t_world_cam = -R_world_cam @ tvec

    pos = t_world_cam.flatten()
    dist_to_origin = np.linalg.norm(pos)

    print("\n[Camera position in world frame]")
    print("pos (m) =", pos.tolist())
    print("distance to board origin (m) =", dist_to_origin)

    # ------------------------------------------------------------
    # 3) Reprojection error (most important)
    # ------------------------------------------------------------
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)

    reproj_err = np.linalg.norm(proj - imgp, axis=1)

    print("\n[Reprojection error]")
    print("mean error (px) =", reproj_err.mean())
    print("max  error (px) =", reproj_err.max())

    if reproj_err.mean() > 2.0:
        print("WARNING: high reprojection error")

    # ------------------------------------------------------------
    # 4) Visual axis check
    # ------------------------------------------------------------
    axis = np.float32([
        [0.0, 0.0, 0.0],
        [axis_length, 0.0, 0.0],   # X (red)
        [0.0, axis_length, 0.0],   # Y (green)
        [0.0, 0.0, axis_length],  # Z (blue)
    ])

    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
    imgpts = imgpts.reshape(-1, 2).astype(int)

    vis = img.copy()
    o = tuple(imgpts[0])

    cv2.line(vis, o, tuple(imgpts[1]), (0, 0, 255), 3)   # X
    cv2.line(vis, o, tuple(imgpts[2]), (0, 255, 0), 3)   # Y
    cv2.line(vis, o, tuple(imgpts[3]), (255, 0, 0), 3)   # Z

    cv2.putText(vis, "X", tuple(imgpts[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.putText(vis, "Y", tuple(imgpts[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(vis, "Z", tuple(imgpts[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    print("\n[Visual check]")
    print("Red   = X axis along board")
    print("Green = Y axis along board")
    print("Blue  = Z axis pointing out of board toward camera")

    cv2.imwrite(output_path, vis)
    print(f"Visualization saved to {output_path}")

    print("\n=== END SANITY CHECK ===\n")


# ===============================
# Load intrinsics
# ===============================
intrinsics_path = "examples/camera_calibration/intrinsics.npz"
intrinsics_data = np.load(intrinsics_path)
K = intrinsics_data['K']
dist = intrinsics_data['dist']
print(f"Loaded intrinsics from {intrinsics_path}")

# ===============================
# Load extrinsics and calibration data
# ===============================
extrinsics_path = "examples/camera_calibration/extrinsics.npz"
extrinsics_data = np.load(extrinsics_path, allow_pickle=True)
rvec = extrinsics_data['rvec']
tvec = extrinsics_data['tvec']
objp = extrinsics_data['objp']
imgp = extrinsics_data['imgp']
image_path = str(extrinsics_data['image_path'])
print(f"Loaded extrinsics from {extrinsics_path}")

# ===============================
# Load image
# ===============================
img = cv2.imread(image_path)
if img is None:
    raise RuntimeError(f"Failed to load image from {image_path}")
print(f"Loaded image from {image_path}")

# ===============================
# Run sanity check
# ===============================
sanity_check_pose(img, objp, imgp, rvec, tvec, K, dist)