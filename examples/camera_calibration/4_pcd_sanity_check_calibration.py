#!/usr/bin/env python
"""
Point Cloud Sanity Check for Camera Calibration.

This script:
1. Captures an RGB-D image from the RealSense camera
2. Creates a 3D point cloud using intrinsics
3. Transforms to world frame using extrinsics
4. Displays the point cloud with world coordinate axes at origin (0,0,0)

This helps verify that the camera calibration (intrinsics + extrinsics) is correct.
The world origin should appear at the chessboard corner where calibration was done.

Usage:
    python examples/camera_calibration/5_pcd_sanity_check_calibration.py
"""

import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except ImportError:
    raise ImportError("pyrealsense2 is required. Install with: pip install pyrealsense2")

try:
    import open3d as o3d
except ImportError:
    raise ImportError("open3d is required. Install with: pip install open3d")


def capture_rgbd_frame(width=640, height=480, fps=30):
    """Capture a single RGB-D frame from RealSense camera."""
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    # Start pipeline
    profile = pipeline.start(config)

    # Get depth scale (to convert depth units to meters)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale: {depth_scale} (depth_value * scale = meters)")

    # Get factory intrinsics for both streams
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()

    color_intrinsics = color_profile.get_intrinsics()
    depth_intrinsics = depth_profile.get_intrinsics()

    print(f"\nRealSense FACTORY intrinsics (color camera):")
    print(f"  fx={color_intrinsics.fx:.2f}, fy={color_intrinsics.fy:.2f}")
    print(f"  cx={color_intrinsics.ppx:.2f}, cy={color_intrinsics.ppy:.2f}")
    print(f"  distortion model: {color_intrinsics.model}")

    print(f"\nRealSense FACTORY intrinsics (depth camera):")
    print(f"  fx={depth_intrinsics.fx:.2f}, fy={depth_intrinsics.fy:.2f}")
    print(f"  cx={depth_intrinsics.ppx:.2f}, cy={depth_intrinsics.ppy:.2f}")

    # Get depth-to-color extrinsics (transformation between sensors)
    depth_to_color_extrinsics = depth_profile.get_extrinsics_to(color_profile)
    print(f"\nRealSense depth-to-color extrinsics:")
    print(f"  Translation: [{depth_to_color_extrinsics.translation[0]:.4f}, "
          f"{depth_to_color_extrinsics.translation[1]:.4f}, "
          f"{depth_to_color_extrinsics.translation[2]:.4f}] m")

    # Align depth to color
    align = rs.align(rs.stream.color)

    # Wait for auto-exposure to stabilize
    print("\nWaiting for camera to stabilize...")
    for _ in range(30):
        pipeline.wait_for_frames()

    # Capture frame
    print("Capturing frame...")
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame or not depth_frame:
        pipeline.stop()
        raise RuntimeError("Failed to capture frames")

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Store factory intrinsics for comparison
    factory_intrinsics = {
        'fx': color_intrinsics.fx,
        'fy': color_intrinsics.fy,
        'cx': color_intrinsics.ppx,
        'cy': color_intrinsics.ppy
    }

    pipeline.stop()

    return color_image, depth_image, depth_scale, factory_intrinsics


def load_calibration():
    """Load intrinsics and extrinsics from calibration files."""
    intrinsics_path = "examples/camera_calibration/intrinsics.npz"
    extrinsics_path = "examples/camera_calibration/extrinsics.npz"

    intrinsics_data = np.load(intrinsics_path)
    extrinsics_data = np.load(extrinsics_path)

    K = intrinsics_data['K']
    dist = intrinsics_data['dist']
    rvec = extrinsics_data['rvec']
    tvec = extrinsics_data['tvec']

    print(f"Loaded intrinsics from {intrinsics_path}")
    print(f"Loaded extrinsics from {extrinsics_path}")

    return K, dist, rvec, tvec


def depth_to_pointcloud(depth_image, color_image, K, depth_scale,
                        min_depth=0.1, max_depth=3.0, downsample=1):
    """
    Convert depth image to colored point cloud in camera frame.

    Args:
        depth_image: (H, W) uint16 depth image
        color_image: (H, W, 3) BGR color image
        K: 3x3 intrinsic matrix
        depth_scale: Scale to convert depth to meters
        min_depth: Minimum valid depth in meters
        max_depth: Maximum valid depth in meters
        downsample: Downsample factor

    Returns:
        points: (N, 3) points in camera frame (meters)
        colors: (N, 3) RGB colors normalized to [0, 1]
    """
    height, width = depth_image.shape

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Create pixel grid (coordinates are in original image space)
    u = np.arange(0, width, downsample)
    v = np.arange(0, height, downsample)
    u, v = np.meshgrid(u, v)

    # Get depth in meters
    z = depth_image[::downsample, ::downsample].astype(np.float32) * depth_scale

    # Valid depth mask
    valid = (z > min_depth) & (z < max_depth)

    # NOTE: Do NOT scale intrinsics - pixel coordinates are still in original image space
    # We're just sampling fewer pixels, not resizing the image

    # Deproject to 3D
    z_valid = z[valid]
    u_valid = u[valid]
    v_valid = v[valid]

    x = (u_valid - cx) * z_valid / fx
    y = (v_valid - cy) * z_valid / fy

    points = np.stack([x, y, z_valid], axis=-1)

    # Get colors (BGR to RGB, normalize)
    color_downsampled = color_image[::downsample, ::downsample]
    colors = color_downsampled[valid][:, ::-1].astype(np.float32) / 255.0

    return points, colors


def transform_to_world(points, rvec, tvec):
    """
    Transform points from camera frame to world frame.

    Args:
        points: (N, 3) points in camera frame
        rvec: Rotation vector (world to camera)
        tvec: Translation vector (world to camera)

    Returns:
        points_world: (N, 3) points in world frame
    """
    # World-to-camera rotation matrix
    R_w2c, _ = cv2.Rodrigues(rvec)
    t_w2c = tvec.flatten()

    # Invert to get camera-to-world
    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ t_w2c

    # Transform: P_world = R_c2w @ P_cam + t_c2w
    points_world = (R_c2w @ points.T).T + t_c2w

    return points_world


def create_coordinate_axes(origin=(0, 0, 0), size=0.1):
    """
    Create coordinate axes visualization at specified origin.

    Args:
        origin: (x, y, z) position of the origin
        size: Length of each axis in meters

    Returns:
        List of Open3D geometries (lines and labels)
    """
    origin = np.array(origin)

    # Axis endpoints
    x_end = origin + np.array([size, 0, 0])
    y_end = origin + np.array([0, size, 0])
    z_end = origin + np.array([0, 0, size])

    # Create line set for axes
    points = [origin, x_end, y_end, z_end]
    lines = [[0, 1], [0, 2], [0, 3]]  # Origin to X, Y, Z
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # R, G, B for X, Y, Z

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Create spheres at axis tips for visibility
    geometries = [line_set]

    # Small spheres at axis ends
    sphere_radius = size * 0.05

    x_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    x_sphere.translate(x_end)
    x_sphere.paint_uniform_color([1, 0, 0])

    y_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    y_sphere.translate(y_end)
    y_sphere.paint_uniform_color([0, 1, 0])

    z_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    z_sphere.translate(z_end)
    z_sphere.paint_uniform_color([0, 0, 1])

    # Origin sphere (white)
    origin_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius * 1.5)
    origin_sphere.translate(origin)
    origin_sphere.paint_uniform_color([1, 1, 1])

    geometries.extend([x_sphere, y_sphere, z_sphere, origin_sphere])

    return geometries


def load_calibration_with_corners():
    """Load calibration including chessboard corners for verification."""
    intrinsics_path = "examples/camera_calibration/intrinsics.npz"
    extrinsics_path = "examples/camera_calibration/extrinsics.npz"

    intrinsics_data = np.load(intrinsics_path)
    extrinsics_data = np.load(extrinsics_path)

    K = intrinsics_data['K']
    dist = intrinsics_data['dist']
    rvec = extrinsics_data['rvec']
    tvec = extrinsics_data['tvec']
    objp = extrinsics_data['objp']  # World points (first one is origin)
    imgp = extrinsics_data['imgp']  # Image points

    return K, dist, rvec, tvec, objp, imgp


def main():
    print("=" * 60)
    print("POINT CLOUD CALIBRATION SANITY CHECK")
    print("=" * 60)

    # Load calibration with corner data
    K, dist, rvec, tvec, objp, imgp = load_calibration_with_corners()

    print(f"\nIntrinsics:")
    print(f"  fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
    print(f"  cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")

    # Compute camera position in world frame
    R_w2c, _ = cv2.Rodrigues(rvec)
    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ tvec.flatten()

    print(f"\nCamera position in world frame:")
    print(f"  X={t_c2w[0]:.3f} m, Y={t_c2w[1]:.3f} m, Z={t_c2w[2]:.3f} m")

    # The first chessboard corner should be at world origin
    origin_world = objp[0]  # Should be [0, 0, 0]
    origin_img = imgp[0]    # Pixel location of origin
    print(f"\nChessboard origin (for verification):")
    print(f"  World coords: {origin_world}")
    print(f"  Image coords: ({origin_img[0]:.1f}, {origin_img[1]:.1f})")

    # Capture RGB-D frame
    print("\n" + "-" * 40)
    color_image, depth_image, depth_scale, factory_intrinsics = capture_rgbd_frame()
    print(f"\nCaptured frame: {color_image.shape[:2]}")
    print(f"Depth range: {depth_image.min()} - {depth_image.max()} (raw values)")
    print(f"Depth range: {depth_image.min() * depth_scale:.3f} - {depth_image.max() * depth_scale:.3f} m")

    # Compare calibrated vs factory intrinsics
    print("\n" + "-" * 40)
    print("INTRINSICS COMPARISON:")
    print(f"  {'Parameter':<10} {'Calibrated':>12} {'Factory':>12} {'Diff':>10}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'fx':<10} {K[0,0]:>12.2f} {factory_intrinsics['fx']:>12.2f} {K[0,0] - factory_intrinsics['fx']:>10.2f}")
    print(f"  {'fy':<10} {K[1,1]:>12.2f} {factory_intrinsics['fy']:>12.2f} {K[1,1] - factory_intrinsics['fy']:>10.2f}")
    print(f"  {'cx':<10} {K[0,2]:>12.2f} {factory_intrinsics['cx']:>12.2f} {K[0,2] - factory_intrinsics['cx']:>10.2f}")
    print(f"  {'cy':<10} {K[1,2]:>12.2f} {factory_intrinsics['cy']:>12.2f} {K[1,2] - factory_intrinsics['cy']:>10.2f}")

    intrinsics_match = (
        abs(K[0,0] - factory_intrinsics['fx']) < 5 and
        abs(K[1,1] - factory_intrinsics['fy']) < 5 and
        abs(K[0,2] - factory_intrinsics['cx']) < 5 and
        abs(K[1,2] - factory_intrinsics['cy']) < 5
    )
    if intrinsics_match:
        print("\n  ✓ Intrinsics are similar (within 5 pixels)")
    else:
        print("\n  ⚠ WARNING: Significant difference between calibrated and factory intrinsics!")
        print("    This could cause depth alignment issues.")

    # Draw world frame on the captured image (like 2D sanity check)
    axis_length = 0.10  # 10cm axes
    axis_3d = np.float32([
        [0.0, 0.0, 0.0],           # Origin
        [axis_length, 0.0, 0.0],   # X (red)
        [0.0, axis_length, 0.0],   # Y (green)
        [0.0, 0.0, axis_length],   # Z (blue)
    ])

    imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, K, dist)
    imgpts = imgpts.reshape(-1, 2).astype(int)

    vis_image = color_image.copy()
    origin_pt = tuple(imgpts[0])

    cv2.line(vis_image, origin_pt, tuple(imgpts[1]), (0, 0, 255), 3)   # X - Red
    cv2.line(vis_image, origin_pt, tuple(imgpts[2]), (0, 255, 0), 3)   # Y - Green
    cv2.line(vis_image, origin_pt, tuple(imgpts[3]), (255, 0, 0), 3)   # Z - Blue

    cv2.putText(vis_image, "X", tuple(imgpts[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(vis_image, "Y", tuple(imgpts[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis_image, "Z", tuple(imgpts[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Save image with world frame overlay
    output_image_path = "examples/camera_calibration/pcd_sanity_check.jpg"
    cv2.imwrite(output_image_path, vis_image)
    print(f"\nSaved captured image with world frame to {output_image_path}")

    # DIAGNOSTIC: Check depth at chessboard origin location
    print("\n" + "-" * 40)
    print("DIAGNOSTIC: Verifying transform at chessboard origin")
    u_origin, v_origin = int(origin_img[0]), int(origin_img[1])
    print(f"  Chessboard origin pixel: ({u_origin}, {v_origin})")

    if 0 <= u_origin < depth_image.shape[1] and 0 <= v_origin < depth_image.shape[0]:
        # Get depth at origin (average small window for robustness)
        window = 5
        v_min, v_max = max(0, v_origin - window), min(depth_image.shape[0], v_origin + window)
        u_min, u_max = max(0, u_origin - window), min(depth_image.shape[1], u_origin + window)
        depth_window = depth_image[v_min:v_max, u_min:u_max]
        valid_depths = depth_window[depth_window > 0]

        if len(valid_depths) > 0:
            z_meters = np.median(valid_depths) * depth_scale
            print(f"  Depth at origin: {z_meters:.4f} m ({z_meters * 1000:.1f} mm)")

            # Test with CALIBRATED intrinsics
            print(f"\n  Using CALIBRATED intrinsics:")
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            x_cam = (u_origin - cx) * z_meters / fx
            y_cam = (v_origin - cy) * z_meters / fy
            z_cam = z_meters
            p_cam = np.array([x_cam, y_cam, z_cam])
            p_world = R_c2w @ p_cam + t_c2w
            print(f"    Camera frame: [{p_cam[0]:.4f}, {p_cam[1]:.4f}, {p_cam[2]:.4f}] m")
            print(f"    World frame:  [{p_world[0]:.4f}, {p_world[1]:.4f}, {p_world[2]:.4f}] m")
            error_calib = np.linalg.norm(p_world)
            print(f"    Error (distance from origin): {error_calib * 1000:.1f} mm")

            # Test with FACTORY intrinsics
            print(f"\n  Using FACTORY intrinsics:")
            fx_f, fy_f = factory_intrinsics['fx'], factory_intrinsics['fy']
            cx_f, cy_f = factory_intrinsics['cx'], factory_intrinsics['cy']
            x_cam_f = (u_origin - cx_f) * z_meters / fx_f
            y_cam_f = (v_origin - cy_f) * z_meters / fy_f
            p_cam_f = np.array([x_cam_f, y_cam_f, z_cam])
            p_world_f = R_c2w @ p_cam_f + t_c2w
            print(f"    Camera frame: [{p_cam_f[0]:.4f}, {p_cam_f[1]:.4f}, {p_cam_f[2]:.4f}] m")
            print(f"    World frame:  [{p_world_f[0]:.4f}, {p_world_f[1]:.4f}, {p_world_f[2]:.4f}] m")
            error_factory = np.linalg.norm(p_world_f)
            print(f"    Error (distance from origin): {error_factory * 1000:.1f} mm")

            # Recommend which intrinsics to use
            print(f"\n  Expected world coords: [0.0000, 0.0000, 0.0000] m")
            if error_calib < error_factory:
                print(f"  → Calibrated intrinsics give better results")
            else:
                print(f"  → Factory intrinsics give better results")

            # Also compute what depth WOULD give us (0,0,0) in world
            # P_world = R_c2w @ P_cam + t_c2w = 0
            # P_cam = R_w2c @ (-t_c2w)
            p_cam_expected = R_w2c @ (-t_c2w)
            z_expected = p_cam_expected[2]
            depth_error = z_meters - z_expected

            print(f"\n  Expected depth at origin for world (0,0,0): {z_expected:.4f} m ({z_expected * 1000:.1f} mm)")
            print(f"  Actual depth measured:                       {z_meters:.4f} m ({z_meters * 1000:.1f} mm)")
            print(f"  Depth difference:                            {depth_error * 1000:.1f} mm")
        else:
            print(f"  WARNING: No valid depth at chessboard origin!")
    else:
        print(f"  WARNING: Origin pixel outside image bounds!")

    # Create point cloud in camera frame
    print("\n" + "-" * 40)
    print("Creating point cloud...")
    points_cam, colors = depth_to_pointcloud(
        depth_image, color_image, K, depth_scale,
        min_depth=0.1, max_depth=2.0, downsample=2
    )
    print(f"Generated {len(points_cam):,} points")

    # Debug: print camera frame bounds
    print(f"\nPoint cloud bounds (CAMERA frame, before transform):")
    print(f"  X: [{points_cam[:, 0].min():.3f}, {points_cam[:, 0].max():.3f}] m")
    print(f"  Y: [{points_cam[:, 1].min():.3f}, {points_cam[:, 1].max():.3f}] m")
    print(f"  Z: [{points_cam[:, 2].min():.3f}, {points_cam[:, 2].max():.3f}] m")

    # Transform to world frame
    print(f"\nTransforming to world frame...")
    print(f"  R_c2w = ")
    print(f"    [{R_c2w[0,0]:7.4f}, {R_c2w[0,1]:7.4f}, {R_c2w[0,2]:7.4f}]")
    print(f"    [{R_c2w[1,0]:7.4f}, {R_c2w[1,1]:7.4f}, {R_c2w[1,2]:7.4f}]")
    print(f"    [{R_c2w[2,0]:7.4f}, {R_c2w[2,1]:7.4f}, {R_c2w[2,2]:7.4f}]")
    print(f"  t_c2w = [{t_c2w[0]:.4f}, {t_c2w[1]:.4f}, {t_c2w[2]:.4f}]")

    points_world = transform_to_world(points_cam, rvec, tvec)

    # Verify single point transform
    print(f"\nVerification - transform a test point:")
    test_idx = len(points_cam) // 2  # middle point
    p_cam_test = points_cam[test_idx]
    p_world_test = R_c2w @ p_cam_test + t_c2w
    print(f"  Camera frame: [{p_cam_test[0]:.4f}, {p_cam_test[1]:.4f}, {p_cam_test[2]:.4f}]")
    print(f"  World frame:  [{p_world_test[0]:.4f}, {p_world_test[1]:.4f}, {p_world_test[2]:.4f}]")
    print(f"  From batch:   [{points_world[test_idx, 0]:.4f}, {points_world[test_idx, 1]:.4f}, {points_world[test_idx, 2]:.4f}]")

    # Print point cloud statistics in world frame
    print(f"\nPoint cloud bounds (world frame):")
    print(f"  X: [{points_world[:, 0].min():.3f}, {points_world[:, 0].max():.3f}] m")
    print(f"  Y: [{points_world[:, 1].min():.3f}, {points_world[:, 1].max():.3f}] m")
    print(f"  Z: [{points_world[:, 2].min():.3f}, {points_world[:, 2].max():.3f}] m")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create world coordinate frame at origin
    # Size 10cm to be visible
    world_axes = create_coordinate_axes(origin=(0, 0, 0), size=0.1)

    # Also add a larger coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)

    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)
    print("\nWorld coordinate frame at origin (0, 0, 0):")
    print("  RED   = X axis")
    print("  GREEN = Y axis")
    print("  BLUE  = Z axis (should point UP from table)")
    print("\nThe origin should be at the chessboard corner used for calibration.")
    print("Table surface should be at approximately Z = 0.")
    print("\nControls:")
    print("  Mouse drag: Rotate")
    print("  Scroll: Zoom")
    print("  Shift+drag: Pan")
    print("  'R': Reset view")
    print("  'Q': Quit")

    # Visualize
    geometries = [pcd, coord_frame] + world_axes
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Point Cloud Calibration Check - World Frame",
        width=1280,
        height=720
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
