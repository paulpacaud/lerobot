import pyrealsense2 as rs
import numpy as np

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

color_stream = profile.get_stream(rs.stream.color)
vsp = color_stream.as_video_stream_profile()
intr = vsp.get_intrinsics()

print("fx, fy:", intr.fx, intr.fy)
print("cx, cy:", intr.ppx, intr.ppy)
print("distortion model:", intr.model)
print("coeffs:", intr.coeffs)

K = np.array([[intr.fx, 0,       intr.ppx],
              [0,       intr.fy, intr.ppy],
              [0,       0,       1      ]], dtype=np.float64)

print(f"K =")
print(K)
pipeline.stop()

# Save intrinsics
output_path = "examples/camera_calibration/intrinsics.npz"
np.savez(output_path, K=K, dist=np.array(intr.coeffs, dtype=np.float64))
print(f"\nIntrinsics saved to {output_path}")