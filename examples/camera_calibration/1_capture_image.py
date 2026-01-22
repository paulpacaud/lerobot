import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

# Allow auto-exposure to settle
for _ in range(30):
    pipeline.wait_for_frames()

frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
if not color_frame:
    pipeline.stop()
    raise RuntimeError("No color frame received")

img = np.asanyarray(color_frame.get_data())
cv2.imwrite("examples/camera_calibration/chessboard.jpg", img)
pipeline.stop()
