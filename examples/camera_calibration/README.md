python examples/camera_calibration/1_capture_image.py
python examples/camera_calibration/2_find_intrinsics.py
python examples/camera_calibration/3_find_extrinsics.py
python examples/camera_calibration/4_pcd_sanity_check_calibration.py

Then, put the robot gripper to the world frame and read is position in forward kinematics:
python examples/post_process_dataset/read_ee_position.py --robot_port=/dev/ttyACM0

you get the translation offset to apply to go from robot frame to world frame
e.g.
EEF robot frame x=0.2755 y=0.0599 z=-0.0257

offset to apply:
x=-0.2755 y=-0.0599 z=+0.0257