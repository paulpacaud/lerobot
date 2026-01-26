Download the dataset:
```bash
huggingface-cli download paulpacaud/depth_test_pointact \
  --repo-type dataset \
  --local-dir depth_test_pointact \
  --local-dir-use-symlinks False
```


Visualize the dataset:
It shows the point cloud at a given frame for a given episode, with the trajectory of the end-effector overlaid.
```bash
python ./visualize_pointact_dataset.py --dataset_dir=./depth_test_pointact --episode_index=0 --pcd_frame=147
```

Note:
The end-effector position is computed using Forward Kinematics (FK) using as input 1) the URDF of the arm, 2) the joint values recorded during teleoperation
The FK uses the theoretical URDF joint origins as the reference.
But in lerobot, we have to manually calibrate the robot arm joints with `lerobot-calibrate`. So the physical "middle" position during calibration doesn't exactly match the theoretical URDF's expected zero pose.
I thus observe in the point cloud a slight offset in the end-effector position computed.
What would fix it and give the very ground-truth end-effector pose would be to adjust the URDF joint origins to match our personal calibration, I asked the library maintainer if this can be done. Until then, we can use the current pipeline that gives reasonable results.