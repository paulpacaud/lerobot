# Data Collection

## put_cube_in_spot

lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=so100_leader \
   --robot.cameras="{ front: {type: intelrealsense, serial_number_or_name: 147122078460, width: 640, height: 480, fps: 30, use_depth: true}}" \
    --display_data=true \
    --dataset.root=$HOME/lerobot_datasets/put_cube_in_spot \
    --dataset.push_to_hub=False \
    --dataset.repo_id=${HF_USER}/put_cube_in_spot \
    --dataset.num_episodes=25 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="put the cube in the green square spot" \
    --resume=true