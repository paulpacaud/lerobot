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

## put_socks_into_drawer

lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=so100_leader \
   --robot.cameras="{ front: {type: intelrealsense, serial_number_or_name: 147122078460, width: 640, height: 480, fps: 30, use_depth: true}}" \
    --display_data=true \
    --dataset.root=$HOME/lerobot_datasets/put_socks_into_drawer \
    --dataset.push_to_hub=False \
    --dataset.repo_id=${HF_USER}/put_socks_into_drawer \
    --dataset.num_episodes=25 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="put the socks into the middle drawer, then close the drawer" \
    --resume=true

## put_banana_and_toy_in_plates

lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=so100_leader \
   --robot.cameras="{ front: {type: intelrealsense, serial_number_or_name: 147122078460, width: 640, height: 480, fps: 30, use_depth: true}}" \
    --display_data=true \
    --dataset.root=$HOME/lerobot_datasets/put_banana_and_toy_in_plates \
    --dataset.push_to_hub=False \
    --dataset.repo_id=${HF_USER}/put_banana_and_toy_in_plates \
    --dataset.num_episodes=25 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="put the banana in the blue plate, then put the green toy in the pink plate" \
    --resume=true

## open_microwave

lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=so100_leader \
   --robot.cameras="{ front: {type: intelrealsense, serial_number_or_name: 147122078460, width: 640, height: 480, fps: 30, use_depth: true}}" \
    --display_data=true \
    --dataset.root=$HOME/lerobot_datasets/open_microwave \
    --dataset.push_to_hub=False \
    --dataset.repo_id=${HF_USER}/open_microwave \
    --dataset.num_episodes=50 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="open the microwave door to 90 degrees" \
    --resume=true

## stack_cups

lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=so100_leader \
   --robot.cameras="{ front: {type: intelrealsense, serial_number_or_name: 147122078460, width: 640, height: 480, fps: 30, use_depth: true}}" \
    --display_data=true \
    --dataset.root=$HOME/lerobot_datasets/stack_cups \
    --dataset.push_to_hub=False \
    --dataset.repo_id=${HF_USER}/stack_cups \
    --dataset.num_episodes=50 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="stack the yellow cup onto the blue cup, then stack the orange cup onto the yellow cup" \
    --resume=true

## move_plates_from_rack_to_box

lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=so100_leader \
   --robot.cameras="{ front: {type: intelrealsense, serial_number_or_name: 147122078460, width: 640, height: 480, fps: 30, use_depth: true}}" \
    --display_data=true \
    --dataset.root=$HOME/lerobot_datasets/move_plates_from_rack_to_box \
    --dataset.push_to_hub=False \
    --dataset.repo_id=${HF_USER}/move_plates_from_rack_to_box \
    --dataset.num_episodes=50 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="move each plate from the rack to the box" \
    --resume=true