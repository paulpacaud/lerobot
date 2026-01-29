# Training and Evaluation with EE vs Joint Action Space

This example demonstrates how to train and evaluate policies using different action representations:
- **EE (End-Effector)**: Actions are Cartesian poses `[x, y, z, wx, wy, wz, gripper]`
- **Joints**: Actions are joint positions `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`

Both datasets undergo the same idle frame trimming (computed on EE pose deltas) to ensure fair comparison.

## Workflow

### Step 1: Prepare Datasets

Convert your joint-space dataset to create two trimmed datasets (trimming based on deltas in observation.state (actual robot position))

```bash
python -m examples.train_eval_baselines_with_EEF.prepare_baseline_datasets \
    --input_dir=/home/ppacaud/lerobot_datasets/put_banana_and_toy_in_plates \
    --urdf_path=./URDF/SO101/so101_new_calib.urdf \
    --num_workers=8
    
###############

lerobot-edit-dataset \
    --repo_id  /home/ppacaud/lerobot_datasets/multitasks_3tasks_ee \
    --operation.type merge \
    --operation.repo_ids "['/home/ppacaud/lerobot_datasets/put_cube_in_spot_ee', '/home/ppacaud/lerobot_datasets/put_banana_and_toy_in_plates_ee', '/home/ppacaud/lerobot_datasets/put_sockets_into_drawer_ee']"

lerobot-edit-dataset \
    --repo_id  /home/ppacaud/lerobot_datasets/multitasks_3tasks_joints \
    --operation.type merge \
    --operation.repo_ids "['/home/ppacaud/lerobot_datasets/put_cube_in_spot_joints', '/home/ppacaud/lerobot_datasets/put_banana_and_toy_in_plates_joints', '/home/ppacaud/lerobot_datasets/put_sockets_into_drawer_joints']"

###############

huggingface-cli upload ${HF_USER}/multitasks_3tasks_ee $HOME/lerobot_datasets/multitasks_3tasks_ee --repo-type dataset

huggingface-cli upload ${HF_USER}/multitasks_3tasks_joints $HOME/lerobot_datasets/multitasks_3tasks_joints --repo-type dataset

###############

huggingface-cli download paulpacaud/multitasks_3tasks_ee \
  --repo-type dataset \
  --local-dir multitasks_3tasks_ee \
  --local-dir-use-symlinks False
  
huggingface-cli download paulpacaud/multitasks_3tasks_joints \
  --repo-type dataset \
  --local-dir multitasks_3tasks_joints \
  --local-dir-use-symlinks False
```

This creates (in the same directory as `input_dir`):
- `<dataset_name>_ee/` - Actions are EE poses
- `<dataset_name>_joints/` - Actions are joint positions (trimmed identically)

### Step 2: Train Policies

LeRobot treats the action as an opaque tensor - it doesn't care whether the values represent joint angles or EE poses. It just learns to predict the next action vector based on the           
observations. The semantic meaning only matters at inference time when you need to convert EE poses back to joint commands via IK.

Train two policies for comparison.

```bash
# Train on EE actions
python src/lerobot/scripts/lerobot_train.py \
    --dataset.root=/path/to/<dataset_name>_ee \
    --policy.type=pi0 \
    --output_dir=/path/to/outputs/pi0_ee \
    ...

# Train on Joint actions
python src/lerobot/scripts/lerobot_train.py \
    --dataset.root=/path/to/<dataset_name>_joints \
    --policy.type=pi0 \
    --output_dir=/path/to/outputs/pi0_joints \
    ...
```

### Step 3: Run Inference
#### Policy server:
`TORCH_COMPILE_DISABLE=1 python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080`

#### Robot client:
`ssh -N -L 8080:127.0.0.1:8080 ppacaud@dgx-station.paris.inria.fr`

**For joint-space policy** (standard):
```bash
# pi0
python -m lerobot.async_inference.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm \
  --robot.cameras="{ front: {type: intelrealsense, serial_number_or_name: 147122078460, width: 640, height: 480, fps: 30, use_depth: false}}" \
  --task="stack the yellow cup onto the blue cup, then stack the orange cup onto the yellow cup" \
  --policy_type=pi0 \
--pretrained_name_or_path=/home/ppacaud/data/lerobot/models/pi0_multitasks_5tasks_joints_20260128_224714-ckpt10k \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0 \
  --aggregate_fn_name=weighted_average

# groot
python -m lerobot.async_inference.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm \
  --robot.cameras="{ front: {type: intelrealsense, serial_number_or_name: 147122078460, width: 640, height: 480, fps: 30, use_depth: false}}" \
  --task="put the cube in the green square spot" \
  --policy_type=groot \
--pretrained_name_or_path=/home/ppacaud/data/lerobot/models/groot1.5_multitasks_3tasks_joints_20260128_033421-ckpt8k \
  --policy_device=cuda \
  --actions_per_chunk=16 \
  --chunk_size_threshold=0 \
  --aggregate_fn_name=weighted_average
```


**For EE-space policy** (requires IK conversion):
add --action_space=ee and --urdf_path=...
and switch to the _ee model
```bash
python -m lerobot.async_inference.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm \
  --robot.cameras="{ front: {type: intelrealsense, serial_number_or_name: 147122078460, width: 640, height: 480, fps: 30, use_depth: false}}" \
  --task="put the banana in the blue plate, then put the green toy in the pink plate" \
  --policy_type=pi0 \
  --pretrained_name_or_path=/home/ppacaud/data/lerobot/models/pi0_multitasks_3tasks_ee_20260128_033350-ckpt10k \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0 \
  --aggregate_fn_name=weighted_average \
  --action_space=ee \
  --urdf_path=./URDF/SO101/so101_new_calib.urdf

# groot
python -m lerobot.async_inference.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm \
  --robot.cameras="{ front: {type: intelrealsense, serial_number_or_name: 147122078460, width: 640, height: 480, fps: 30, use_depth: false}}" \
  --task="put the banana in the blue plate, then put the green toy in the pink plate" \
  --policy_type=groot \
--pretrained_name_or_path=/home/ppacaud/data/lerobot/models/groot1.5_multitasks_3tasks_ee_20260128_033421-ckpt8k \
  --policy_device=cuda \
  --actions_per_chunk=16 \
  --chunk_size_threshold=0 \
  --aggregate_fn_name=weighted_average \
  --action_space=ee \
  --urdf_path=./URDF/SO101/so101_new_calib.urdf
```

Notes:
increasing the value of chunk_size_threshold will result in sending out to the PolicyServer observations for inference more often, resulting in a larger number of updates action chunks, overlapping on significant portions. This results in high adaptability

## Trimming Strategy

The trimming is computed on **EE pose deltas** (not joint deltas) because:
- EE pose movement better reflects task-space activity
- A robot could have joint motion with minimal EE movement (or vice versa)

# cleps

ssh -N -v -L 17000:gpu017:17000 ppacaud@cleps.inria.fr

```bash
python -m lerobot.async_inference.pointact_robot_client --server_address=localhost:17000 --robot.type=so100_follower --robot.port=/dev/ttyACM0 --robot.id=follower_arm --robot.cameras="{ front: {type: intelrealsense, serial_number_or_name: 147122078460, width: 640, height: 480, fps: 30, use_depth: true}}" --urdf_path=./examples/post_process_dataset/constants/SO101/so101_new_calib.urdf --intrinsics_file=./examples/post_process_dataset/constants/intrinsics.npz --extrinsics_file=./examples/post_process_dataset/constants/extrinsics.npz --task="put the cube in the green square spot" --repo_id=paulpacaud/put_banana_and_toy_in_plates_pointact --fps=30
```
python -m lerobot.async_inference.pointact_robot_client \                                                                                                                                                                                                                                                                                                                         
    --server_address=localhost:17000 \                                                                                                                                                                                                                                                                                                                                              
    --robot.type=so100_follower \                                                                                                                                                                                                                                                                                                                                                   
    --robot.port=/dev/ttyACM0 \                                                                                                                                                                                                                                                                                                                                                     
    --robot.id=follower_arm \                                                                                                                                                                                                                                                                                                                                                       
    --robot.cameras="{ front: {type: intelrealsense, serial_number_or_name: 147122078460, width: 640, height: 480, fps: 30, use_depth: true}}" \                                                                                                                                                                                                                                    
    --urdf_path=./examples/post_process_dataset/constants/SO101/so101_new_calib.urdf \                                                                                                                                                                                                                                                                                              
    --intrinsics_file=./examples/post_process_dataset/constants/intrinsics.npz \                                                                                                                                                                                                                                                                                                    
    --extrinsics_file=./examples/post_process_dataset/constants/extrinsics.npz \                                                                                                                                                                                                                                                                                                    
    --task="put the cube in the green square spot" \                                                                                                                                                                                                                                                                                                                                
    --repo_id=paulpacaud/put_banana_and_toy_in_plates_pointact \                                                                                                                                                                                                                                                                                                                    
    --fps=30 


python -m lerobot.async_inference.pointact_robot_client --server_address=localhost:17000 --robot.type=so100_follower --robot.port=/dev/ttyACM0 --robot.id=follower_arm --robot.cameras="{ front: {type: intelrealsense, serial_number_or_name: 147122078460, width: 640, height: 480, fps: 30, use_depth: true}}"                                                                 
  --urdf_path=./examples/post_process_dataset/constants/SO101/so101_new_calib.urdf --intrinsics_file=./examples/post_process_dataset/constants/intrinsics.npz --extrinsics_file=./examples/post_process_dataset/constants/extrinsics.npz --task="put the cube in the green square spot" --repo_id=paulpacaud/put_banana_and_toy_in_plates_pointact --fps=30