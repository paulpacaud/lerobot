import os
import argparse
import json
import shutil
import pickle as pkl
from PIL import Image
import copy

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from rlbench_utils.config import RLBENCH_FEATURES
from rlbench_utils.camera_depth_utils import extract_camera_info

from pointact.utils.depth import get_real_metric_depth, image_to_float_array
from pointact.utils.point_cloud import get_voxelized_point_cloud_from_rgb_depth

DEPTH_RGB_SCALE_FACTOR = 2**24 - 1

POINT_WORKSPACE = {
    'X_BBOX': [-0.5, 1.5],      # 0 is the robot base
    'Y_BBOX': [-1, 1],          # 0 is the robot base 
    'Z_BBOX': [0.7505, 2],      # 0 is the floor, 0.75 table height
}

def load_keystep_episodes(
    keystep_dir, microstep_dir, taskvars=None, keep_stop_action=False,
):
    
    if taskvars is None:
        taskvars = [x for x in os.listdir(keystep_dir) if '+' in x]
    print(f'#taskvars {len(taskvars)}')

    CAMERA_NAMES = ['left_shoulder', 'right_shoulder', 'wrist', 'front']

    for taskvar in taskvars:
        taskvar_dir = os.path.join(keystep_dir, taskvar)
        task, variation = taskvar.split('+')
        micro_taskvar_dir = os.path.join(microstep_dir, task, f'variation{variation}', 'episodes')

        with lmdb.open(taskvar_dir, readonly=True) as lmdb_env:
            with lmdb_env.begin() as txn:
                num_episodes = txn.stat()['entries']
                # for key, value in txn.cursor(): # the order is wrong
                for episode_id in range(num_episodes):
                    episode_key = f'episode{episode_id}'
                    value = txn.get(episode_key.encode('ascii'))

                    robot_state_file = os.path.join(
                        micro_taskvar_dir, episode_key, 'low_dim_obs.pkl'
                    )
                    robot_states = pkl.load(open(robot_state_file, 'rb'))

                    value = msgpack.unpackb(value)
                    demo_len = len(value['key_frameids'])

                    key_frameids = value['key_frameids']
                    
                    points_frontview, points_4views = [], []
                    for t, frameid in enumerate(key_frameids):
                        # TODO: check if the robot_state file and the keystep file match
                        pose_diff = np.abs(value['action'][t][:7] - robot_states[frameid].gripper_pose)
                        # print(t, frameid, pose_diff.mean(), pose_diff.max())
                        assert np.isclose(np.mean(pose_diff), 0, atol=1e-5), np.mean(pose_diff)

                        camera_info = extract_camera_info(
                            robot_states[frameid].misc, convert_camera=True
                        )
                        depths, intrinsics, extrinsics = [], [], []
                        for cam_name in CAMERA_NAMES:
                            depth_image = np.array(Image.open(
                                os.path.join(micro_taskvar_dir, episode_key, f'{cam_name}_depth', f'{frameid}.png'))
                            )
                            # use RGB 3 channels to represent depth (24 bits)
                            depth = image_to_float_array(depth_image, scale_factor=DEPTH_RGB_SCALE_FACTOR)
                            depth = get_real_metric_depth(
                                depth, camera_info[cam_name]['near'], camera_info[cam_name]['far']
                            )
                            depths.append(depth)
                            intrinsics.append(camera_info[cam_name]['intrinsics'])
                            extrinsics.append(camera_info[cam_name]['extrinsics'])
                        
                        rgbs = value['rgb'][t]

                        points_frontview.append(
                            get_voxelized_point_cloud_from_rgb_depth(
                                rgbs[-1:], depths[-1:], intrinsics[-1:], extrinsics[-1:], 
                                POINT_WORKSPACE,
                                voxel_size=0.01
                            )
                        )
                        points_4views.append(
                            get_voxelized_point_cloud_from_rgb_depth(
                                rgbs, depths, intrinsics, extrinsics, 
                                POINT_WORKSPACE,
                                voxel_size=0.01
                            )
                        )

                    gripper_poses = copy.copy(value['action'])
                    state = gripper_poses
                    action = np.concatenate([gripper_poses[1:], gripper_poses[-1:]], 0)

                    if keep_stop_action:
                        end_idx = None
                    else:
                        end_idx = -1
                        demo_len = demo_len - 1
                    episode = {
                        'observation.images.left_shoulder_image': value['rgb'][:end_idx, 0],
                        'observation.images.right_shoulder_image': value['rgb'][:end_idx, 1],
                        'observation.images.wrist_image': value['rgb'][:end_idx, 2],
                        'observation.images.front_image': value['rgb'][:end_idx, 3],
                        "observation.points.frontview": points_frontview[:end_idx],
                        "observation.points.4views": points_4views[:end_idx],
                        'observation.state': np.array(state[:end_idx], dtype=np.float32),
                        'action': np.array(action[:end_idx], dtype=np.float32),
                    }

                    yield (taskvar, [{k: v[i] for k, v in episode.items()} for i in range(demo_len)])


def main(args):

    repo_dir = os.path.join(args.output_dir, args.repo_id)
    if os.path.exists(repo_dir):
        print(f'{repo_dir} exists!')
        shutil.rmtree(repo_dir)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=repo_dir,
        fps=20,
        robot_type="franka",
        features=RLBENCH_FEATURES,
    )
    
    taskvar2instrs = json.load(open(args.task_instruction_file))

    if args.taskvar_file is not None:
        taskvars = json.load(open(args.taskvar_file))
    else:
        taskvars = args.taskvars

    # Check if taskvar has instructions
    for taskvar in taskvars:
        assert taskvar in taskvar2instrs

    raw_dataset = load_keystep_episodes(
        args.keystep_dir, args.microstep_dir, 
        taskvars=taskvars, 
        keep_stop_action=args.keep_stop_action
    )

    point_frontview_lmdb_env = lmdb.open(
        os.path.join(os.path.join(repo_dir, 'points_frontview')),
        map_size=int(1024**4)
    )
    point_4views_lmdb_env = lmdb.open(
        os.path.join(os.path.join(repo_dir, 'points_4views')),
        map_size=int(1024**4)
    )

    npoints_frontview, npoints_4views = [], []

    for episode_index, (taskvar, episode_data) in enumerate(raw_dataset):

        # multiple instructions: use special token <br> to separate different sentences
        task_instruction = '<br>'.join(taskvar2instrs[taskvar]) #[0]

        for frame_data in episode_data:
            point_frontview = frame_data['observation.points.frontview']
            point_4views = frame_data['observation.points.4views']
            npoints_frontview.append(len(point_frontview))
            npoints_4views.append(len(point_4views))
            del frame_data['observation.points.frontview']
            del frame_data['observation.points.4views']
            
            dataset.add_frame(
                frame_data,
                task=task_instruction,
            )

            episode_buffer = dataset.episode_buffer
            point_key = "{episode_id}-{step_id}".format(
                episode_id=episode_buffer["episode_index"],
                step_id=episode_buffer["frame_index"][-1]
            )

            out_txn = point_frontview_lmdb_env.begin(write=True)
            out_txn.put(
                point_key.encode('ascii'), msgpack.packb(point_frontview)
            )
            out_txn.commit()

            out_txn = point_4views_lmdb_env.begin(write=True)
            out_txn.put(
                point_key.encode('ascii'), msgpack.packb(point_4views)
            )
            out_txn.commit()

        dataset.save_episode()
        print(f"process done for {dataset.repo_id}, episode {episode_index}, len {len(episode_data)}")
        print('npoints front view', np.min(npoints_frontview), np.max(npoints_frontview), np.mean(npoints_frontview))
        print('npoints multi view', np.min(npoints_4views), np.max(npoints_4views), np.mean(npoints_4views))

    print('#data', len(npoints_frontview))
    print('npoints front view', np.min(npoints_frontview), np.max(npoints_frontview), np.mean(npoints_frontview))
    print('npoints multi view', np.min(npoints_4views), np.max(npoints_4views), np.mean(npoints_4views))

    point_frontview_lmdb_env.close()
    point_4views_lmdb_env.close()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--keystep_dir', required=True)
    parser.add_argument('--microstep_dir', required=True)
    parser.add_argument('--task_instruction_file', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--repo_id', required=True)
    parser.add_argument('--taskvar_file', default=None, type=str)
    parser.add_argument('--taskvars', default=None, nargs='+', type=str)
    parser.add_argument('--keep_stop_action', action='store_true', default=False)
    args = parser.parse_args()

    main(args)