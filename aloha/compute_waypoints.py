from PIL import Image, ImageDraw
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm, trange
from pathlib import Path
from torchvision.transforms.functional import to_pil_image
import imageio
import gymnasium as gym
import numpy as np
import gym_aloha
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from torchvision.utils import Optional, Tuple, Union, ImageDraw, List, Image
import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from lerobot.common.datasets.utils import get_hf_dataset_safe_version, load_dataset

cat = lambda *args: np.concatenate(args, axis=0)



def read_joint_xpos(model, data):
    # Read all joint positions from the MuJoCo model
    joint_left_xpos = {}
    joint_right_xpos = {}
    for joint_id in range(model.njnt):
        joint_name = model.id2name(joint_id, "joint")
        if not joint_name.startswith("vx300s"):
            # Skip the joints that are not part of the robot
            continue
        if joint_name.startswith("vx300s_left"):
            joint_left_xpos[joint_name] = data.xpos[model.jnt_bodyid[joint_id]]
        elif joint_name.startswith("vx300s_right"):
            joint_right_xpos[joint_name] = data.xpos[model.jnt_bodyid[joint_id]]
    return joint_left_xpos, joint_right_xpos


def project_joint_positions(joint_xpos, cam_pose, intrinsic_matrix):
    # Project the joint positions to the camera image
    joint_xpos_homog = np.hstack([joint_xpos, np.ones((joint_xpos.shape[0], 1))])
    joint_xpos_cam = joint_xpos_homog @ cam_pose.T
    joint_xpos_cam = joint_xpos_cam[:, :3]
    joint_xpos_img = joint_xpos_cam @ intrinsic_matrix.T
    joint_xpos_img = joint_xpos_img[:, :2] / joint_xpos_img[:, 2:]
    return joint_xpos_img


def get_joint_project_pos_from_env(env, cam_id="top"):
    mujoco_model = env.env.env.env._env.physics.model
    mujoco_data = env.env.env.env._env.physics.data
    joint_left_xpos, joint_right_xpos = read_joint_xpos(mujoco_model, mujoco_data)
    # Read camera pos & parameters
    cam_pos = mujoco_model.cam_pos[mujoco_model.name2id(cam_id, "camera")]
    cam_quat = mujoco_model.cam_quat[mujoco_model.name2id(cam_id, "camera")]
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = R.from_quat(cam_quat).as_matrix()
    cam_pose[:3, 3] = cam_pos
    cam_fovy = mujoco_model.cam_fovy[mujoco_model.name2id(cam_id, "camera")]
    img_width = mujoco_model.vis.global_.offwidth
    img_height = mujoco_model.vis.global_.offheight
    # Compute intrinsic matrix
    fovy = np.deg2rad(cam_fovy)
    f = img_height / (2 * np.tan(fovy / 2))
    intrinsic_matrix = np.array(
        [[f, 0, img_width / 2], [0, f, img_height / 2], [0, 0, 1]]
    )
    # Project the joint positions to the camera image
    left_keys = list(joint_left_xpos.keys())
    right_keys = list(joint_right_xpos.keys())

    
    joint_left_3d = np.array([joint_left_xpos[k] for k in left_keys])
    joint_left_3d = np.concatenate([joint_left_3d, joint_left_3d[6:8].mean(0, keepdims=True)])
    joint_left_2d = project_joint_positions(
        joint_left_3d, cam_pose, intrinsic_matrix
    )
    joint_right_3d = np.array([joint_right_xpos[k] for k in right_keys])
    joint_right_3d = np.concatenate([joint_right_3d, joint_right_3d[6:8].mean(0, keepdims=True)])
    joint_right_2d = project_joint_positions(
        joint_right_3d, cam_pose, intrinsic_matrix
    )
    return joint_left_2d, joint_left_3d, joint_right_2d, joint_right_3d, left_keys, right_keys


repo_envs = [
    ("lerobot/aloha_sim_insertion_human", "gym_aloha/AlohaInsertion-v0"),
    ("lerobot/aloha_sim_transfer_cube_human", "gym_aloha/AlohaTransferCube-v0"),
]

output_path = Path("./outputs/dataset_with_waypoints")


if __name__ == "__main__": # env.env.env.env._env._task
    for repo_id, env_name in tqdm(repo_envs):
        episode_id, frame_count = 0, 0
        storage = {}
        D = LeRobotDataset(repo_id, delta_timestamps={'action':[i / 50 for i in range(100)]})
        env = gym.make(env_name)
        observation, info = env.reset()
        L = len(D)
        for i in trange(L, leave=False):   # L
            d = D[i]
            if d['episode_index'].item() != episode_id:
                episode_id = d['episode_index'].item()
                frame_count = 0
                env.reset()

            joint_left_2d, joint_left_3d, joint_right_2d, joint_right_3d, left_keys, right_keys = get_joint_project_pos_from_env(
                    env, "top"
            )
            storage[(episode_id, frame_count)] = [joint_left_2d, joint_left_3d, joint_right_2d, joint_right_3d]
            observation, reward, terminated, truncated, info = env.step(d['action'][0])
            frame_count += 1
        
        safe_version = get_hf_dataset_safe_version(repo_id, "v1.6")
        D = load_dataset(repo_id, revision=safe_version, split="train")

        def make_patch(store):
            def patch(cell):
                row = store[(cell['episode_index'], cell['frame_index'])]
                cell['left_pts_2d'] = row[0]
                cell['left_pts_3d'] = row[1]
                cell['right_pts_2d'] = row[2]
                cell['right_pts_3d'] = row[3]
                return cell
            return patch

        for new_column in ['left_pts_2d', 'left_pts_3d', 'right_pts_2d', 'right_pts_3d']:
            D = D.add_column(new_column, [0] * len(D))

        D = D.map(make_patch(storage))
        D.save_to_disk(output_path / repo_id / "train")