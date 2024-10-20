import clip
import torch
import random
import numpy.random as npr
from typing import Any, List
from tqdm import tqdm
import os.path as osp
import numpy as np
import os
import json
import logging
from rlbench.demo import Demo
import pickle
from PIL import Image
from rlbench.backend.utils import image_to_float_array
from pyrep.objects import VisionSensor
from dataclasses import dataclass
from collections import defaultdict
import utils.math3d as math3d
from utils.clip import clip_encode_text
from rlbench.backend.observation import Observation
from utils.structure import *
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader, RandomSampler
from rlbench.backend.const import DEPTH_SCALE


def get_demo_essential_info(data_path, episode_ind):
    EPISODE_FOLDER = 'episode%d'
    episode_path = osp.join(data_path, EPISODE_FOLDER % episode_ind)
    # low dim pickle file
    with open(osp.join(episode_path, LOW_DIM_PICKLE), 'rb') as f:
        obs = pickle.load(f)

    with open(osp.join(episode_path, VARIATION_NUMBER_PICKLE), 'rb') as f:
        obs.variation_number = pickle.load(f)
    return obs


def retreive_full_observation(essential_obs, episode_path, i, load_mask=False, skip_rgb=False):
    CAMERA_FRONT = 'front'
    CAMERA_LS = 'left_shoulder'
    CAMERA_RS = 'right_shoulder'
    CAMERA_WRIST = 'wrist'
    CAMERAS = [CAMERA_FRONT, CAMERA_LS, CAMERA_RS, CAMERA_WRIST]

    IMAGE_RGB = 'rgb'
    IMAGE_DEPTH = 'depth'
    IMAGE_FORMAT  = '%d.png'

    obs = {}

    if load_mask:
        for c in CAMERAS:
            obs[f"{c}_mask"] = np.array(
                Image.open(osp.join(episode_path, f"{c}_mask", IMAGE_FORMAT % i))
            )

    if not skip_rgb:
        obs['front_rgb'] = np.array(Image.open(osp.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_RGB), IMAGE_FORMAT % i)))
        obs['left_shoulder_rgb'] = np.array(Image.open(osp.join(episode_path, '%s_%s' % (CAMERA_LS, IMAGE_RGB), IMAGE_FORMAT % i)))
        obs['right_shoulder_rgb'] = np.array(Image.open(osp.join(episode_path, '%s_%s' % (CAMERA_RS, IMAGE_RGB), IMAGE_FORMAT % i)))
        obs['wrist_rgb'] = np.array(Image.open(osp.join(episode_path, '%s_%s' % (CAMERA_WRIST, IMAGE_RGB), IMAGE_FORMAT % i)))

    obs['front_depth'] = image_to_float_array(Image.open(osp.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
    near = essential_obs.misc['%s_camera_near' % (CAMERA_FRONT)]
    far = essential_obs.misc['%s_camera_far' % (CAMERA_FRONT)]
    obs['front_depth'] = near + obs['front_depth'] * (far - near)

    obs['left_shoulder_depth'] = image_to_float_array(Image.open(osp.join(episode_path, '%s_%s' % (CAMERA_LS, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
    near = essential_obs.misc['%s_camera_near' % (CAMERA_LS)]
    far = essential_obs.misc['%s_camera_far' % (CAMERA_LS)]
    obs['left_shoulder_depth'] = near + obs['left_shoulder_depth'] * (far - near)

    obs['right_shoulder_depth'] = image_to_float_array(Image.open(osp.join(episode_path, '%s_%s' % (CAMERA_RS, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
    near = essential_obs.misc['%s_camera_near' % (CAMERA_RS)]
    far = essential_obs.misc['%s_camera_far' % (CAMERA_RS)]
    obs['right_shoulder_depth'] = near + obs['right_shoulder_depth'] * (far - near)

    obs['wrist_depth'] = image_to_float_array(Image.open(osp.join(episode_path, '%s_%s' % (CAMERA_WRIST, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
    near = essential_obs.misc['%s_camera_near' % (CAMERA_WRIST)]
    far = essential_obs.misc['%s_camera_far' % (CAMERA_WRIST)]
    obs['wrist_depth'] = near + obs['wrist_depth'] * (far - near)

    obs['front_point_cloud'] = VisionSensor.pointcloud_from_depth_and_camera_params(obs['front_depth'],
                                                                                    essential_obs.misc['front_camera_extrinsics'],
                                                                                    essential_obs.misc['front_camera_intrinsics'])
    obs['left_shoulder_point_cloud'] = VisionSensor.pointcloud_from_depth_and_camera_params(obs['left_shoulder_depth'],
                                                                                            essential_obs.misc['left_shoulder_camera_extrinsics'],
                                                                                            essential_obs.misc['left_shoulder_camera_intrinsics'])
    obs['right_shoulder_point_cloud'] = VisionSensor.pointcloud_from_depth_and_camera_params(obs['right_shoulder_depth'],
                                                                                            essential_obs.misc['right_shoulder_camera_extrinsics'],
                                                                                            essential_obs.misc['right_shoulder_camera_intrinsics'])
    obs['wrist_point_cloud'] = VisionSensor.pointcloud_from_depth_and_camera_params(obs['wrist_depth'],
                                                                                    essential_obs.misc['wrist_camera_extrinsics'],
                                                                                    essential_obs.misc['wrist_camera_intrinsics'])
    return obs


def encode_time(t, episode_length=25):
    return (1. - (t / float(episode_length - 1))) * 2. - 1.


def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i != (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
            obs.gripper_open == demo[i - 1].gripper_open and
            demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
            next_is_not_final and gripper_state_no_change)
    return stopped


def keypoint_discovery(demo: Demo, stopping_delta: float=0.1) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # If change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or
                        last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open
    if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
            episode_keypoints[-2]:
        episode_keypoints.pop(-2)
    return episode_keypoints


def query_next_kf(f, kfs, return_index=False):
    for i, kf in enumerate(kfs):
        if kf > f:
            if return_index:
                return i
            else:
                return kf
    raise RuntimeError("No more keyframes")


def get_reasonable_low_dim_state(essential_obs): # dim=18
    return np.array([
            essential_obs.gripper_open,
            essential_obs.ignore_collisions,
            *essential_obs.gripper_joint_positions,
            *essential_obs.joint_positions,
            *essential_obs.gripper_pose
        ]).astype(np.float32) # 18


class TransitionDataset(Dataset):
    def __init__(self, root: str, tasks: List[str], cameras:List[str]=["front", "left_shoulder", "right_shoulder", "wrist"],
                batch_num: int=1000, batch_size: int=6, scene_bounds=[-0.3,-0.5,0.6,0.7,0.5,1.6],
                voxel_size:int=100, rotation_resolution:int=5, cached_data_path=None,
                origin_style_state=True,
                episode_length=25, time_in_state=False, k2k_sample_ratios={}, o2k_window_size=10):
        super().__init__()
        self._num_batches = batch_num
        self._batch_size = batch_size
        self.tasks = tasks
        self.cameras = cameras
        self.origin_style_state = origin_style_state
        if not origin_style_state:
            assert not time_in_state, "should not include a discrete timestep in state"

        self.episode_length = episode_length
        self.root = root
        self.k2k_sample_ratios = k2k_sample_ratios
        self.o2k_window_size = o2k_window_size

        self.scene_bounds = scene_bounds
        self.voxel_size = voxel_size
        self.rotation_resolution = rotation_resolution
        self.include_time_in_state = time_in_state

        # task -> episode_id -> step_id
        if cached_data_path and osp.exists(cached_data_path):
            self.data = torch.load(cached_data_path)
        else:
            self.data = {}
            for task in tqdm(tasks, desc="building meta data"):
                episodes_path = osp.join(root, task, 'all_variations/episodes')
                if task not in self.data: self.data[task] = {}
                for episode in tqdm(os.listdir(episodes_path), desc="episodes", leave=False):
                    if 'episode' not in episode:
                        continue
                    else:
                        if episode not in self.data[task]: self.data[task][episode] = dict(keypoints=[], lang_emb=None, obs=None)
                        ep = osp.join(episodes_path, episode)
                        with open(osp.join(ep, KEYPOINT_JSON)) as f:
                            self.data[task][episode]['keypoints'] = json.load(f)
                        with open(osp.join(ep, LANG_GOAL_EMB), 'rb') as f:
                            self.data[task][episode]['lang_emb'] = pickle.load(f)
                        with open(osp.join(ep, LOW_DIM_PICKLE), 'rb') as f:
                            obs = pickle.load(f)
                        with open(osp.join(ep, VARIATION_NUMBER_PICKLE), 'rb') as f:
                            obs.variation_number = pickle.load(f)
                        self.data[task][episode]['obs'] = obs

            if cached_data_path:
                if not osp.exists(osp.dirname(cached_data_path)):
                    os.makedirs(osp.dirname(cached_data_path))
                torch.save(self.data, cached_data_path)


    def __len__(self): return self._num_batches

    def get(self, **kwargs):
        return self.__getitem__(0, **kwargs)

    def __getitem__(self, _):
        batch = defaultdict(list)
        for _ in range(self._batch_size):
            task = random.choice(list(self.data.keys()))
            episode = random.choice(list(self.data[task].keys()))
            episode_idx = int(episode[len('episode'):])
            episode_path = osp.join(self.root, task, 'all_variations/episodes', episode)
            episode = self.data[task][episode]

            # --------------------------------------- #
            u = random.random()
            if u < self.k2k_sample_ratios.get(task, 0.8):
                # k2k
                kp = random.randint(0, len(episode['keypoints'])-1) #! target keypoint
                obs_frame_id = 0 if kp == 0 else episode['keypoints'][kp-1]
            else:
                # o2k
                obs_frame_id = episode['keypoints'][0]
                while obs_frame_id in episode['keypoints']:
                    obs_frame_id = random.randint(0, episode['keypoints'][-1])
                # obs_frame_id is just an ordinary frame, not key frame
                kp = query_next_kf(obs_frame_id, episode['keypoints'], return_index=True)

            # --------------------------------------- #

            kp_frame_id = episode['keypoints'][kp]
            variation_id = episode['obs'].variation_number
            essential_obs = episode['obs'][obs_frame_id]
            essential_kp_obs = episode['obs'][kp_frame_id]
            obs_media_dict = retreive_full_observation(essential_obs, episode_path, obs_frame_id)

            if self.origin_style_state:
                curr_low_dim_state = np.array([essential_obs.gripper_open, *essential_obs.gripper_joint_positions])
                if self.include_time_in_state:
                    curr_low_dim_state = np.concatenate(
                        [curr_low_dim_state,
                        [encode_time(kp, episode_length=self.episode_length)]]
                    ).astype(np.float32)
            else:
                curr_low_dim_state = get_reasonable_low_dim_state(essential_obs)

            sample_dict = {
                "lang_goal_tokens": clip.tokenize(load_pkl(osp.join(episode_path, DESC_PICKLE))[0])[0].numpy(),
                "lang_goal_embs": episode['lang_emb'],
                "keypoint_idx": kp,
                "kp_frame_idx": kp_frame_id,
                "frame_idx": obs_frame_id,
                "episode_idx": episode_idx,
                "variation_idx": variation_id,
                "task_idx": self.tasks.index(task),

                "gripper_pose": essential_kp_obs.gripper_pose,
                "ignore_collisions": int(essential_kp_obs.ignore_collisions),

                "gripper_action": int(essential_kp_obs.gripper_open),
                "low_dim_state": curr_low_dim_state,

                **obs_media_dict
            }

            for k, v in sample_dict.items():
                batch[k].append(v)

            # reset
            task = episode = kp = obs_frame_id = None

        # lang_goals = batch.pop('lang_goals')
        batch = {k: np.array(v) for k, v in batch.items()}
        batch = {k: torch.from_numpy(v.astype('float32') if v.dtype == np.float64 else v)
                for k, v in batch.items()}
        batch = {k: v.permute(0, 3, 1, 2) if k.endswith('rgb') or k.endswith('point_cloud')
                else v for k,v in batch.items()}
        # batch['lang_goals'] = lang_goals
        return batch


    def dataloader(self, num_workers=1, pin_memory=True, distributed=False, pin_memory_device=''):
        if distributed:
            sampler = DistributedSampler(self)
        else:
            sampler = RandomSampler(range(len(self)))
        if pin_memory and pin_memory_device != '':
            pin_memory_device = f'cuda:{pin_memory_device}'
        return DataLoader(self, batch_size=None, shuffle=False, pin_memory=pin_memory,
                        sampler=sampler, num_workers=num_workers, pin_memory_device=pin_memory_device), sampler

if __name__ == "__main__":
    only_key_frames_ratios = {
      "place_cups": 1,
      "stack_cups": 1,
      "close_jar": 1,
      "push_buttons": 1,
      "meat_off_grill": 1,
      "stack_blocks": 1,
      "reach_and_drag": 1,
      "slide_block_to_color_target": 1,
      "place_shape_in_shape_sorter": 1,
      "open_drawer": 1,
      "sweep_to_dustpan_of_size": 1,
      "put_groceries_in_cupboard": 1,
      "light_bulb_in": 1,
      "turn_tap": 1,
      "insert_onto_square_peg": 1,
      "put_item_in_drawer": 1,
      "put_money_in_safe": 1,
      "place_wine_at_rack_location": 1
    }
    D = TransitionDataset("./data/train", ["open_drawer"],
                          origin_style_state=True, time_in_state=False, k2k_sample_ratios=only_key_frames_ratios)
    D[0]
