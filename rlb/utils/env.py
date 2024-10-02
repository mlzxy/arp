from clip import tokenize
import imageio.v3 as iio
import os
import os.path as osp
from dataset import get_reasonable_low_dim_state
from abc import ABC, abstractmethod
from typing import Any, List, Type
import numpy as np
from utils.structure import ObservationElement, Transition, ROBOT_STATE_KEYS, ActResult, \
    Summary, VideoSummary, TextSummary, ImageSummary, Env
from utils.str import insert_uline_before_cap 
try:
    from rlbench import ObservationConfig, Environment, CameraConfig
except (ModuleNotFoundError, ImportError) as e:
    print("You need to install RLBench: 'https://github.com/stepjam/RLBench'")
    raise e
from rlbench.action_modes.action_mode import ActionMode
from rlbench.backend.observation import Observation
from rlbench.backend.task import Task
from pyrep.objects import VisionSensor, Dummy
from pyrep.const import RenderMode
from rlbench.action_modes.arm_action_modes import (
    EndEffectorPoseViaPlanning as _EndEffectorPoseViaPlanning,
    Scene,
)
from pyrep.errors import IKError, ConfigurationPathError
from rlbench.backend.exceptions import InvalidActionError



class CameraMotion(object):
    def __init__(self, cam: VisionSensor):
        self.cam = cam

    def step(self):
        raise NotImplementedError()

    def save_pose(self):
        self._prev_pose = self.cam.get_pose()

    def restore_pose(self):
        self.cam.set_pose(self._prev_pose)


class CircleCameraMotion(CameraMotion):

    def __init__(self, cam: VisionSensor, origin: Dummy,
                 speed: float, init_rotation: float = np.deg2rad(0)):
        super().__init__(cam)
        self.origin = origin
        self.speed = speed  # in radians
        self.origin.rotate([0, 0, init_rotation])

    def step(self):
        self.origin.rotate([0, 0, self.speed])



class EndEffectorPoseViaPlanning(_EndEffectorPoseViaPlanning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def action(self, scene: Scene, action: np.ndarray, ignore_collisions: bool = True):
        action[:3] = np.clip(
            action[:3],
            np.array(
                [scene._workspace_minx, scene._workspace_miny, scene._workspace_minz]
            )
            + 1e-7,
            np.array(
                [scene._workspace_maxx, scene._workspace_maxy, scene._workspace_maxz]
            )
            - 1e-7,
        )
        super().action(scene, action, ignore_collisions)



def rlbench_obs_config(camera_names: List[str],
                    camera_resolution: List[int],
                    method_name: str):
    unused_cams = CameraConfig()
    unused_cams.set_all(False)
    used_cams = CameraConfig(
        rgb=True,
        point_cloud=True,
        mask=True,
        depth=False,
        image_size=camera_resolution,
        render_mode=RenderMode.OPENGL)

    cam_obs = []
    kwargs = {}
    for n in camera_names:
        kwargs[n] = used_cams
        cam_obs.append('%s_rgb' % n)
        cam_obs.append('%s_pointcloud' % n)

    obs_config = ObservationConfig(
        front_camera=kwargs.get('front', unused_cams),
        left_shoulder_camera=kwargs.get('left_shoulder', unused_cams),
        right_shoulder_camera=kwargs.get('right_shoulder', unused_cams),
        wrist_camera=kwargs.get('wrist', unused_cams),
        overhead_camera=kwargs.get('overhead', unused_cams),
        joint_forces=False,
        joint_positions=True,
        joint_velocities=True,
        task_low_dim_state=False,
        gripper_touch_forces=False,
        gripper_pose=True,
        gripper_open=True,
        gripper_matrix=True,
        gripper_joint_positions=True,
    )

    obs_config.left_shoulder_camera.masks_as_one_channel = False
    obs_config.right_shoulder_camera.masks_as_one_channel = False
    obs_config.overhead_camera.masks_as_one_channel = False
    obs_config.wrist_camera.masks_as_one_channel = False
    obs_config.front_camera.masks_as_one_channel = False
    return obs_config



def _get_cam_observation_elements(camera: CameraConfig, prefix: str, channels_last):
    elements = []
    img_s = list(camera.image_size)
    shape = img_s + [3] if channels_last else [3] + img_s
    if camera.rgb:
        elements.append(
            ObservationElement('%s_rgb' % prefix, shape, np.uint8))
    if camera.point_cloud:
        elements.append(
            ObservationElement('%s_point_cloud' % prefix, shape, np.float32))
        elements.append(
            ObservationElement('%s_camera_extrinsics' % prefix, (4, 4),
                               np.float32))
        elements.append(
            ObservationElement('%s_camera_intrinsics' % prefix, (3, 3),
                               np.float32))
    if camera.depth:
        shape = img_s + [1] if schannels_last else [1] + img_s
        elements.append(
            ObservationElement('%s_depth' % prefix, shape, np.float32))
    if camera.mask:
        raise NotImplementedError()

    return elements


def _observation_elements(observation_config, channels_last) -> List[ObservationElement]:
    elements = []
    robot_state_len = 0
    if observation_config.joint_velocities:
        robot_state_len += 7
    if observation_config.joint_positions:
        robot_state_len += 7
    if observation_config.joint_forces:
        robot_state_len += 7
    if observation_config.gripper_open:
        robot_state_len += 1
    if observation_config.gripper_pose:
        robot_state_len += 7
    if observation_config.gripper_joint_positions:
        robot_state_len += 2
    if observation_config.gripper_touch_forces:
        robot_state_len += 2
    if observation_config.task_low_dim_state:
        raise NotImplementedError()
    if robot_state_len > 0:
        elements.append(ObservationElement(
            'low_dim_state', (robot_state_len,), np.float32))
    elements.extend(_get_cam_observation_elements(
        observation_config.left_shoulder_camera, 'left_shoulder', channels_last))
    elements.extend(_get_cam_observation_elements(
        observation_config.right_shoulder_camera, 'right_shoulder', channels_last))
    elements.extend(_get_cam_observation_elements(
        observation_config.front_camera, 'front', channels_last))
    elements.extend(_get_cam_observation_elements(
        observation_config.wrist_camera, 'wrist', channels_last))
    return elements


def _extract_obs(obs: Observation, channels_last: bool, observation_config: ObservationConfig):
    misc = obs.misc
    obs_dict = vars(obs)
    obs_dict = {k: v for k, v in obs_dict.items() if v is not None}
    robot_state = obs.get_low_dim_data()
    # **Remove** all of the individual state elements
    obs_dict = {k: v for k, v in obs_dict.items()
                if k not in ROBOT_STATE_KEYS} 
    if not channels_last:
        # Swap channels from last dim to 1st dim
        obs_dict = {k: np.transpose(
            v, [2, 0, 1]) if v.ndim == 3 else np.expand_dims(v, 0)
                    for k, v in obs_dict.items()}
    else:
        # Add extra dim to depth data
        obs_dict = {k: v if v.ndim == 3 else np.expand_dims(v, -1)
                    for k, v in obs_dict.items()}
    obs_dict['low_dim_state'] = np.array(robot_state, dtype=np.float32)
    for (k, v) in [(k, v) for k, v in obs_dict.items() if 'point_cloud' in k]:
        obs_dict[k] = v.astype(np.float32)

    for config, name in [
        (observation_config.left_shoulder_camera, 'left_shoulder'),
        (observation_config.right_shoulder_camera, 'right_shoulder'),
        (observation_config.front_camera, 'front'),
        (observation_config.wrist_camera, 'wrist'),
        (observation_config.overhead_camera, 'overhead')]:
        if config.point_cloud:
            obs_dict['%s_camera_extrinsics' % name] = obs.misc['%s_camera_extrinsics' % name]
            obs_dict['%s_camera_intrinsics' % name] = obs.misc['%s_camera_intrinsics' % name]
    
    if 'object_ids' in misc:
        obs_dict['object_ids'] = misc['object_ids']
    return obs_dict


class MultiTaskRLBenchEnv(Env):

    def __init__(self,
                task_classes: List[Type[Task]],
                observation_config: ObservationConfig,
                action_mode: ActionMode,
                dataset_root: str = '',
                channels_last=False,
                headless=True,
                swap_task_every: int = 1, include_lang_goal_in_obs=False):

        self._eval_env = False
        self._include_lang_goal_in_obs = include_lang_goal_in_obs
        self._task_classes = task_classes
        self._observation_config = observation_config
        self._channels_last = channels_last
        self._rlbench_env = Environment(
            action_mode=action_mode, obs_config=observation_config,
            dataset_root=dataset_root, headless=headless)
        self._task = None
        self._lang_goal = 'unknown goal'
        self._swap_task_every = swap_task_every
        self._rlbench_env
        self._episodes_this_task = 0
        self._active_task_id = -1
        
        self._task_name_to_idx = {insert_uline_before_cap(tc.__name__): i 
                                for i, tc in enumerate(self._task_classes)}

    
    @property
    def eval(self):
        return self._eval_env

    @eval.setter
    def eval(self, is_eval):
        self._eval_env = is_eval

    @property
    def active_task_id(self) -> int:
        return self._active_task_id

    def _set_new_task(self, shuffle=False):
        if shuffle:
            self._active_task_id = np.random.randint(0, len(self._task_classes))
        else:
            self._active_task_id = (self._active_task_id + 1) % len(self._task_classes)
        task = self._task_classes[self._active_task_id]
        self._task = self._rlbench_env.get_task(task)
    
    def set_task(self, task_name: str):
        self._active_task_id = self._task_name_to_idx[task_name]
        task = self._task_classes[self._active_task_id]
        self._task = self._rlbench_env.get_task(task)

        descriptions, _ = self._task.reset()
        self._lang_goal = descriptions[0] # first description variant

    def extract_obs(self, obs: Observation):
        extracted_obs = _extract_obs(obs, self._channels_last, self._observation_config)
        if self._include_lang_goal_in_obs:
            extracted_obs['lang_goal_tokens'] = tokenize([self._lang_goal])[0].numpy()
        return extracted_obs

    def launch(self):
        self._rlbench_env.launch()
        self._set_new_task()

    def shutdown(self):
        self._rlbench_env.shutdown()

    def reset(self) -> dict:
        self._episodes_this_task += 1
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0

        descriptions, obs = self._task.reset()
        self._lang_goal = descriptions[0] # first description variant
        return self.extract_obs(obs)

    def step(self, action: np.ndarray) -> Transition:
        obs, reward, terminal = self._task.step(action)
        obs = self.extract_obs(obs)
        return Transition(obs, reward, terminal)

    @property
    def observation_elements(self) -> List[ObservationElement]:
        """ return the specification of observable data """
        return _observation_elements(self._observation_config, self._channels_last)

    @property
    def action_shape(self):
        return (self._rlbench_env.action_size, )

    @property
    def env(self) -> Environment:
        return self._rlbench_env

    @property
    def num_tasks(self) -> int:
        return len(self._task_classes)
    
    

class CustomMultiTaskRLBenchEnv(MultiTaskRLBenchEnv):
    def __init__(self,
                task_classes: List[Type[Task]],
                observation_config: ObservationConfig,
                action_mode: ActionMode,
                episode_length: int,
                dataset_root: str = '',
                channels_last: bool = False,
                reward_scale=100.0,
                headless: bool = True,
                swap_task_every: int = 1,
                time_in_state: bool = False,
                include_lang_goal_in_obs: bool = False,
                origin_style_state: bool = True,
                record_video_folder: str = './outputs/recording',
                record_every_n: int = 20,
                record_resolution=[1280, 720]):
        super().__init__(
            task_classes, observation_config, action_mode, dataset_root,
            channels_last, headless=headless, swap_task_every=swap_task_every, 
            include_lang_goal_in_obs=include_lang_goal_in_obs)
        
        self.origin_style_state = origin_style_state
        if not origin_style_state:
            assert not time_in_state, "time_in_state is only supported with origin_style_state=True"
        
        self._reward_scale = reward_scale
        self._episode_index = 0
        self._record_current_episode = False
        self._record_cam = None
        self._previous_obs, self._previous_obs_dict = None, None
        self._recorded_images = []
        self._episode_length = episode_length
        self._time_in_state = time_in_state
        self._record_every_n = record_every_n
        self._record_video_folder = record_video_folder
        self._record_resolution = record_resolution
        self._i = 0
        self._error_type_counts = {
            'IKError': 0,
            'ConfigurationPathError': 0,
            'InvalidActionError': 0,
        }
        self._last_exception = None
    
    @property
    def observation_elements(self) -> List[ObservationElement]:
        obs_elems = super().observation_elements
        for oe in obs_elems:
            if oe.name == 'low_dim_state':
                oe.shape = (oe.shape[0] - 7 * 3 + int(self._time_in_state),)  # remove pose and joint velocities as they will not be included
                self.low_dim_state_len = oe.shape[0]
        return obs_elems

    def extract_obs(self, obs: Observation, t=None, prev_action=None):
        obs.joint_velocities = None
        grip_mat = obs.gripper_matrix
        grip_pose = obs.gripper_pose
        joint_pos = obs.joint_positions
        obs.gripper_pose = None
        obs.gripper_matrix = None
        obs.wrist_camera_matrix = None
        obs.joint_positions = None
        if obs.gripper_joint_positions is not None:
            obs.gripper_joint_positions = np.clip(
                obs.gripper_joint_positions, 0., 0.04)

        obs_dict = super().extract_obs(obs)

        if self._time_in_state: # will be overriden if not origin_style_state
            time = (1. - ((self._i if t is None else t) / float(
                self._episode_length - 1))) * 2. - 1.
            obs_dict['low_dim_state'] = np.concatenate(
                [obs_dict['low_dim_state'], [time]]).astype(np.float32)

        obs.gripper_matrix = grip_mat
        obs.joint_positions = joint_pos
        obs.gripper_pose = grip_pose
        obs_dict['gripper_pose'] = grip_pose
        obs_dict['gripper_open'] = obs.gripper_open

        if not self.origin_style_state:
            obs_dict['low_dim_state'] = get_reasonable_low_dim_state(obs)

        return obs_dict
    
    def reset(self) -> dict:
        self._i = 0
        self._previous_obs_dict = super().reset()
        self._episode_index += 1
        self._recorded_images.clear()
        self._record_current_episode = (
            self.eval
            and self._record_every_n > 0
            and self._episode_index % self._record_every_n == 0
        )
        if self._record_cam is not None:
            self._record_cam.restore_pose()
        return self._previous_obs_dict
    
    def launch(self):
        super().launch()

        def callback():
            if self._record_current_episode:
                self._record_cam.step()
                self._recorded_images.append(
                    (self._record_cam.cam.capture_rgb() * 255.).astype(np.uint8))

        self._task._scene.register_step_callback(callback)
        if self.eval and self._record_every_n > 0:
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            cam = VisionSensor.create(self._record_resolution)
            cam.set_pose(cam_placeholder.get_pose())
            cam.set_parent(cam_placeholder)
            cam_motion = CircleCameraMotion(cam, Dummy('cam_cinematic_base'), 0.005)
            self._record_cam = cam_motion
            self._record_cam.save_pose()
    
    def step(self, act_result: ActResult) -> Transition:
        action = act_result.action # from model
        success = False
        obs = self._previous_obs_dict  # in case action fails.

        try:
            obs, reward, terminal = self._task.step(action)
            if reward >= 1:
                success = True
                reward *= self._reward_scale
            else:
                reward = 0.0
            obs = self.extract_obs(obs)
            self._previous_obs_dict = obs
        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            print(e)
            terminal = True
            reward = 0.0

            if isinstance(e, IKError):
                self._error_type_counts['IKError'] += 1
            elif isinstance(e, ConfigurationPathError):
                self._error_type_counts['ConfigurationPathError'] += 1
            elif isinstance(e, InvalidActionError):
                self._error_type_counts['InvalidActionError'] += 1

            self._last_exception = e

        self._i += 1
        if ((terminal or self._i == self._episode_length) and
                self._record_current_episode):
            # append final frame            
            img = (self._record_cam.cam.capture_rgb() * 255).astype(np.uint8)
            self._recorded_images.append(img)

            frames = np.array(self._recorded_images)
            task_name = insert_uline_before_cap(self._task._task.__class__.__name__)

            fps = 30
            out_name = osp.join(self._record_video_folder, 'episode_rollout_' + ('success' if success else 'fail') + f'/{task_name}/{self._episode_index}.mp4')
            os.makedirs(osp.dirname(out_name), exist_ok=True)
            iio.imwrite(out_name, frames, fps=fps, plugin='FFMPEG') 
        return Transition(obs, reward, terminal)
    
    
    def reset_to_demo(self, i, variation_number=-1):
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1
        
        self._i = 0
        self._task.set_variation(-1)
        d = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=i, 

            image_paths=True # skip rgb/pcd loading, as they are not used
        )[0] # from dataset_root

        self._task.set_variation(d.variation_number)
        desc, obs = self._task.reset_to_demo(d)
        self._lang_goal = desc[0]

        self._previous_obs_dict = self.extract_obs(obs)
        self._record_current_episode = (
            self.eval
            and self._record_every_n > 0
            and self._episode_index % self._record_every_n == 0
        )
        if self._record_current_episode:
            print("recording current episode!")
        self._episode_index += 1
        self._recorded_images.clear()
        if self._record_cam is not None:
            self._record_cam.restore_pose()
        return self._previous_obs_dict
    



def get_active_task_of_env(env):
    return {v:k for k,v in env._task_name_to_idx.items()}[env._active_task_id]