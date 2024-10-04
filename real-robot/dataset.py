import sys
import os
from tqdm.auto import tqdm
from utils.preprocess import CubePointCloudRenderer, preprocess_images_in_batch, \
    flatten_img_pc_to_points, clamp_pc_in_bound, place_pc_in_cube, generate_heatmap_from_screen_pts
import torch
import open3d as o3d
import pickle
from glob import glob
import numpy as np
import os.path as osp
npl = np.linalg
import cv2
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image
import wrench
import torch
import utils.spatial as spatial
from torch.utils.data import Dataset, DataLoader, RandomSampler
from utils.math3d import sensitive_gimble_fix, normalize_rad, denormalize_rad, quat_to_norm_rpy, quat_to_degree_rpy
from utils.object import load_pkl
from utils.vis import im_concat
from collections import defaultdict
from wrench import eef_pose7_to_wrench_pose7, wrench_pose7_to_sm_eef_pose7, wrench_pose3_to_eef_pose3
from utils.preprocess import CubePointCloudRenderer, preprocess_images_in_batch, \
        flatten_img_pc_to_points, clamp_pc_in_bound, place_pc_in_cube, generate_heatmap_from_screen_pts
        

def collate_function(samples):
    batch = defaultdict(list)
    for sample in samples:
        for k, v in sample.items():
            batch[k].append(v)
    
    for k in ['direction', 'command']:
        if k in batch:
            batch[k] = torch.as_tensor(batch[k])
    
    # for k in ['curr_xyz', 'curr_rpy', 'target_xyz', 'target_rpy', 'target_xyz_traj', 'target_rpy_traj']:
    for k in ['curr_wrench', 'curr_gripper', 'target_wrench', 'target_gripper', 'target_refined_wrench', 'target_gripper_traj']:
        if k in batch:
            batch[k] = torch.stack(batch[k])
    return batch
    

class Episode:
    def __init__(self, initial_state, adjust_states, rotate_state):
        self.id = -1
        self.initial_state = initial_state
        self.adjust_states = adjust_states
        self.rotate_state = rotate_state

        self.depth_scale = 0.0010000000474974513
        self.intrinsics = None
        self.extrinsics = None
    
    @property
    def states(self):
        return [self.initial_state] + self.adjust_states + [self.rotate_state]
        

COMMANDS = ['reach', 'adjust', 'rotate']
REACH, ADJUST, ROTATE = 0, 1, 2

def read_episode(p, offset_z=0):
    state_files = sorted(glob(osp.join(p, '*.state.pkl')))
    states = [load_pkl(f) for f in state_files]

    def find_to_states(to_state):
        state_is = []
        _states = []
        for state_i, state in enumerate(states):
            if state['next_state'].strip() == to_state:
                state_is.append(state_i)
                _states.append(state)
        return state_is, _states
    
    over_nut_is, over_nuts = find_to_states('over nut') 
    over_nut_i, over_nut = over_nut_is[-1], over_nuts[-1]
    initial_state = {
        'image_id': 0, 
        'target_wrench_pose': over_nut['target_eef_pose'].astype(np.float32),
        'current_wrench_pose': eef_pose7_to_wrench_pose7(over_nuts[0]['current_eef_pose'].astype(np.float32)), 
        'current_eef_pose': over_nuts[0]['current_eef_pose'].astype(np.float32),
        'command': 'reach' 
    }

    initial_state['current_wrench_pose'][2] += offset_z

    recover_state_is, recover_states = find_to_states('recover')
    adjust_states = []
    for recover_state_i, recover_state in zip(recover_state_is, recover_states):
        target_wrench_pose = recover_state['target_eef_pose'].astype(np.float32).copy()
        target_wrench_pose[2] = recover_state['current_eef_pose'][2]
        adjust_states.append({
            'image_id': recover_state_i,
            'adjustment': recover_state['adjustment_direction'],
            'target_wrench_pose': eef_pose7_to_wrench_pose7(target_wrench_pose.astype(np.float32)),
            'current_wrench_pose': eef_pose7_to_wrench_pose7(recover_state['current_eef_pose'].astype(np.float32)), 
            'current_eef_pose': recover_state['current_eef_pose'].astype(np.float32),
            'command': 'adjust' 
        })
        adjust_states[-1]['current_wrench_pose'][2] += offset_z

    if len(adjust_states) > 0:
        _ = recover_states[-1]['target_eef_pose'].copy()
        _[2] = initial_state['target_wrench_pose'][2]
        initial_state['target_wrench_pose'] = eef_pose7_to_wrench_pose7(_)
    else:
        initial_state['target_wrench_pose'] = eef_pose7_to_wrench_pose7(initial_state['target_wrench_pose'])
    initial_state['target_wrench_pose'][2] += offset_z

    rotate_state_is, rotate_states = find_to_states('rotate')
    rotate_state = {
        'image_id': rotate_state_is[0],
        
        'current_wrench_pose': eef_pose7_to_wrench_pose7(rotate_states[0]['current_eef_pose'].astype(np.float32)),
        'target_wrench_poses': [eef_pose7_to_wrench_pose7(rotate_state['target_eef_pose'].astype(np.float32)) for rotate_state in rotate_states][:6],
        
        'current_eef_pose': rotate_states[0]['current_eef_pose'].astype(np.float32),
        'command': 'rotate' 
    }
    rotate_state['current_wrench_pose'][2] += offset_z
    for target_wrench_pose in rotate_state['target_wrench_poses']:
        target_wrench_pose[2] += offset_z
    episode = Episode(initial_state, adjust_states, rotate_state)

    episode.intrinsics = states[0]['intrinsics']
    episode.extrinsics = states[0]['extrinsics']
    episode.depth_scale = states[0]['depth_scale']
    episode.id = int(osp.basename(p))
    return episode
    

SCENE_BOUNDS = [
    0.48,
    -0.15,
    0.22,

    0.72,
    0.08,
    0.40
]

# 0.1, 0.05, 0.05


# [FINAL BOUND] =>  tensor([ 0.5075, -0.1583,  0.2639]) tensor([0.7249, 0.0687, 0.4990])
    
class RealRobotDataset(Dataset):
    def __init__(self, folder, episode_ids, batch_num=-1, offset_z=0.0) -> None:
        super().__init__()
        self.eval_mode = False
        self.episodes = [read_episode(osp.join(folder, str(i)), offset_z=offset_z) for i in episode_ids]
        real_size = sum([len(ep.states) for ep in self.episodes])
        if batch_num < 0:
            self.eval_mode = True
            batch_num = real_size
        
        self.pcds = {}
        self.batch_num = batch_num
        
        with tqdm(total=real_size, desc="processing") as bar:
            for ep in self.episodes:
                for state in ep.states:
                    self.pcds[(ep.id, state['image_id'])] = self.preprocess(osp.join(folder, str(ep.id), f'{str(state["image_id"]).zfill(3)}.rgb.png'), 
                                                                            osp.join(folder, str(ep.id), f'{str(state["image_id"]).zfill(3)}.depth.png'),
                                                                            ep.depth_scale, ep.intrinsics, ep.extrinsics)
                    bar.update()
        
        self.uniform_sample_groups = defaultdict(list) 
        for ep_id, ep in enumerate(self.episodes):
            for state_id, state in enumerate(ep.states):
                command = state['command']
                adjustment = state.get('adjustment', None)
                self.uniform_sample_groups[(command, adjustment)].append([ep_id, state_id])
                
    
    def direction_splits(self):
        splits = defaultdict(list)
        for ep in self.episodes:
            if len(ep.adjust_states) == 0:
                splits['one-shot'].append(ep.id)
            else:
                for k in {adj['adjustment'] for adj in ep.adjust_states}:
                    splits[k].append(ep.id)
        return dict(splits)
    
    def bound(self, name, exclude_reach=False):
        all_data = []
        for i in range(len(self)):
            sample = self[i]
            if exclude_reach:
                if sample['command'] == REACH:
                    continue
            if name in sample:
                all_data.append(sample[name].reshape(-1, sample[name].shape[-1]))
            
        all_data = torch.cat(all_data, dim=0)
        return all_data.min(dim=0).values, all_data.max(dim=0).values
    
    
    def preprocess(self, rgb_path, depth_path, depth_scale=None, intrinsics=None, extrinsics=None):
        import uuid
        if not isinstance(rgb_path, str):
            bs = rgb_path.read()
            rgb_path = f'/tmp/{uuid.uuid4()}.png'
            with open(rgb_path, 'wb') as f: f.write(bs)
        
        if not isinstance(depth_path, str):
            bs = depth_path.read()
            depth_path = f'/tmp/{uuid.uuid4()}.png'
            with open(depth_path, 'wb') as f: f.write(bs)
    
        if depth_scale is None:
            depth_scale = self.episodes[0].depth_scale
        if intrinsics is None:
            intrinsics = self.episodes[0].intrinsics
        if extrinsics is None:
            extrinsics = self.episodes[0].extrinsics

        rgb = o3d.io.read_image(rgb_path)
        depth = o3d.io.read_image(depth_path)

        height, width = np.asarray(rgb).shape[:2]
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb, depth, convert_rgb_to_intensity=False, depth_scale=1 / depth_scale)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                    rgbd_image, o3d.camera.PinholeCameraIntrinsic(width=width, height=height, intrinsic_matrix=intrinsics))
        pcd.transform(npl.inv(extrinsics))
        points = np.asarray(pcd.points)
        X, Y, Z = 0, 1, 2
        x_min, y_min, z_min, x_max, y_max, z_max = SCENE_BOUNDS
        valid_mask =(points[:, X] < x_max) & (points[:, X] > x_min) & (points[:, Y] < y_max) & (points[:, Y] > y_min) & (points[:, Z] > z_min) & (points[:, Z] < z_max)
        sub_pcd = pcd.select_by_index(valid_mask.nonzero()[0])
        return sub_pcd
    
    
    def __len__(self):
        return self.batch_num
    
    def __getitem__(self, index):
        return self.get(index)
    
    def get(self, index):
        """
        {   
            'points',
            'colors',

            'command',
            'direction',
            
            'curr_wrench',
            'curr_gripper', 

            'target_wrench',
            'target_gripper',
            
            'target_refined_wrench',
            'target_gripper_traj'
        }
        """
        if self.eval_mode:
            for ep in self.episodes:
                if index < len(ep.states):
                    state = ep.states[index]
                    break
                index -= len(ep.states)
            ep_id = ep.id 
            image_id = state['image_id']
        else:
            # _k = np.random.choice(list(self.uniform_sample_groups.keys()))
            keys = [('reach', None), ('adjust', '<'), ('rotate', None), ('adjust', '>')]
            _k = np.random.choice(list(range(4)), p=[0.27, 0.27, 0.27, 0.19])
            _k = keys[_k]
            ep_id, state_id = self.uniform_sample_groups[_k][np.random.choice(range(len(self.uniform_sample_groups[_k])))]
            ep = self.episodes[ep_id]
            ep_id = ep.id
            state = ep.states[state_id]
            image_id = state['image_id']
        pcd = self.pcds[(ep_id, state['image_id'])]
        
        points = torch.from_numpy(np.asarray(pcd.points)).float()
        colors = torch.from_numpy(np.asarray(pcd.colors)).float()
        result = {
            'points': points,
            'colors': colors,
            'image_id': image_id,
            'ep_id': ep.id,
            'command': COMMANDS.index(state['command']),
        }
        result['curr_wrench'] = torch.from_numpy(state['current_wrench_pose'][:3]).float()
        result['curr_rpy'] = torch.from_numpy(quat_to_degree_rpy(state['current_wrench_pose'][3:])).float()
        result['curr_gripper'] = torch.from_numpy(wrench_pose7_to_sm_eef_pose7(state['current_wrench_pose'])[:3]).float()
        result['curr_eef_pose'] = torch.from_numpy(state['current_eef_pose']).float()
        

        if state['command'] == 'reach':
            result['target_wrench'] = torch.from_numpy(state['target_wrench_pose'][:3]).float()
            result['target_rpy'] = torch.from_numpy(quat_to_degree_rpy(state['target_wrench_pose'][3:])).float()
            result['target_gripper'] = torch.from_numpy(wrench_pose7_to_sm_eef_pose7(state['target_wrench_pose'])[:3]).float()

            # THIS NEEDS TO BE tested
            result['target_eef_pose'] = wrench.wrench_pose7_to_eef_pose7(state['target_wrench_pose'])

        elif state['command'] == 'adjust':
            result['direction'] = 0 if state['adjustment'] == '<' else 1
            result['target_refined_wrench'] = torch.from_numpy(state['target_wrench_pose'][:3]).float()

        elif state['command'] == 'rotate':
            result['target_gripper_traj'] = torch.stack([torch.from_numpy(wrench_pose7_to_sm_eef_pose7(pose)[:3]).float() 
                                                         for pose in state['target_wrench_poses']])
            # result['target_rpy_traj'] = torch.stack([torch.from_numpy(quat_to_degree_rpy(pose[3:])).float() for pose in state['target_wrench_poses']])
        return result
    
    
    def full_gripper_to_wrench_and_small_gripper(self, gripper_pose):
        wrench_pose = eef_pose7_to_wrench_pose7(gripper_pose.astype(np.float32))
        sm_gripper_pose = wrench_pose7_to_sm_eef_pose7(wrench_pose)
        return wrench_pose, sm_gripper_pose

    def to_target_full_gripper(self, target_wrench_xyz, target_sm_gripper_xyz, gt_pose=None):
        if hasattr(target_wrench_xyz, 'numpy'):
            target_wrench_xyz = target_wrench_xyz.numpy()
        if hasattr(target_sm_gripper_xyz, 'numpy'):
            target_sm_gripper_xyz = target_sm_gripper_xyz.numpy()

        from scipy.optimize import minimize
        x0, y0 = wrench_pose3_to_eef_pose3(target_wrench_xyz)[:2]
        rz0 = np.pi * 0.75
        z0 = target_wrench_xyz[2] + wrench.eef_T_wrenchHead[2,3]
        rx0, ry0 = np.pi, 0
        X0 = np.array([x0, y0, z0, rx0, ry0, rz0])

        if gt_pose is not None:
            gt = np.array([gt_pose[0], gt_pose[1],  *Rotation.from_quat(gt_pose[3:]).as_euler('xyz').tolist()]).astype(np.float32)
        else:
            gt = None

        def objective_function(x):
            xyz = np.array([x[0], x[1], x[2]])
            quat = Rotation.from_euler('xyz', x[3:]).as_quat()
            target_p7 = np.concatenate([xyz, quat])

            target_wrench_p7 = eef_pose7_to_wrench_pose7(target_p7)
            target_sm_eef_p7 = wrench_pose7_to_sm_eef_pose7(target_wrench_p7)

            e1 = np.linalg.norm(target_wrench_p7[:3] - target_wrench_xyz)
            e2 = np.linalg.norm(target_sm_eef_p7[:3] - target_sm_gripper_xyz)
            return e1 + e2

        res = minimize(objective_function, X0, method='BFGS')
        # Rotation.from_euler(res.x[2:]).as_quat()
        pred_quat = Rotation.from_euler('xyz', res.x[3:]).as_quat()

        pred_pose = np.array([res.x[0], res.x[1], res.x[2], *pred_quat])
        return pred_pose, res.fun
    
    def dataloader(self, batch_size, num_workers, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_function)



if __name__ == "__main__":
    np.set_printoptions(precision=2)
    episode_ids = os.listdir('./data/demonstrations')
    D = RealRobotDataset('./data/demonstrations', episode_ids, offset_z=0.0)

    bounds = []

    for name in [
        'target_wrench',
        'target_refined_wrench',
        'curr_wrench',
        'curr_gripper',
        'target_gripper_traj'
    ]:
        if name in ['curr_gripper', 'curr_wrench']:
            _ = D.bound(name, exclude_reach=True)
        else:
            _ = D.bound(name)
        bounds += list(_)
        print(f'[BOUND] {name} => {_}')
    
    bounds = torch.stack(bounds)
    print('[FINAL BOUND] => ', bounds.min(dim=0).values, bounds.max(dim=0).values) # these bound are used to crop point clouds

    renderer = CubePointCloudRenderer("cuda:0", (420, 420), with_depth=True, cameras=['top', 'left', 'front']) 

    for i in tqdm(range(len(D)), desc='render testing'):
        sample = D[i]

        if 'target_wrench' in sample:
            target_wrench = sample['target_wrench']
            target_sm_gripper = sample['target_gripper']
            gt_pose = sample['target_eef_pose']
            D.to_target_full_gripper(target_wrench, target_sm_gripper, gt_pose=gt_pose)
            

        th_points = sample['points'].cuda()
        th_colors = sample['colors'].cuda()

        sub_pcd_in_cube, rev_trans = place_pc_in_cube(th_points, with_mean_or_bounds=False, scene_bounds=SCENE_BOUNDS)
        virtual_images = renderer(sub_pcd_in_cube.float(), torch.cat((sub_pcd_in_cube, th_colors), dim=-1).float()).unsqueeze(0)
        images = [virtual_images[0, 0, :, :, 3:6].permute(2, 0, 1),
          virtual_images[0, 1, :, :, 3:6].permute(2, 0, 1),
          virtual_images[0, 2, :, :, 3:6].permute(2, 0, 1)]
        pil_images = [to_pil_image(im) for im in images]
        
        def get_2d_pts(pts):
            target_pts, _ = place_pc_in_cube(th_points, pts.cuda().reshape(-1, 3),  with_mean_or_bounds=False, scene_bounds=SCENE_BOUNDS)
            target_2d_pts = renderer.points3d_to_screen2d(target_pts[None, ...])
            return target_2d_pts

        def draw_dots(dots, color):
            dot_radius = 3
            dots = dots.reshape(-1, 3, 2)
            for a in range(len(dots)):
                for b in range(3):
                    im = pil_images[b]
                    draw = ImageDraw.Draw(im)
                    x, y = dots[a, b].long().tolist() 
                    draw.ellipse((x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius), fill=color)
            return dots
            
        if sample['command'] == REACH:
            draw_dots(get_2d_pts(sample['target_wrench']), (255, 0, 0))
            draw_dots(get_2d_pts(sample['target_gripper']), (0, 255, 0))
        elif sample['command'] == ADJUST:
            dots1 = draw_dots(get_2d_pts(sample['curr_wrench']), (255, 0, 0))
            dots2 = draw_dots(get_2d_pts(sample['target_refined_wrench']), (0, 0, 255))
            tqdm.write(f'{dots1[0].flatten().cpu().numpy() - dots2[0].flatten().cpu().numpy()}')
        else:
            draw_dots(get_2d_pts(sample['curr_wrench']), (255, 0, 0))
            draw_dots(get_2d_pts(sample['target_gripper_traj']), (0, 255, 0))
        
        os.makedirs(f'data/vis', exist_ok=True)
        fig = im_concat(*pil_images)
        fig.save(f'data/vis/{sample["ep_id"]}-{sample["image_id"]}.png')
        
        
    print('===================== \n', D.direction_splits())