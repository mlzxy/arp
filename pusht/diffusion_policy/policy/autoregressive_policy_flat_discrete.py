from typing import Dict, Tuple, List, Union
import torchvision
from dataclasses import dataclass
from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
import arp


def augment(item):
    action = item['action']
    images = item['obs']['image'] # (bs, N, C, H, W)
    pos = item['obs']['agent_pos']
    coordinates = torch.cat([action, pos], dim=1)
    L = coordinates.size(1)
    Limg = images.size(1)
    bs, dev = len(action), action.device

    def rotate_keypoints(keypoints, center, angle_radians):
        """
        keypoints: (N, 2)
        center: (N/1, 2)
        angle_radians: (N, )
        """
        angle_radians = angle_radians[:, None]
        rotation_matrix = torch.cat([
            torch.cos(angle_radians), -torch.sin(angle_radians),
            torch.sin(angle_radians), torch.cos(angle_radians)
        ], dim=1).reshape(-1, 2, 2)
        translated_keypoints = keypoints - center
        rotated_keypoints = torch.bmm(translated_keypoints[:, None, :], rotation_matrix).flatten(1)
        rotated_keypoints += center
        return rotated_keypoints
    
    def affine_batch_images(images, radians, translations):
        """
        images: (N, C, H, W)
        radians: (N)
        translations: (N, 2) 
        """
        affine_matrices = torch.zeros((len(images), 2, 3), device=images.device)
        if radians is not None:
            cos_vals = torch.cos(radians)
            sin_vals = torch.sin(radians)
            affine_matrices[:, 0, 0] = cos_vals
            affine_matrices[:, 0, 1] = -sin_vals
            affine_matrices[:, 1, 0] = sin_vals
            affine_matrices[:, 1, 1] = cos_vals
            affine_matrices[:, 1, 1] = cos_vals
        else:
            affine_matrices[:, 0, 0] = 1
            affine_matrices[:, 1, 1] = 1

        if translations is not None:
            affine_matrices[:, 0, 2] = translations[:, 0]
            affine_matrices[:, 1, 2] = translations[:, 1]
        grid = F.affine_grid(affine_matrices, images.size(), align_corners=False)
        rotated_images = F.grid_sample(images, grid, align_corners=False, padding_mode="border", mode='bilinear')
        return rotated_images
        
    radians =  torch.pi * (0.5 - torch.rand(bs, device=dev))
    coordinates = rotate_keypoints(coordinates.flatten(0, 1), torch.as_tensor([256, 256], device=dev).reshape(1, 2), 
                                   radians[:, None].repeat(1, L).flatten())
    coordinates = coordinates.clamp_(0, 511)
    coordinates = coordinates.reshape(bs, L, 2)

    trans_offsets = coordinates, 511 - coordinates
    trans_offsets = trans_offsets[0].min(dim=1).values, trans_offsets[1].min(dim=1).values #(bs, 2)
    sign = torch.randint(0, 2, size=(bs,), device=dev)

    translations = 0.5 * trans_offsets[1] * sign[:, None] + 0.5 * trans_offsets[0] * (sign - 1)[:, None] #(bs, 2)
    coordinates = coordinates + translations[:, None, :]

    translations = -(translations / 256)

    images = affine_batch_images(images.flatten(0, 1), 
                                 radians[:, None].repeat(1, Limg).flatten(), None)
    images = affine_batch_images(images, None, translations[:, None, :].repeat(1, Limg, 1).flatten(0, 1))
    images = images.reshape(bs, Limg, *images.shape[1:])
    
    result = {
        'obs': {
            'image': images,
            'agent_pos': coordinates[:, :L//2, :]
        },
        'action': coordinates[:, L//2:, :]
    }
    return result



def segmented_range_list(start, end, segment_size):
    if (end - start) % segment_size != 0:
        end = start + int(math.ceil((end - start) / segment_size)) * segment_size
    multiple = (end - start) // segment_size
    lst = [start + i for i in range(multiple) for j in range(segment_size) ]
    return lst[:end - start]



class ARPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict, # has act_dim 
            horizon,  # act_horizon
            n_action_steps,
            n_obs_steps, # 2

            # image
            crop_shape=(76, 76),
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            
            # arch
            arp_cfg={}
):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            if attr['type'] == 'rgb':
                shape = attr['shape']
                obs_key_shapes[key] = list(shape)
                obs_config['rgb'].append(key)

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        obs_feature_dim = obs_encoder.output_shape()[0] # shall be 64

        self.obs_encoder = obs_encoder
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps

        #region arp ########################################################
        self.use_sample = arp_cfg.get('sample', True)
        self.augment_ratio = arp_cfg.get('augment_ratio', 0.0)
        self.action_chunk_size = arp_cfg.get('action_chunk_size', 1)

        self.policy = arp.AutoRegressivePolicy(arp.ModelConfig(
            n_embd=arp_cfg['n_embd'],
            embd_pdrop=arp_cfg['embd_pdrop'],
            layer_norm_every_block=arp_cfg.get('layer_norm_every_block', True),
            max_chunk_size=self.horizon * 2,
            max_seq_len=self.horizon * 2,
            layers=[arp.LayerType.make(
                    **arp_cfg['layer_cfg'],
                    condition_on='visual-token'
                )] * arp_cfg['num_layers'],
            tokens=[
                arp.TokenType.make(name='x', dim=1, is_continuous=True, embedding='discrete', dict_sizes=[100], bounds=[-1, 1], predictor='class'),
                arp.TokenType.make(name='y', dim=1, is_continuous=True, embedding='discrete', dict_sizes=[100], bounds=[-1, 1], predictor='class')
            ]
        ))

        self.obs_feat_linear = nn.Linear(obs_feature_dim, arp_cfg['n_embd'])
        #endregion ################################################################


    def get_optimizer(
            self, 
            transformer_weight_decay: float, 
            obs_encoder_weight_decay: float,
            lr: float, 
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = [{'params': self.policy.parameters(), 'weight_decay': transformer_weight_decay}]
        for m in [self.obs_encoder, self.obs_feat_linear]:
            optim_groups.append({
                "params": m.parameters(),
                "weight_decay": obs_encoder_weight_decay
            })
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
        return optimizer

    def compute_loss(self, batch):
        return self.predict_action(batch, training=True)
        
    def predict_action(self, batch_or_obs_dict, training=False) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        nobs_ = batch_or_obs_dict['obs'] if 'obs' in batch_or_obs_dict else batch_or_obs_dict
        batch_size = len(nobs_['image'])
        dev = nobs_['image'].device
        nobs = {k: v.clone() for k, v in nobs_.items()}

        if training and self.augment_ratio > 0.0:
            nobs['image'] = nobs['image'][:, :self.n_obs_steps]
            aug_num = int(batch_size * self.augment_ratio)
            item = augment({
                'obs': {
                    'image': nobs['image'][:aug_num],
                    'agent_pos': nobs['agent_pos'][:aug_num]    
                },
                'action': batch_or_obs_dict['action'][:aug_num]
            })
            nobs = {
                'image': torch.cat([item['obs']['image'], nobs['image'][aug_num:]], dim=0), 
                'agent_pos': torch.cat([item['obs']['agent_pos'], nobs['agent_pos'][aug_num:]], dim=0), 
            }
            batch_or_obs_dict = {
                'action': torch.cat([item['action'], batch_or_obs_dict['action'][aug_num:]], dim=0)
            }

        #region normalization and sequenize
        assert nobs['image'].max() <= 1.0 and nobs['image'].min() >= 0.0
        assert nobs['agent_pos'].max() <= 511.0 and nobs['agent_pos'].min() >= 0.0

        nobs['image'] -= 0.5
        nobs['image'] /= 0.5

        nobs['agent_pos'] -= 256
        nobs['agent_pos'] /= 256.

        if training:
            label_actions = batch_or_obs_dict['action'].clone()
            assert label_actions.max() <= 511.0 and label_actions.min() >= 0.0
            label_actions -= 256
            label_actions /= 256.
        else:
            label_actions = None
        #endregion ########################################

        future_tk_types = ['x', 'y'] * self.horizon      
        future_chk_ids = segmented_range_list(0, self.horizon * 2, self.action_chunk_size)
        future_tk_chk_ids = [{'tk_id': self.policy.token_name_2_ids[tk_type], 'chk_id': chk_id} for chk_id, tk_type in zip(future_chk_ids, future_tk_types)]       

        # region: feature preparation & model inference
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = self.obs_feat_linear(nobs_features)
        nobs_features = nobs_features.reshape(batch_size, self.n_obs_steps, self.policy.cfg.n_embd)

        if training:
            # NOTE sequence preparation
            tk_vals = label_actions.flatten(1).unsqueeze(-1)
            tk_names = future_tk_types
            tk_types = torch.as_tensor([self.policy.token_name_2_ids[tname] for tname in tk_names]).reshape(1, -1, 1).repeat(batch_size, 1, 1).to(dev)
            seq = torch.cat([tk_vals, tk_types], dim=-1)
            loss_dict = self.policy.compute_loss(seq, contexts={'visual-token': nobs_features}) # the 1 is for control bit
            return loss_dict
        else:
            # NOTE sequence preparation
            seq = torch.zeros([batch_size, 0, 2], device=dev)
            action_pred = self.policy.generate(seq, future_tk_chk_ids, contexts={'visual-token': nobs_features}, sample=self.use_sample)
            action_pred = action_pred[..., 0].reshape(-1, self.horizon, 2)

            # region de-normalization
            action_pred *= 256.
            action_pred += 256.
            action_pred.clamp_(0, 511)
            # endregion ##################

            start = self.n_obs_steps - 1
            end = start + self.n_action_steps # 1 + 8 = 9
            action = action_pred[:, start:end] # [56, 8, 2]
            result = {
                'action_pred': action_pred,
                'action': action
            }
            return result
        # endregion ########################################

        
        

