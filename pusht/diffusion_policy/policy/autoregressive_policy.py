from typing import Dict, Tuple, List, Union
import torchvision
from PIL import Image
from dataclasses import dataclass
from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import numpy as np
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
from torchvision.transforms.functional import to_pil_image

def denorm_rgb(img):
    return (img + 1) / 2

def to_red_heatmap(heatmap, normalize=True):
    if normalize:
        heatmap = heatmap / heatmap.max()
    heatmap = torch.cat((heatmap[:, None, : ,:], torch.zeros(len(heatmap), 2, *heatmap.shape[1:], device=heatmap.device)), dim=1)
    return heatmap

def generate_heatmap_from_screen_pts(pt, res, sigma, thres_sigma_times=3):
    """
    Pytorch code to generate heatmaps from point. Points with values less than
    thres are made 0
    :type pt: torch.FloatTensor of size (num_pt, 2)
    :type res: int or (int, int)
    :param sigma: the std of the gaussian distribition. if it is -1, we
        generate a hm with one hot vector
    :type sigma: float
    :type thres: float
    """
    num_pt, x = pt.shape
    assert x == 2
    assert sigma > 0

    if isinstance(res, int):
        resx = resy = res
    else:
        resx, resy = res

    _hmx = torch.arange(0, resy).to(pt.device)
    _hmx = _hmx.view([1, resy]).repeat(resx, 1).view([resx, resy, 1])
    _hmy = torch.arange(0, resx).to(pt.device)
    _hmy = _hmy.view([resx, 1]).repeat(1, resy).view([resx, resy, 1])
    hm = torch.cat([_hmx, _hmy], dim=-1)
    hm = hm.view([1, resx, resy, 2]).repeat(num_pt, 1, 1, 1) # one HxW heatmap for each point?

    pt = pt.view([num_pt, 1, 1, 2])
    hm = torch.exp(-1 * torch.sum((hm - pt) ** 2, -1) / (2 * (sigma**2))) # RBF Kernel
    thres = np.exp(-1 * (thres_sigma_times**2) / 2) # truncated
    hm[hm < thres] = 0.0

    hm /= torch.sum(hm, (1, 2), keepdim=True) + 1e-6 # normalization
    return hm # (n_pt, h, w)

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
        self.plan_steps = arp_cfg.get('plan_steps', 5)
        self.plan_dict_size = arp_cfg.get('plan_dict_size', 100)
        self.plan_upscale_ratio = arp_cfg.get('plan_upscale_ratio', 8)
        self.low_var_eval = arp_cfg.get('low_var_eval', True)
        self.num_latents = arp_cfg.get('num_latents', 1)
        self.reverse_plan = arp_cfg.get('reverse_plan', False)

        self.plan_chunk_size = arp_cfg.get('plan_chunk_size', 1)
        self.action_chunk_size = arp_cfg.get('action_chunk_size', horizon)

        image_shape = obs_shape_meta['image']['shape']
        self.image_shape = image_shape
        plan_attn_size = image_shape[-1] // self.plan_upscale_ratio

        self.policy = arp.AutoRegressivePolicy(arp.ModelConfig(
            n_embd=arp_cfg['n_embd'],
            embd_pdrop=arp_cfg['embd_pdrop'],
            layer_norm_every_block=arp_cfg.get('layer_norm_every_block', True),
            max_chunk_size=horizon if self.plan_chunk_size > 0 else (horizon + self.plan_steps),
            max_seq_len=(self.n_obs_steps + self.plan_steps + self.horizon) * 2,
            layers=[arp.LayerType.make(
                    **arp_cfg['layer_cfg'],
                    condition_on='visual-token'
                )] * arp_cfg['num_layers'],
            tokens=[
                arp.TokenType.make(name='pos', dim=2, is_continuous=True, embedding='linear', is_control=True),

                arp.TokenType.make(name='coarse-plan', is_continuous=True, dim=2, 
                                embedding='position_2d', predictor="upsample_from_2d_attn", 
                                predictor_kwargs={'attn_with': (arp_cfg['n_embd'], plan_attn_size, plan_attn_size), 
                                                  'upscale_ratio': self.plan_upscale_ratio, 'label_name': 'smooth-heatmap', 
                                                  'corr_dim': arp_cfg.get('plan_corr_dim', -1)}),

                arp.TokenType.make(name='fine-action', dim=2, is_continuous=True, embedding='linear', predictor='gmm', predictor_kwargs={'num_latents': self.num_latents, 'low_var_eval': self.low_var_eval}),
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

        future_tk_types = ['coarse-plan'] * self.plan_steps + ['fine-action'] * self.horizon      

        if self.plan_chunk_size > 0:
            plan_chk_ids = segmented_range_list(self.n_obs_steps, self.n_obs_steps + self.plan_steps, self.plan_chunk_size)
            action_chk_ids = segmented_range_list(max(plan_chk_ids) + 1, max(plan_chk_ids) + 1 + self.horizon, self.action_chunk_size)
            future_chk_ids = plan_chk_ids + action_chk_ids
        else:
            future_chk_ids = [self.n_obs_steps] * (self.plan_steps + self.horizon)
        future_tk_chk_ids = [{'tk_id': self.policy.token_name_2_ids[tk_type], 'chk_id': chk_id} for tk_type, chk_id in zip(future_tk_types, future_chk_ids)]

        # region: feature preparation & model inference
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = self.obs_feat_linear(nobs_features)
        nobs_features = nobs_features.reshape(batch_size, self.n_obs_steps, self.policy.cfg.n_embd)

        if training:
            # NOTE sequence preparation
            action_BCL = label_actions.permute(0, 2, 1) # (bs, 2, L)
            coarse_plan_BCL = F.interpolate(action_BCL, size=self.plan_steps, mode='linear', align_corners=self.plan_steps >= 3)
            coarse_plans = coarse_plan_BCL.permute(0, 2, 1) # (bs, L, 2)
            if self.reverse_plan:
                coarse_plans = torch.flip(coarse_plans, [1])
            
            half_img_size = self.image_shape[1] // 2
            coarse_plans *= half_img_size
            coarse_plans += half_img_size
            coarse_plans.clamp_(0, 2 * half_img_size - 1)
            coarse_plans.round_()

            smooth_heatmap = generate_heatmap_from_screen_pts(coarse_plans.flatten(0, 1), self.image_shape[1:], 
                                sigma=1, thres_sigma_times=3).reshape(batch_size, coarse_plans.size(1), *self.image_shape[1:])
                
            # NOTE: visualize        
            # VIS_ID = 0
            # hm_img = to_pil_image(to_red_heatmap(smooth_heatmap[VIS_ID]).sum(dim=0))
            # img = to_pil_image(denorm_rgb(nobs['image'][VIS_ID,1]))
            # Image.blend(img, hm_img, 0.5).save('./outputs/test.jpg')


            tk_vals = torch.cat([nobs['agent_pos'][:, :self.n_obs_steps], coarse_plans, label_actions], dim=1)
            tk_names = ['pos'] * self.n_obs_steps + future_tk_types
            tk_types = torch.as_tensor([self.policy.token_name_2_ids[tname] for tname in tk_names]).reshape(1, -1, 1).repeat(batch_size, 1, 1).to(dev)
            seq = torch.cat([tk_vals, tk_types], dim=-1)

            loss_dict = self.policy.compute_loss(seq, chk_ids=torch.as_tensor(list(range(self.n_obs_steps)) + future_chk_ids).to(dev),
                                                 contexts={'visual-token': nobs_features, 'smooth-heatmap': smooth_heatmap.flatten(0, 1)}) # the 1 is for control bit
            return loss_dict
        else:
            # NOTE sequence preparation
            tk_vals = nobs['agent_pos'][:, :self.n_obs_steps]
            tk_names = ['pos'] * self.n_obs_steps
            tk_types = torch.as_tensor([self.policy.token_name_2_ids[tname] for tname in tk_names]).reshape(1, -1, 1).repeat(batch_size, 1, 1).to(dev)
            seq = torch.cat([tk_vals, tk_types], dim=-1)

            action_pred = self.policy.generate(seq, future_tk_chk_ids, contexts={'visual-token': nobs_features}, sample=self.use_sample)
            start = seq.size(1) + self.plan_steps
            action_pred = action_pred[:, seq.size(1) + self.plan_steps:, :-1] 

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

        
        

