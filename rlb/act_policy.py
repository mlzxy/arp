from math import ceil
from copy import deepcopy
import wandb
from collections import defaultdict, ChainMap
from omegaconf import DictConfig
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat
from typing import Optional, Tuple
import torch

import torchvision
import numpy as np
import clip
from torch.cuda.amp import autocast, GradScaler
from scipy.spatial.transform import Rotation
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.optim import Lamb, GradualWarmupScheduler
from utils.structure import ActResult
from argparse import Namespace
import utils.math3d as math3d
from utils.clip import clip_encode_text
from PIL import Image
from preprocess import CubePointCloudRenderer, preprocess_images_in_batch, \
    flatten_img_pc_to_points, clamp_pc_in_bound, place_pc_in_cube, generate_heatmap_from_screen_pts, \
    apply_se3_augmentation, transform_pc, grid_sample_from_heatmap, add_uniform_noise, denorm_rgb

from utils.layers import (
    Conv2DBlock,
    Conv2DUpsampleBlock,
    PreNorm,
    Attention,
    DenseBlock,
    FeedForward,
    FixedPositionalEncoding
)
import math
import utils.act as act
from autoregressive_policy import Policy, MultiViewTransformer
from arp import ConvUpsample


def embed_position_2d(positions, n_embd, N=10000):
    d_model = n_embd // 2 
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(N) / d_model))[None, :].to(positions.device)
    pe = torch.zeros(positions.size(0), n_embd, device=positions.device) 
    pe[:, 0:d_model:2] = torch.sin(positions[:, :1] * div_term)
    pe[:, 1:d_model:2] = torch.cos(positions[:, :1] * div_term)       
    pe[:, d_model: :2] = torch.sin(positions[:, 1:] * div_term)
    pe[:, d_model: :2] = torch.cos(positions[:, 1:] * div_term)       
    return pe


class PolicyNetwork(nn.Module):
    def __init__(self, model_cfg, env_cfg, render_device):
        super().__init__()
        self._num_rotation_classes = model_cfg.num_rotation_classes
        self._rotation_resolution = 360 / self._num_rotation_classes
        self._image_resolution = [env_cfg.image_size, env_cfg.image_size]
        self._transform_augmentation = model_cfg.transform_augmentation
        self._place_with_mean = model_cfg.place_with_mean
        self._transform_augmentation_xyz = torch.from_numpy(np.array(model_cfg.transform_augmentation_xyz))
        self._transform_augmentation_rpy = model_cfg.transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = self._rotation_resolution

        self.gt_hm_sigma = model_cfg.gt_hm_sigma
        self.add_rgc_loss = model_cfg.add_rgc_loss
        self.amp = model_cfg.amp

        self.scene_bounds = env_cfg.scene_bounds
        self.cameras = env_cfg.cameras
        self.move_pc_in_bound = model_cfg.move_pc_in_bound

        self.rotation_aug = model_cfg.rotation_aug # 2
        self.stage2_zoom_scale = model_cfg.stage2_zoom_scale # st_sca
        self.stage2_waypoint_label_noise = model_cfg.stage2_waypoint_label_noise # st_wpt_loc_aug
        self.point_augment_noise = model_cfg.point_augment_noise # img_aug_2

        self.num_all_rot = self._num_rotation_classes * 3
        self.proprio_dim = model_cfg.proprio_dim
        self.img_size = model_cfg.img_size
        self.img_patch_size = model_cfg.img_patch_size
        self.renderer = CubePointCloudRenderer(render_device, (model_cfg.img_size, model_cfg.img_size), with_depth=model_cfg.add_depth, cameras=model_cfg.mvt_cameras)
        self.num_cameras = len(model_cfg.mvt_cameras)
        if model_cfg.render_with_cpp:
            assert model_cfg.mvt_cameras == ['top', 'left', 'front']
            self.render_with_cpp = True
            from point_renderer.rvt_renderer import RVTBoxRenderer
            self.cpp_renderer = RVTBoxRenderer(device=render_device,
                                               img_size=(model_cfg.img_size, model_cfg.img_size),
                                               three_views=True,
                                               with_depth=model_cfg.add_depth)
        else:
            self.render_with_cpp = False

        self.mvt1 = MultiViewTransformer(model_cfg, renderer=self.renderer)
        self.mvt2 = MultiViewTransformer(model_cfg, renderer=self.renderer)

        dim_model = model_cfg.im_channels * 2
        layer_kwargs = dict(n_heads=4, dropout=0.1, dim_feedforward=4 * dim_model, feedforward_activation="relu")
        self.latent_dim = 64

        self.stage1_vae_encoder_cls_embed = nn.Embedding(1, dim_model)
        self.stage1_vae_encoder_pos_embed = nn.Embedding(3, dim_model) # three views       
        self.stage1_vae_encoder = act.ACTEncoder(dim_model, 4, pre_norm=False, layer_kwargs=layer_kwargs)
        self.stage1_vae_encoder_latent_output_proj = nn.Linear(dim_model, self.latent_dim * 2)
        self.stage1_latent_to_embed = nn.Linear(self.latent_dim, dim_model) 
        self.stage1_encoder = act.ACTEncoder(dim_model, 4, pre_norm=False, layer_kwargs=layer_kwargs)
        self.stage1_decoder = act.ACTDecoder(dim_model, 4, pre_norm=False, layer_kwargs=layer_kwargs)
        self.stage1_decoder_input_tokens = nn.Embedding(3, dim_model) 
        self.stage1_up0 = ConvUpsample(dim_model, 1, self.img_patch_size)

        self.stage2_vae_encoder_cls_embed = nn.Embedding(1, dim_model)
        self.stage2_vae_encoder_pos_embed = nn.Embedding(3 + 3 + 2, dim_model) # three views       
        self.stage2_vae_encoder_rot_embeds = nn.ModuleList([nn.Embedding(72, dim_model) for _ in range(3)])
        self.stage2_vae_encoder_gripper_embeds = nn.ModuleList([nn.Embedding(2, dim_model) for _ in range(2)])
        self.stage2_vae_encoder = act.ACTEncoder(dim_model, 4, pre_norm=False, layer_kwargs=layer_kwargs)
        self.stage2_vae_encoder_latent_output_proj = nn.Linear(dim_model, self.latent_dim * 2)
        self.stage2_latent_to_embed = nn.Linear(self.latent_dim, dim_model) 
        self.stage2_encoder = act.ACTEncoder(dim_model, 4, pre_norm=False, layer_kwargs=layer_kwargs)
        self.stage2_decoder = act.ACTDecoder(dim_model, 4, pre_norm=False, layer_kwargs=layer_kwargs)
        self.stage2_decoder_input_tokens = nn.Embedding(3 + 3 + 2, dim_model)  
        self.stage2_up0 = ConvUpsample(dim_model, 1, self.img_patch_size)
        self.stage2_rot_fc = nn.ModuleList([nn.Linear(dim_model, 72) for _ in range(3)])
        self.stage2_gripper_fc = nn.ModuleList([nn.Linear(dim_model, 2) for _ in range(3)])

        self.cfg = model_cfg

    
    def multi_view_coordinate_sampler(self, lst_of_spatial_logits):
        hm_logits = torch.cat([a for a in lst_of_spatial_logits], dim=1)
        hm = F.softmax(hm_logits.flatten(2), dim=2)
        bs = len(hm_logits)
        hm = hm.view(bs, 3, 224, 224)
        pred_pt = [self.renderer.get_most_likely_point_3d(hm[i : i + 1]) for i in range(bs)]
        spatial_point = torch.cat(pred_pt, 0) # bs, 3
        screen_points = self.renderer.points3d_to_screen2d(spatial_point[:, None, :])
        screen_points = screen_points[:, 0]
        return spatial_point, screen_points
    
    def get_gt_rot_grip_collision(
        self,
        batch_size,
        action_rot,
        action_grip,
        action_ignore_collisions,
        device,
    ):
        """
        :param batch_size: int
        :param action_rot: np.array of shape (bs, 4), quternion xyzw format
        :param action_grip: torch.tensor of shape (bs)
        :param action_ignore_collisions: torch.tensor of shape (bs)
        :param device:
        """
        bs = batch_size
        assert action_rot.shape == (bs, 4)
        assert action_grip.shape == (bs,), (action_grip, bs)

        action_rot_x_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_y_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_z_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_grip_one_hot = torch.zeros((bs, 2), dtype=int, device=device)
        action_collision_one_hot = torch.zeros((bs, 2), dtype=int, device=device)

        # fill one-hots
        for b in range(bs):
            gt_rot = action_rot[b]
            gt_rot = math3d.quaternion_to_discrete_euler(
                gt_rot, self._rotation_resolution
            )
            action_rot_x_one_hot[b, gt_rot[0]] = 1
            action_rot_y_one_hot[b, gt_rot[1]] = 1
            action_rot_z_one_hot[b, gt_rot[2]] = 1

            # grip
            gt_grip = action_grip[b]
            action_grip_one_hot[b, gt_grip] = 1

            # ignore collision (to one hot, if result = 0, then don't ignore collision)
            gt_ignore_collisions = action_ignore_collisions[b, :]
            action_collision_one_hot[b, gt_ignore_collisions[0]] = 1

        return (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,
            action_collision_one_hot,
        )

    def get_gt_translation_action(
        self,
        waypoint, # this is groundtruth 3d point
        dims,
    ): # note: will be called separately for stage 1 / 2
        bs, nc, h, w = dims
        wpt_img = self.renderer.points3d_to_screen2d(waypoint.unsqueeze(1))
        assert wpt_img.shape[1] == 1
        wpt_img = wpt_img.squeeze(1)  # (bs, num_img, 2)
        action_trans = generate_heatmap_from_screen_pts(
            wpt_img.reshape(-1, 2), #! just the winning points
            (h, w),
            sigma=self.gt_hm_sigma,
            thres_sigma_times=3,
        )
        action_trans = action_trans.view(bs, nc, h * w).transpose(1, 2).clone()
        return action_trans, wpt_img

    def render(self, pc, img_feat, mvt: MultiViewTransformer):
        renderer = self.cpp_renderer if self.render_with_cpp else self.renderer
        with torch.no_grad():
            with autocast(enabled=False):
                if mvt.add_corr:
                    if mvt.norm_corr:
                        img = []
                        for _pc, _img_feat in zip(pc, img_feat):
                            max_pc = 1.0 if len(_pc) == 0 else torch.max(torch.abs(_pc))
                            img.append(
                                renderer(_pc, torch.cat((_pc / max_pc, _img_feat), dim=-1)).unsqueeze(0) # [3, 224, 224, 7], 3 -> views, 7 -> feats
                            )
                    else:
                        img = [renderer(_pc, torch.cat((_pc, _img_feat), dim=-1)).unsqueeze(0) for _pc, _img_feat in zip(pc, img_feat)]
                else:
                    img = [renderer(_pc, _img_feat).unsqueeze(0) for _pc, _img_feat in zip(pc, img_feat)]

        img = torch.cat(img, 0)
        img = img.permute(0, 1, 4, 2, 3) # [1, 3, 7, 224, 224]

        if mvt.add_pixel_loc:
            bs = img.shape[0]
            pixel_loc = mvt.pixel_loc.to(img.device) # extra feature
            img = torch.cat(
                (img, pixel_loc.unsqueeze(0).repeat(bs, 1, 1, 1, 1)), dim=2
            )
        return img

    def forward(self, observation):
        loss_dicts = []
        nc, h, w = len(self.cfg.mvt_cameras), self.img_size, self.img_size
        dev = observation["lang_goal_embs"].device
        if self.training:
            action_grip = observation["gripper_action"].int() # (b,) of int
            action_ignore_collisions = observation["ignore_collisions"].view(-1, 1).int()  # (b, 1) of int
            action_gripper_pose = observation["gripper_pose"]  # (b, 7)
            action_trans_con = action_gripper_pose[:, 0:3]  # (b, 3), translation in xyz
            action_rot = action_gripper_pose[:, 3:7]  # (b, 4), rotation in quaternion xyzw

        lang_goal_embs = observation["lang_goal_embs"].float()
        proprio = observation["low_dim_state"]

        #region preprocess and augmentation

        obs, pcd = preprocess_images_in_batch(observation, self.cameras)
        pc, img_feat = flatten_img_pc_to_points(obs, pcd)

        with torch.no_grad():
            if self._transform_augmentation and self.training:
                action_trans_con, action_rot, pc = apply_se3_augmentation( #! where the gt really comes out (for SE3 trans)
                    pcd=pc,
                    action_gripper_pose=action_gripper_pose,
                    bounds=torch.tensor(self.scene_bounds),
                    trans_aug_range=torch.tensor(self._transform_augmentation_xyz),
                    rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                )
                action_trans_con = torch.tensor(action_trans_con).to(pc.device)
                action_rot = torch.tensor(action_rot).to(pc.device)
                action_rot = action_rot.cpu().numpy()
                for i, _action_rot in enumerate(action_rot):
                    _action_rot = math3d.normalize_quaternion(_action_rot)
                    if _action_rot[-1] < 0:
                        _action_rot = -_action_rot
                    action_rot[i] = _action_rot

        pc, img_feat = clamp_pc_in_bound(pc, img_feat, self.scene_bounds, skip=not self.move_pc_in_bound)
        pc_new, rev_trans_stage1, waypoint_stage1 = [], [], []

        for i, _pc in enumerate(pc):
            a, b = place_pc_in_cube(_pc,
                with_mean_or_bounds=self._place_with_mean,
                scene_bounds=None if self._place_with_mean else self.scene_bounds,
            )
            if self.training:
                waypoint_stage1.append(place_pc_in_cube(_pc, action_trans_con[i][:3],
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0].unsqueeze(0))
            pc_new.append(a)
            rev_trans_stage1.append(b)
        pc = pc_new
        bs = len(pc)

        if self.training:
            waypoint_stage1 = torch.cat(waypoint_stage1, axis=0).clone().detach()
            if self.point_augment_noise != 0:
                with torch.no_grad():
                    for x in img_feat:
                        stdv = self.point_augment_noise * torch.rand(1, device=x.device)
                        noise = stdv * ((2 * torch.rand(*x.shape, device=x.device)) - 1)
                        x = x + noise

        img = self.render(pc, img_feat, self.mvt1)
        #endregion ###########################
        
        compute_ce_loss = nn.CrossEntropyLoss()

        visual_featmap_1 = self.mvt1(img=img, proprio=proprio, lang_emb=lang_goal_embs) # [B, num_cameras, 128, np, np]
        visual_tokens_1 = visual_featmap_1.permute(0, 1, 3, 4, 2).flatten(1, 3)

        if self.training:
            smooth_spatial_label_stage1, screen_waypoint_stage1 = self.get_gt_translation_action(waypoint_stage1, dims=(bs, nc, h, w))
            tmp_loss_dict = {}
            input_embeds = embed_position_2d(screen_waypoint_stage1.flatten(0, 1), 128).reshape(bs, 3, -1) # bs, nc, 128
            input_embeds += self.stage1_vae_encoder_pos_embed.weight[None, ...]
            input_embeds = torch.cat([
                self.stage1_vae_encoder_cls_embed.weight[None, ...].repeat(bs, 1, 1),
                input_embeds ], dim=1)
            input_embeds = input_embeds.permute(1, 0, 2) # nc, bs, 128
            cls_token_out = self.stage1_vae_encoder(input_embeds)[0]
            latent_pdf_params = self.stage1_vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu) 
            tmp_loss_dict['kld_loss'] = act.compute_kld_loss(mu, log_sigma_x2)
        else:
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(dev)

        latent_embed = self.stage1_latent_to_embed(latent_sample)
        tokens = torch.cat([latent_embed[:, None], visual_tokens_1], dim=1)
        encoder_out = self.stage1_encoder(tokens.permute(1, 0, 2))
        decoder_in = self.stage1_decoder_input_tokens.weight[:, None, :].repeat(1, bs, 1)
        decoder_out = self.stage1_decoder(decoder_in, encoder_out)
        decoder_out = decoder_out.transpose(0, 1) # bs, nc, 128

        spatial_logits_buffer = []
        spatial_losses = []
        for i in range(3):
            embs = decoder_out[:, i].reshape(bs, 128, 1, 1)
            corr = embs * visual_featmap_1[:, i]
            hm_logits = self.stage1_up0(corr)
            spatial_logits_buffer.append(hm_logits)
            if self.training:
                smooth_heatmap = smooth_spatial_label_stage1[:, :, i]
                spatial_losses.append(compute_ce_loss(hm_logits.flatten(1), smooth_heatmap.flatten(1)))
        
        if self.training:
            tmp_loss_dict['stage1_trans_loss'] = sum(spatial_losses) / len(spatial_losses)
            loss_dicts.append(tmp_loss_dict)
        else:
            hms = torch.cat([F.softmax(hm_logits.reshape(bs, -1), dim=1).reshape(bs, 1, 224, 224) 
                             for hm_logits in spatial_logits_buffer], dim=1)
            pred_pt = [self.renderer.get_most_likely_point_3d(hms[i : i + 1]) for i in range(bs)]
            waypoint_stage1 = torch.cat(pred_pt, 0) # bs, 3
  
        with torch.no_grad():
            if self.training:
                waypoint_stage1_noisy = add_uniform_noise(
                    waypoint_stage1.clone().detach(), 2 * self.stage2_waypoint_label_noise
                )
                pc, rev_trans_stage2 = transform_pc(pc, loc=waypoint_stage1_noisy, sca=self.stage2_zoom_scale)
                waypoint_stage2, _ = transform_pc(waypoint_stage1, loc=waypoint_stage1_noisy, sca=self.stage2_zoom_scale)
            else:
                pc, rev_trans_stage2 = transform_pc(pc, loc=waypoint_stage1, sca=self.stage2_zoom_scale)
                waypoint_stage1_noisy = waypoint_stage1
                waypoint_stage2 = None

        img = self.render(pc, img_feat, self.mvt2)
        visual_featmap_2 = self.mvt2(img=img, proprio=proprio, lang_emb=lang_goal_embs)
        visual_tokens_2 = visual_featmap_2.permute(0, 1, 3, 4, 2).flatten(1, 3)

        if self.training:
            (
                action_rot_x,
                action_rot_y,
                action_rot_z,
                action_grip,       # (bs)
                action_collision,  # (bs)
            ) = [v.argmax(-1) for v in self.get_gt_rot_grip_collision(bs, action_rot, action_grip, action_ignore_collisions, device=dev)]
            smooth_spatial_label_stage2, screen_waypoint_stage2 = self.get_gt_translation_action(waypoint_stage2, dims=(bs, nc, h, w))

            tmp_loss_dict = {}
            input_embeds = embed_position_2d(screen_waypoint_stage2.flatten(0, 1), 128).reshape(bs, 3, -1)
            input_embeds = torch.cat([input_embeds, 
                                    *[layer(rot).unsqueeze(1) for rot, layer in zip([action_rot_x, action_rot_y, action_rot_z], self.stage2_vae_encoder_rot_embeds)],
                                    *[layer(grip).unsqueeze(1) for grip, layer in zip([action_grip, action_collision], self.stage2_vae_encoder_gripper_embeds)]
                                ], dim=1)
            input_embeds += self.stage2_vae_encoder_pos_embed.weight[None, ...]
            input_embeds = torch.cat([
                self.stage2_vae_encoder_cls_embed.weight[None, ...].repeat(bs, 1, 1),
                input_embeds ], dim=1)

            input_embeds = input_embeds.permute(1, 0, 2) # nc, bs, 128
            cls_token_out = self.stage2_vae_encoder(input_embeds)[0]
            latent_pdf_params = self.stage2_vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu) 
            tmp_loss_dict['stage2_kld_loss'] = act.compute_kld_loss(mu, log_sigma_x2)
        else:
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(dev)
        
        
        latent_embed = self.stage2_latent_to_embed(latent_sample)
        tokens = torch.cat([latent_embed[:, None], visual_tokens_2], dim=1)
        encoder_out = self.stage2_encoder(tokens.permute(1, 0, 2))
        decoder_in = self.stage2_decoder_input_tokens.weight[:, None, :].repeat(1, bs, 1)
        decoder_out = self.stage2_decoder(decoder_in, encoder_out)
        decoder_out = decoder_out.transpose(0, 1) # bs, nc, 128

        spatial_logits_buffer = []
        spatial_losses = []
        for i in range(3):
            embs = decoder_out[:, i].reshape(bs, 128, 1, 1)
            corr = embs * visual_featmap_2[:, i]
            hm_logits = self.stage2_up0(corr)
            spatial_logits_buffer.append(hm_logits)
            if self.training:
                smooth_heatmap = smooth_spatial_label_stage2[:, :, i]
                spatial_losses.append(compute_ce_loss(hm_logits.flatten(1), smooth_heatmap.flatten(1)))

        pred_rot_x, pred_rot_y, pred_rot_z = [self.stage2_rot_fc[i](decoder_out[:, i + 3]) for i in range(3)]
        pred_grip, pred_collision = [self.stage2_gripper_fc[i](decoder_out[:, i + 6]) for i in range(2)]

        if self.training:
            tmp_loss_dict['stage2_trans_loss'] = sum(spatial_losses) / len(spatial_losses)
            tmp_loss_dict['rot_x_loss'] = compute_ce_loss(pred_rot_x, action_rot_x) 
            tmp_loss_dict['rot_y_loss'] = compute_ce_loss(pred_rot_y, action_rot_y)
            tmp_loss_dict['rot_z_loss'] = compute_ce_loss(pred_rot_z, action_rot_z)
            tmp_loss_dict['grip_loss'] = compute_ce_loss(pred_grip, action_grip)
            tmp_loss_dict['collision_loss'] = compute_ce_loss(pred_collision, action_collision)
            loss_dicts.append(tmp_loss_dict)     
        else:
            pred_rot = torch.cat((
                pred_rot_x.argmax(1, keepdim=True),
                pred_rot_y.argmax(1, keepdim=True),
                pred_rot_z.argmax(1, keepdim=True),
            ), dim=-1)
            pred_grip = pred_grip.argmax(1, keepdim=True)
            pred_coll = pred_collision.argmax(1, keepdim=True) 

            hms = torch.cat([F.softmax(hm_logits.reshape(bs, -1), dim=1).reshape(bs, 1, 224, 224) 
                             for hm_logits in spatial_logits_buffer], dim=1)
            pred_pt = [self.renderer.get_most_likely_point_3d(hms[i : i + 1]) for i in range(bs)]
            waypoint_stage2 = torch.cat(pred_pt, 0) 

        if self.training:
            loss_dict = {}
            for d in loss_dicts: loss_dict.update(d)
            norm = lambda x: torch.norm(x.flatten(1), dim=1).mean().item()
            loss_dict['stat_dict'] = {
                    'v1_norm': norm(visual_featmap_1.flatten(0, 1)),
                    'v2_norm': norm(visual_featmap_2.flatten(0, 1))
            }
            return loss_dict
        else:
            final_waypoint = rev_trans_stage1[0](rev_trans_stage2(waypoint_stage2))
            pred_rot_quat = math3d.discrete_euler_to_quaternion(pred_rot.cpu().numpy(), self._rotation_resolution)
            continuous_action = np.concatenate(
                (
                    final_waypoint[0].cpu().numpy(),
                    pred_rot_quat[0],
                    pred_grip[0].cpu().numpy(),
                    pred_coll[0].cpu().numpy(),
                )
            )
            return continuous_action
        
