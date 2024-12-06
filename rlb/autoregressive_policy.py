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

from arp import AutoRegressivePolicy, TokenType, LayerType, ModelConfig


class MultiViewTransformer(nn.Module):
    def __init__(self, cfg: DictConfig, renderer: Optional[CubePointCloudRenderer]=None):
        super().__init__()
        self.depth = cfg.depth
        self.img_feat_dim = cfg.img_feat_dim
        self.img_size = cfg.img_size
        self.add_proprio = cfg.add_proprio
        self.proprio_dim = cfg.proprio_dim
        self.add_lang = cfg.add_lang
        self.lang_dim = cfg.lang_dim
        self.lang_len = cfg.lang_len
        self.im_channels = cfg.im_channels # 64
        self.img_patch_size = cfg.img_patch_size
        self.attn_dropout = cfg.attn_dropout
        self.add_corr = cfg.add_corr
        self.add_pixel_loc = cfg.add_pixel_loc
        self.add_depth = cfg.add_depth
        self.pe_fix = cfg.pe_fix
        self.attn_dim = cfg.attn_dim
        self.attn_heads = cfg.attn_heads
        self.attn_dim_head = cfg.attn_dim_head
        self.attn_dropout = cfg.attn_dropout
        self.use_xformers = cfg.use_xformers
        self.feat_dim = cfg.feat_dim
        activation = "lrelu"

        self.norm_corr = cfg.norm_corr
        self.num_rot = cfg.num_rotation_classes
        print(f"MVT Vars: {vars(self)}")

        assert not renderer is None
        self.renderer = renderer
        self.num_cameras = self.renderer.num_cameras

        # patchified input dimensions
        spatial_size = self.img_size // self.img_patch_size  # 16

        if self.add_proprio:
            # 64 img features + 64 proprio features
            self.input_dim_before_seq = self.im_channels * 2
        else:
            self.input_dim_before_seq = self.im_channels

        # learnable positional encoding
        if self.add_lang:
            lang_emb_dim, lang_max_seq_len = self.lang_dim, self.lang_len
        else:
            lang_emb_dim, lang_max_seq_len = 0, 0
        self.lang_emb_dim = lang_emb_dim
        self.lang_max_seq_len = lang_max_seq_len

        if self.pe_fix:
            num_pe_token = spatial_size**2 * self.num_cameras
        else:
            num_pe_token = lang_max_seq_len + (spatial_size**2 * self.num_cameras)

        self.pos_encoding = nn.Parameter(torch.randn(1, num_pe_token, self.input_dim_before_seq))

        inp_img_feat_dim = self.img_feat_dim
        if self.add_corr:
            inp_img_feat_dim += 3
        if self.add_pixel_loc:
            inp_img_feat_dim += 3
            self.pixel_loc = torch.zeros((self.num_cameras, 3, self.img_size, self.img_size))
            self.pixel_loc[:, 0, :, :] = torch.linspace(-1, 1, self.num_cameras).unsqueeze(-1).unsqueeze(-1)
            self.pixel_loc[:, 1, :, :] = torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(-1)
            self.pixel_loc[:, 2, :, :] = torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(0) # [3, 3, 224, 224]

        if self.add_depth:
            inp_img_feat_dim += 1

        if self.add_proprio:
            # proprio preprocessing encoder
            self.proprio_preprocess = DenseBlock(
                self.proprio_dim,
                self.im_channels,
                norm="group",
                activation=activation,
            )

        self.patchify = Conv2DBlock(
            inp_img_feat_dim,
            self.im_channels,
            kernel_sizes=self.img_patch_size,
            strides=self.img_patch_size,
            norm="group",
            activation=activation,
            padding=0,
        )

        # lang preprocess
        if self.add_lang:
            self.lang_preprocess = DenseBlock(
                lang_emb_dim,
                self.im_channels * 2,
                norm="group",
                activation=activation,
            )

        self.fc_bef_attn = DenseBlock(
            self.input_dim_before_seq,
            self.attn_dim,
            norm=None,
            activation=None,
        )
        self.fc_aft_attn = DenseBlock(
            self.attn_dim,
            self.input_dim_before_seq,
            norm=None,
            activation=None,
        )

        get_attn_attn = lambda: PreNorm(
            self.attn_dim,
            Attention(
                self.attn_dim,
                heads=self.attn_heads,
                dim_head=self.attn_dim_head,
                dropout=self.attn_dropout,
                use_fast=self.use_xformers,
            ),
        )
        get_attn_ff = lambda: PreNorm(self.attn_dim, FeedForward(self.attn_dim))
        # self-attention layers
        self.layers = nn.ModuleList([])
        attn_depth = self.depth

        for _ in range(attn_depth):
            self.layers.append(nn.ModuleList([get_attn_attn(), get_attn_ff()]))

        # self.view_embedding = nn.Embedding(self.num_cameras, self.input_dim_before_seq)


    def forward(
        self,
        img,
        proprio=None,
        lang_emb=None
    ):
        """
        :param img: tensor of shape (bs, num_cameras, img_feat_dim, h, w)
        :param proprio: tensor of shape (bs, priprio_dim)
        :param lang_emb: tensor of shape (bs, lang_len, lang_dim)
        """

        bs, num_cameras, img_feat_dim, h, w = img.shape
        num_pat_img = h // self.img_patch_size
        assert num_cameras == self.num_cameras
        # assert img_feat_dim == self.img_feat_dim
        assert h == w == self.img_size

        img = img.view(bs * num_cameras, img_feat_dim, h, w)
        # preprocess
        # (bs * num_img, im_channels, h, w) ->
        # (bs * num_img, im_channels, h / img_patch_strid, w / img_patch_strid) patches
        ins = self.patchify(img)
        # (bs, im_channels, num_img, h / img_patch_strid, w / img_patch_strid) patches
        ins = (
            ins.view(
                bs,
                num_cameras,
                self.im_channels,
                num_pat_img,
                num_pat_img,
            )
            .transpose(1, 2)
            .clone()
        )

        # concat proprio
        _, _, _d, _h, _w = ins.shape
        if self.add_proprio:
            p = self.proprio_preprocess(proprio)  # [B,4] -> [B,64]
            p = p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, _d, _h, _w)
            ins = torch.cat([ins, p], dim=1)  # [B, 128, num_img, np, np]

        # channel last
        ins = rearrange(ins, "b d ... -> b ... d")  # [B, num_img, np, np, 128]

        # save original shape of input for layer
        ins_orig_shape = ins.shape

        # flatten patches into sequence
        ins = rearrange(ins, "b ... d -> b (...) d")  # [B, num_img * np * np, 128]
        # add learable pos encoding
        # only added to image tokens
        if self.pe_fix:
            ins += self.pos_encoding

        # append language features as sequence
        num_lang_tok = 0
        if self.add_lang:
            l = self.lang_preprocess(
                lang_emb.view(bs * self.lang_max_seq_len, self.lang_emb_dim)
            )
            l = l.view(bs, self.lang_max_seq_len, -1)
            num_lang_tok = l.shape[1]
            ins = torch.cat((l, ins), dim=1)  # [B, num_img * np * np + 77, 128]

        # add learable pos encoding
        if not self.pe_fix:
            ins = ins + self.pos_encoding

        x = self.fc_bef_attn(ins)
        lx, imgx = x[:, :num_lang_tok], x[:, num_lang_tok:]

        # within image self attention
        imgx = imgx.reshape(bs * num_cameras, num_pat_img * num_pat_img, -1)
        for self_attn, self_ff in self.layers[: len(self.layers) // 2]:
            imgx = self_attn(imgx) + imgx
            imgx = self_ff(imgx) + imgx

        imgx = imgx.view(bs, num_cameras * num_pat_img * num_pat_img, -1)
        x = torch.cat((lx, imgx), dim=1)
        # cross attention
        for self_attn, self_ff in self.layers[len(self.layers) // 2 :]:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # append language features as sequence
        if self.add_lang:
            # throwing away the language embeddings
            x = x[:, num_lang_tok:]
        x = self.fc_aft_attn(x)

        # reshape back to orginal size
        x = x.view(bs, *ins_orig_shape[1:-1], x.shape[-1])  # [B, num_cameras, np, np, 128]
        x = rearrange(x, "b ... d -> b d ...")  # [B, 128, num_cameras, np, np]
        x = x.transpose(1, 2) # [B, num_cameras, 128, np, np]
        return x



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

        self.spatial_logits_buffer = []

        def sample_callback(lst_of_spatial_logits):
            assert len(lst_of_spatial_logits) == 1
            self.spatial_logits_buffer.append(lst_of_spatial_logits[0])
            bs = len(lst_of_spatial_logits[0])
            dev = lst_of_spatial_logits[0].device
            return torch.zeros(bs, 1, 2, device=dev) # dummy output

        self.sample_callback = sample_callback

        # produce each xyz for stage 1
        # then use xyz feature as a condition, to produce each xyz for stage 2
        # then produce rot and grip separately

        arp_cfg = ModelConfig(
            n_embd=128,
            embd_pdrop = 0.1, 
            max_seq_len = 6 + 6 + 3 + 2,
            max_chunk_size = 2, # grip and collision
            layer_norm_every_block=False,
            tokens=[
                TokenType.make(name='prompt-features', dim=1, 
                            embedding='discrete', is_control=True, 
                            embedding_kwargs={'embed_from': "prompt-features"}), 
                TokenType.make(
                        name='stage1-screen-pts', dim=2, is_continuous=True, dict_sizes=[self.img_size, self.img_size],
                        embedding="zero", predictor="upsample_from_2d_attn", 
                        predictor_kwargs={'attn_with': 'visual-featmap', 'upscale_ratio': self.img_patch_size, 'label_name': 'smooth-heatmap', 'hidden_dim_mult': 1.25}),
                TokenType.make(
                    name='stage2-screen-pts', dim=2, is_continuous=True, dict_sizes=[self.img_size, self.img_size],
                    embedding="zero", predictor="upsample_from_2d_attn", 
                    predictor_kwargs={'attn_with': 'visual-featmap', 'upscale_ratio': self.img_patch_size, 'label_name': 'smooth-heatmap',  'hidden_dim_mult': 1.25}), 
            ] +  [
                TokenType.make(name=f'rot-{c}', dim=1, is_continuous=False, dict_sizes=[self._num_rotation_classes], embedding='position_1d', predictor='class', predictor_kwargs={'label_name': f'rot-{c}'}) for c in ['x', 'y', 'z']
            ] + [
                TokenType.make(name='grip', dim=1, is_continuous=False, dict_sizes=[2], embedding='discrete', predictor='class'),
                TokenType.make(name='collision', dim=1, is_continuous=False, dict_sizes=[2], embedding='discrete', predictor='class')
            ],
            layers=[
                LayerType.make(n_head=8, AdaLN=True, norm_before_AdaLN=True, condition_on='visual-tokens', name='cross')
            ] * 4 + [
                LayerType.make(n_head=8, name='self')
            ] * 4
        )
        self.policy = AutoRegressivePolicy(arp_cfg)
        
        # gripper state only depends on xyz, but not rotation
        self.block_attn_directions = [(n, f'rot-{c}') for c in ['x', 'y', 'z'] for n in ['grip', 'collision']]
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
    
    def to_tk_reg_ids(self, token_name_regs):
        result = []
        for v in token_name_regs:
            r = [self.token_name_2_ids[v[0]], v[1]]
            if len(v) == 3: r.append(v[2])
            result.append(r)
        return result

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
                        x += noise

        img = self.render(pc, img_feat, self.mvt1)
        #endregion ###########################

        visual_featmap_1 = self.mvt1(img=img, proprio=proprio, lang_emb=lang_goal_embs) # [B, num_cameras, 128, np, np]

        if self.training:
            smooth_spatial_label_stage1, screen_waypoint_stage1 = self.get_gt_translation_action(waypoint_stage1, dims=(bs, nc, h, w))
            stage1_chk_ids = torch.as_tensor([0], device=dev)[None, :]

            # the 0, 0 are dummy input
            seq = torch.as_tensor([0, 0, self.policy.token_name_2_ids['stage1-screen-pts']], device=dev).reshape(1, 1, 3).repeat(bs, 1, 1)
            tmp_loss_dict = defaultdict(list)
            for view_id in range(3):
                _loss_dict = self.policy.compute_loss(seq, stage1_chk_ids, match_layer='cross',
                            contexts={
                                'visual-tokens': visual_featmap_1[:, view_id].flatten(-2, -1).permute(0, 2, 1),
                                'visual-featmap': visual_featmap_1[:, view_id],
                                'smooth-heatmap': smooth_spatial_label_stage1[:, :, view_id]
                            })
                for k, v in _loss_dict.items():
                    tmp_loss_dict[k].append(v)
            
            loss_dicts.append({k: sum(v) / len(v) for k, v in tmp_loss_dict.items()})
        else:
            prompt_seq = torch.zeros([bs, 0, 3], device=dev, dtype=torch.float32)
            future_tk_chk_ids = [dict(chk_id=0, tk_id=self.policy.token_name_2_ids['stage1-screen-pts'])]

            assert len(self.spatial_logits_buffer) == 0
            for view_id in range(3):
                self.policy.generate(prompt_seq, future_tk_chk_ids, match_layer='cross', sample_function=self.sample_callback,
                                    contexts={
                                            'visual-tokens': visual_featmap_1[:, view_id].flatten(-2, -1).permute(0, 2, 1),
                                            'visual-featmap': visual_featmap_1[:, view_id],
                                    })
                assert len(self.spatial_logits_buffer) == (view_id + 1)
            hms = torch.cat([F.softmax(hm_logits.reshape(bs, -1), dim=1).reshape(bs, 1, 224, 224) 
                             for hm_logits in self.spatial_logits_buffer], dim=1)
            pred_pt = [self.renderer.get_most_likely_point_3d(hms[i : i + 1]) for i in range(bs)]
            waypoint_stage1 = torch.cat(pred_pt, 0) # bs, 3
            self.spatial_logits_buffer.clear()

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

        if self.training:
            (
                action_rot_x,
                action_rot_y,
                action_rot_z,
                action_grip,       # (bs)
                action_collision,  # (bs)
            ) = [v.argmax(-1) for v in self.get_gt_rot_grip_collision(bs, action_rot, action_grip, action_ignore_collisions, device=dev)]

            if self.rotation_aug:
                rotation_aug = torch.from_numpy(np.random.choice(self.rotation_aug[0], p=self.rotation_aug[1], size=(bs, 3))).to(dev)
                action_rot_aug_x = action_rot_x + rotation_aug[:, 0]
                action_rot_aug_y = action_rot_y + rotation_aug[:, 1]
                action_rot_aug_z = action_rot_z + rotation_aug[:, 2]
            else:
                action_rot_aug_x = action_rot_x
                action_rot_aug_y = action_rot_y
                action_rot_aug_z = action_rot_z
            action_rot_aug_x %= self._num_rotation_classes
            action_rot_aug_y %= self._num_rotation_classes
            action_rot_aug_z %= self._num_rotation_classes
            smooth_spatial_label_stage2, screen_waypoint_stage2 = self.get_gt_translation_action(waypoint_stage2, dims=(bs, nc, h, w))

            stage2_chk_ids = torch.as_tensor([0], device=dev)[None, :]
            seq = torch.as_tensor([0, 0, self.policy.token_name_2_ids['stage2-screen-pts']], device=dev).reshape(1, 1, 3).repeat(bs, 1, 1)
            tmp_loss_dict = defaultdict(list)
            for view_id in range(3):
                _loss_dict = self.policy.compute_loss(seq, stage2_chk_ids, match_layer='cross',
                            contexts={
                                'visual-tokens': visual_featmap_2[:, view_id].flatten(-2, -1).permute(0, 2, 1),
                                'visual-featmap': visual_featmap_2[:, view_id],
                                'smooth-heatmap': smooth_spatial_label_stage2[:, :, view_id]
                            })
                for k, v in _loss_dict.items():
                    tmp_loss_dict[k].append(v)
            loss_dicts.append({k: sum(v) / len(v) for k, v in tmp_loss_dict.items()})

            # ------------------------------------------- #

            prompt_features = torch.cat([ # [bs, 6, 128]
                    grid_sample_from_heatmap(screen_waypoint_stage2.reshape(-1, 1, 2) / self.img_patch_size, 
                                            visual_featmap_2.flatten(0, 1))[0].reshape(bs, -1, 128),
                    visual_featmap_2.max(dim=-1)[0].max(dim=-1)[0]], dim=1)
            
            seq = torch.as_tensor([(i, self.policy.token_name_2_ids['prompt-features']) for i in range(6)], 
                                  device=dev).reshape(1, 6, 2).repeat(bs, 1, 1)
            seq = torch.cat([seq, torch.cat([
                                    torch.cat([
                                        action_rot_aug_x[:, None, None],
                                        action_rot_aug_y[:, None, None],
                                        action_rot_aug_z[:, None, None],
                                        action_grip[:, None, None],
                                        action_collision[:, None, None]], dim=1), 
                                    torch.as_tensor([self.policy.token_name_2_ids[k] for k in ['rot-x', 'rot-y', 'rot-z', 'grip', 'collision']], 
                                                    device=dev)[None, :, None].repeat(bs, 1, 1)], dim=-1)
                             ], dim=1)
            chk_ids = torch.as_tensor(list(range(11)), device=dev)[None, :]
            loss_dict_gripper = self.policy.compute_loss(seq, chk_ids,
                                                         block_attn_directions=self.block_attn_directions, 
                                                         match_layer='self', contexts={
                                                            'prompt-features': prompt_features,
                                                            'rot-x': action_rot_x[:, None],
                                                            'rot-y': action_rot_y[:, None], 'rot-z': action_rot_z[:, None]
                                                         })
            loss_dicts.append(loss_dict_gripper)
        else:
            prompt_seq = torch.zeros([bs, 0, 3], device=dev, dtype=torch.float32)
            future_tk_chk_ids = [dict(chk_id=0, tk_id=self.policy.token_name_2_ids['stage2-screen-pts'])]
            for view_id in range(3):
                self.policy.generate(prompt_seq, future_tk_chk_ids, match_layer='cross', sample_function=self.sample_callback,
                                    contexts={
                                            'visual-tokens': visual_featmap_2[:, view_id].flatten(-2, -1).permute(0, 2, 1),
                                            'visual-featmap': visual_featmap_2[:, view_id],
                                    })
                assert len(self.spatial_logits_buffer) == (view_id + 1)

            hms = torch.cat([F.softmax(hm_logits.reshape(bs, -1), dim=1).reshape(bs, 1, 224, 224) 
                             for hm_logits in self.spatial_logits_buffer], dim=1)
            pred_pt = [self.renderer.get_most_likely_point_3d(hms[i : i + 1]) for i in range(bs)]
            waypoint_stage2 = torch.cat(pred_pt, 0) # bs, 3
            self.spatial_logits_buffer.clear()

            screen_waypoint_stage2 = self.renderer.points3d_to_screen2d(waypoint_stage2[:, None, :])[:, 0]

            prompt_features = torch.cat([ # [bs, 6, 128]
                    grid_sample_from_heatmap(screen_waypoint_stage2.reshape(-1, 1, 2) / self.img_patch_size, 
                                            visual_featmap_2.flatten(0, 1))[0].reshape(bs, -1, 128),
                    visual_featmap_2.max(dim=-1)[0].max(dim=-1)[0]], dim=1)
            
            prompt_seq = torch.as_tensor([(i, self.policy.token_name_2_ids['prompt-features']) for i in range(6)], 
                                  device=dev).reshape(1, 6, 2).repeat(bs, 1, 1)
            future_tk_chk_ids = [dict(chk_id=chk_id, tk_id=self.policy.token_name_2_ids[tk_name]) 
                                 for chk_id, tk_name in zip(range(6, 11), ['rot-x', 'rot-y', 'rot-z', 'grip', 'collision'])]
            
            result_seq_stage2 = self.policy.generate(prompt_seq, future_tk_chk_ids, match_layer='self', 
                                                    sample=False,  block_attn_directions=self.block_attn_directions,  
                                                    contexts={
                                                        'prompt-features': prompt_features
                                                    })

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
            pred_rot = result_seq_stage2[:, 6:9, 0]
            pred_rot_quat = math3d.discrete_euler_to_quaternion(pred_rot.cpu().numpy(), self._rotation_resolution)
            continuous_action = np.concatenate(
                (
                    final_waypoint[0].cpu().numpy(),
                    pred_rot_quat[0],
                    result_seq_stage2[:, 9, 0].cpu().numpy(),
                    result_seq_stage2[:, 10, 0].cpu().numpy(),
                )
            )
            return continuous_action
        




class Policy:
    def __init__(self, network: PolicyNetwork, model_cfg: DictConfig, log_dir=""):
        self._optimizer_type = model_cfg.optimizer_type
        self.warmup_steps = model_cfg.warmup_steps
        self.lr_cos_dec = model_cfg.lr_cos_dec
        self.cos_dec_max_step = model_cfg.cos_dec_max_step

        self._lr = model_cfg.lr
        self._lambda_weight_l2 = model_cfg.lambda_weight_l2
        self.amp = model_cfg.amp
        self.bnb = model_cfg.bnb
        self.add_lang = model_cfg.add_lang
        self.proprio_dim = model_cfg.proprio_dim

        self._network = network
        self.log_dir = log_dir
        self.scaler = GradScaler(enabled=self.amp)

    def build(self, training: bool, device: torch.device = 'cpu'):
        self._training = training
        self._device = device

        if self._training:
            if self._optimizer_type == "lamb":
                if self.bnb:
                    import bitsandbytes as bnb
                    print("Using 8-Bit Optimizer")
                    self._optimizer = bnb.optim.LAMB(
                        self._network.parameters(),
                        lr=self._lr,
                        weight_decay=self._lambda_weight_l2,
                        betas=(0.9, 0.999),
                    )
                else:
                    # From: https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
                    self._optimizer = Lamb(
                        self._network.parameters(),
                        lr=self._lr,
                        weight_decay=self._lambda_weight_l2,
                        betas=(0.9, 0.999),
                        adam=False,
                    )
            elif self._optimizer_type == "adam":
                self._optimizer = torch.optim.Adam(
                    self._network.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                )
            else:
                raise Exception("Unknown optimizer")

            if self.lr_cos_dec:
                after_scheduler = CosineAnnealingLR(
                    self._optimizer,
                    T_max=self.cos_dec_max_step,
                    eta_min=self._lr / 100,  # mininum lr
                )
            else:
                after_scheduler = None
            self._lr_sched = GradualWarmupScheduler(
                self._optimizer,
                multiplier=1,
                total_epoch=self.warmup_steps,
                after_scheduler=after_scheduler,
            )

    def load_clip(self):
        self.clip_model, self.clip_preprocess = clip.load("RN50", device=self._device)
        self.clip_model.eval()

    def unload_clip(self):
        del self.clip_model
        del self.clip_preprocess
        with torch.cuda.device(self._device):
            torch.cuda.empty_cache()

    def reset(self, **kwargs):
        self._network.renderer.reset()

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()

    def load(self, model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        epoch = checkpoint.get("epoch", checkpoint.get("step", None))
        model = self._network
        if isinstance(model, DistributedDataParallel):
            model.module.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint["model_state"])
        try:
            self._optimizer.load_state_dict(checkpoint["optimizer_state"])
        except:
            print("WARNING: Optimizer state not loaded. KNOW WHAT YOU ARE DOING!!")
        try:
            self._lr_sched.load_state_dict(checkpoint["lr_sched_state"])
        except:
            print("WARNING: No lr_sched_state in checkpoint" "KNOW WHAT YOU ARE DOING!!")
        return epoch

    def save(self, step):
        model_path = f"{self.log_dir}/model_{step}.pth"
        model = self._network
        optimizer = self._optimizer
        lr_sched = self._lr_sched
        if isinstance(model, DistributedDataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
        torch.save(
            {
                "step": step,
                "model_state": model_state,
                "optimizer_state": optimizer.state_dict(),
                "lr_sched_state": lr_sched.state_dict(),
            },
            model_path
        )

    def update(
        self,
        replay_sample: dict,
    ) -> dict:
        assert replay_sample["gripper_pose"].shape[1:] == (7, )
        assert replay_sample["lang_goal_embs"].shape[1:] == (77, 512)
        assert replay_sample["low_dim_state"].shape[1:] == (self.proprio_dim,)
        assert self._network.training
        with autocast(enabled=self.amp):
            loss_dict = self._network(replay_sample)
            stat_dict = loss_dict.pop('stat_dict',  {})
            total_loss = sum(loss_dict.values())
            self._optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self._optimizer)
            self.scaler.update()
            self._lr_sched.step()
            loss_log = {
                **{k: v.item() for k, v in loss_dict.items()},
                "lr": self._optimizer.param_groups[0]["lr"],
                **stat_dict
            }
        return loss_log

    @torch.no_grad()
    def act(
        self, step: int, observation: dict
    ) -> ActResult:
        assert observation['left_shoulder_rgb'].size(0) == 1, "Only batch size 1 is supported for evaluation"
        if self.add_lang:
            lang_goal_tokens = observation.get("lang_goal_tokens", None).long()
            _, lang_goal_embs = clip_encode_text(self.clip_model, lang_goal_tokens)
            lang_goal_embs = lang_goal_embs.float()
            observation['lang_goal_embs'] = lang_goal_embs
        else:
            lang_goal_embs = (
                torch.zeros(observation["lang_goal_embs"].shape)
                .float()
                .to(self._device)
            )
            observation['lang_goal_embs'] = lang_goal_embs
        assert not self._network.training
        continuous_action = self._network(observation)
        return ActResult(continuous_action)

