from torch import nn
from einops import rearrange
import random
from transformers import Dinov2Model
from peft import LoraConfig, get_peft_model
import numpy as np
from collections import defaultdict, Counter
import arp
from torchvision.transforms.functional import to_pil_image
import torch
import os.path as osp
import bitsandbytes as bnb
import torch.nn.functional as F
from dataset import SCENE_BOUNDS
from utils.preprocess import CubePointCloudRenderer, preprocess_images_in_batch, \
        flatten_img_pc_to_points, clamp_pc_in_bound, place_pc_in_cube, generate_heatmap_from_screen_pts, apply_se3_augmentation
from utils.optim import GradualWarmupScheduler
from PIL import Image, ImageDraw
from torch.cuda.amp import autocast, GradScaler
from vit import ViT

from torch.optim.lr_scheduler import CosineAnnealingLR


def draw_dots(renderer, pil_images, pts, color):
    dots = renderer.points3d_to_screen2d(pts.reshape(1, -1, 3))
    dot_radius = 3
    dots = dots.reshape(-1, 3, 2)
    for a in range(len(dots)):
        for b in range(3):
            im = pil_images[b]
            draw = ImageDraw.Draw(im)
            x, y = dots[a, b].long().tolist() 
            draw.ellipse((x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius), fill=color)
    return dots


class RobotPolicy(nn.Module):
    def __init__(self, model_cfg, device):
        super().__init__()
        image_size = self.image_size = model_cfg.image_size
        patch_size = model_cfg.patch_size
        num_tokens = image_size // patch_size
        hidden_dim = model_cfg.hidden_dim
        cameras = ['top', 'left', ] # 'front'
        self.num_cameras = len(cameras)
        self.num_inp_channels = 10
        self.hidden_dim = hidden_dim
        self.num_tokens_hw = num_tokens

        self.trans_aug_range = model_cfg.trans_aug_range
        self.rot_aug_range = model_cfg.rot_aug_range

        self.patchify = nn.Sequential(
            nn.Conv2d(self.num_inp_channels, hidden_dim, patch_size, patch_size,  padding=0),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        self.pixel_loc = torch.zeros((len(cameras), 3, image_size, image_size))
        self.pixel_loc[:, 0, :, :] = torch.linspace(-1, 1, len(cameras)).unsqueeze(-1).unsqueeze(-1)
        self.pixel_loc[:, 1, :, :] = torch.linspace(-1, 1, image_size).unsqueeze(0).unsqueeze(-1)
        self.pixel_loc[:, 2, :, :] = torch.linspace(-1, 1, image_size).unsqueeze(0).unsqueeze(0) # [3, 3, 224, 224]
        self.vit = ViT(dim=hidden_dim, depth=model_cfg.n_encoder_layers, heads=model_cfg.n_heads, 
                       mlp_dim=model_cfg.dim_feedforward, num_patches=(num_tokens ** 2) * self.num_cameras, 
                       dim_head=hidden_dim//2, dropout=model_cfg.dropout, emb_dropout=model_cfg.dropout)

        self.renderer = CubePointCloudRenderer(f"cuda:{device}", (image_size, image_size), with_depth=True, cameras=cameras) 


        self.policy = arp.AutoRegressivePolicy(
            arp.ModelConfig(
                n_embd=model_cfg.hidden_dim,
                embd_pdrop = model_cfg.dropout,
                max_seq_len = model_cfg.max_seq_len,
                max_chunk_size = model_cfg.max_chunk_size,  # 
                layer_norm_every_block=False,
                tokens=[
                    arp.TokenType.make(name='curr_wrench', dim=2, is_continuous=True, embedding="position_2d", is_control=True),
                    arp.TokenType.make(name='curr_gripper', dim=2, is_continuous=True, is_control=True, embedding='position_2d'),

                    arp.TokenType.make(name='command', dim=1, embedding='discrete', dict_sizes=[3], predictor='class'),

                    arp.TokenType.make(name='next_wrench', dim=2, is_continuous=True, dict_sizes=[image_size, image_size],
                        embedding="feat_grid_2d", embedding_kwargs={'sampling_from': 'visual-featmap', 'stride': patch_size}, predictor="upsample_from_2d_attn", 
                        predictor_kwargs={'attn_with': 'visual-featmap', 'upscale_ratio': patch_size, 'label_name': 'smooth-heatmap-wrench'}),

                    arp.TokenType.make(name='next_gripper', dim=2, is_continuous=True, dict_sizes=[image_size, image_size],
                        embedding="feat_grid_2d", embedding_kwargs={'sampling_from': 'visual-featmap', 'stride': patch_size}, predictor="upsample_from_2d_attn", 
                        predictor_kwargs={'attn_with': 'visual-featmap', 'upscale_ratio': patch_size, 'label_name': 'smooth-heatmap-gripper'}),

                    arp.TokenType.make(name='next_wrench_fine', dim=2, is_continuous=True, dict_sizes=[image_size, image_size],
                        embedding="feat_grid_2d", embedding_kwargs={'sampling_from': 'visual-featmap', 'stride': patch_size}, predictor="upsample_from_2d_attn", 
                        predictor_kwargs={'attn_with': 'visual-featmap', 'upscale_ratio': patch_size, 'label_name': 'smooth-heatmap-wrench'}),

                    arp.TokenType.make(name='next_gripper_traj', dim=2, is_continuous=True, dict_sizes=[image_size, image_size],
                        embedding="feat_grid_2d", embedding_kwargs={'sampling_from': 'visual-featmap', 'stride': patch_size}, predictor="upsample_from_2d_attn", 
                        predictor_kwargs={'attn_with': 'visual-featmap', 'upscale_ratio': patch_size, 'label_name': 'smooth-heatmap-gripper'}),
                    
                     arp.TokenType.make(name='prompt-features', dim=1, 
                        embedding='discrete', is_control=True, 
                        embedding_kwargs={'embed_from': "prompt-features"})
                ],
                layers=[
                    arp.LayerType.make(n_head=8, AdaLN=True, condition_on='visual-tokens')
                ] * model_cfg.depth
            )
        )

        self.learning = {}
    
    def build(self, train_cfg):
        optimizer = bnb.optim.LAMB(
            self.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.lambda_weight_l2,
            betas=(0.9, 0.999),
        )
        after_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=train_cfg.num_steps,
                eta_min=train_cfg.lr / 100,  # mininum lr
        )
        lr_scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=train_cfg.warmup_steps,
            after_scheduler=after_scheduler,
        )

        scaler = GradScaler(enabled=True)
        self.learning.update({
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'scaler': scaler
        })
    
    def save(self, steps, path):
        torch.save(self.state_dict(), osp.join(path, f"model_{steps:08d}.pth"))
    

    def forward_train(self, batch, backprop=True):
        """
        {  
            'points': list of tensor
            'colors': list of tensor

            'command': full tensor
            
            'target_wrench',
            'target_refined_wrench',
            'curr_wrench',
            'curr_gripper',
            'target_gripper_traj' 
        } 
        """
        dev = batch['command'].device
        batch_size = len(batch['command'])
        REACH, ADJUST, ROTATE = 0, 1, 2
        COMMANDS = [REACH, ADJUST, ROTATE]
        scene_bounds = torch.tensor(SCENE_BOUNDS).to(dev)
        rot_aug_range = torch.tensor(self.rot_aug_range).to(dev)
        trans_aug_range = torch.tensor(self.trans_aug_range).to(dev)
        bound_min, bound_max = torch.tensor(SCENE_BOUNDS[:3]).cuda()[None, ...], torch.tensor(SCENE_BOUNDS[3:]).cuda()[None, ...]

        # region Section 1: point cloud augmentation and place pc in cube
        for C in COMMANDS:
            indices = batch['command'] == C
            num_samples = indices.sum().item()
            if num_samples == 0: continue
            indices_lst = indices.nonzero().flatten().tolist()
            points = [batch['points'][i] for i in indices_lst]
            colors = [batch['colors'][i] for i in indices_lst]
            
            if C == REACH:
                center = batch['target_wrench']
                control_points = [batch['curr_wrench'][indices, None], batch['curr_gripper'][indices, None], batch['target_wrench'][:, None], batch['target_gripper'][:, None]]
            elif C == ADJUST:
                center = batch['target_refined_wrench']
                control_points = [batch['curr_wrench'][indices, None], batch['curr_gripper'][indices, None], batch['target_refined_wrench'][:, None]]
            else:
                center = batch['curr_wrench'][indices]
                control_points = [batch['curr_wrench'][indices, None], batch['curr_gripper'][indices, None], batch['target_gripper_traj']]
            
            control_points = torch.cat(control_points, 1)
            for i in range(len(points)):
                batch_i = indices_lst[i]
                pcd = points[i]
                color = colors[i]
                ctrl_pts = control_points[i]
                if self.training:
                    all_pts = torch.cat([pcd, ctrl_pts], dim=0)
                    all_pts, _ = apply_se3_augmentation(all_pts[None, ...], center[[i]], scene_bounds, trans_aug_range, rot_aug_range)
                    all_pts = all_pts[0]
                    pcd = all_pts[:pcd.shape[0]]
                    ctrl_pts = all_pts[pcd.shape[0]:]
                
                pcd, color = clamp_pc_in_bound(pcd[None, ...], color[None, ...], SCENE_BOUNDS)
                pcd, color = pcd[0], color[0]
                if C == REACH:
                    ctrl_pts[2:] = ctrl_pts[2:].clamp(bound_min, bound_max)
                else:
                    ctrl_pts = ctrl_pts.clamp(bound_min, bound_max)
                pcd_cube, _ = place_pc_in_cube(pcd, with_mean_or_bounds=False, scene_bounds=SCENE_BOUNDS)
                ctrl_pts_cube, _ = place_pc_in_cube(pcd, ctrl_pts, with_mean_or_bounds=False, scene_bounds=SCENE_BOUNDS)

                batch['points'][batch_i] = pcd_cube
                batch['colors'][batch_i] = color    
                # assert pcd_cube.min() >= -1 and pcd_cube.max() <= 1

                batch['curr_wrench'][batch_i] = ctrl_pts_cube[0]
                batch['curr_gripper'][batch_i] = ctrl_pts_cube[1]
                if C == REACH:
                    batch['target_wrench'][i] = ctrl_pts_cube[2]
                    batch['target_gripper'][i] = ctrl_pts_cube[3]
                elif C == ADJUST:
                    batch['target_refined_wrench'][i] = ctrl_pts_cube[2]
                else:
                    batch['target_gripper_traj'][i] = ctrl_pts_cube[2:]
        # endregion
        
        # region Section 2: rendering and feature extraction
        all_virtual_images = []
        for bi in range(batch_size):
            pts, pixels = batch['points'][bi], batch['colors'][bi]
            pixels = (pixels - 0.5) * 2
            max_pc = torch.max(torch.abs(pts))
            virtual_images = self.renderer(pts, torch.cat((pts / max_pc, pixels), dim=-1)).unsqueeze(0)
            all_virtual_images.append(virtual_images)

        img = torch.cat(all_virtual_images)
        img = img.permute(0, 1, 4, 2, 3) # [1, 3, 7, 224, 224]
        img = torch.cat((img, self.pixel_loc.to(dev).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)), dim=2)
        img = img.flatten(0, 1)

        def forward():
            feats = self.patchify(img)
            feats = feats.view(
                    batch_size,
                    self.num_cameras,
                    self.hidden_dim,
                    self.num_tokens_hw,
                    self.num_tokens_hw,
                ).transpose(1, 2)

            feats = rearrange(feats, "b d ... -> b ... d")
            orig_shape = feats.shape # (bs, cameras, h, w, hidden_dim)
            feats = rearrange(feats, "b ... d -> b (...) d")
            global_feat, feats = self.vit(feats) # b, ..., d
            global_feat = global_feat[:, None, :]
            visual_tokens = feats
            feats = feats.reshape(*orig_shape)
            feats = rearrange(feats, "b ... d -> b d ...")  # (bs, hidden_dim, cameras, h, w)
            feats = feats.transpose(1, 2) # (bs, cameras, hidden_dim, h, w)
            # endregion

            loss_dict = defaultdict(list)
            def add_to_losses(_loss_dict):
                for k, v in _loss_dict.items():
                    loss_dict[k].append(v)

            # region Section 3: autoregression + loss computation + prediction

            # NOTE 1) first predict command 
            chk_ids_list = list(range(2 + 2 * self.num_cameras))
            START_CHK = max(chk_ids_list) + 1
            tk_names_lst = ['curr_wrench'] * self.num_cameras + ['curr_gripper'] * self.num_cameras + ['prompt-features'] + ['command']

            all_commands = batch['command'].reshape(batch_size, -1, 1)
            tk_vals = arp.cat_uneven_blc_tensors(self.renderer.points3d_to_screen2d(batch['curr_wrench'][:, None])[:, 0], 
                                                self.renderer.points3d_to_screen2d(batch['curr_gripper'][:, None])[:, 0], 
                                                torch.zeros(batch_size, 1, 1, device=dev),
                                                all_commands)

            tk_names = tk_names_lst 
            chk_ids = torch.as_tensor(chk_ids_list).to(dev)
            tk_ids = torch.as_tensor([self.policy.token_name_2_ids[n] for n in tk_names]).to(dev)
            tk_ids = tk_ids.reshape(1, -1, 1).repeat(batch_size, 1, 1)
            tks = torch.cat((tk_vals, tk_ids), dim=-1)
            _loss_dict = self.policy.compute_loss(tks, chk_ids, contexts={
                    'prompt-features': global_feat,
                    'visual-tokens': None
            })
            add_to_losses(_loss_dict) 
            
            # NOTE 2) command specific prediction
            for C in COMMANDS:
                indices = batch['command'] == C
                num_samples = indices.sum().item()
                if num_samples == 0: continue
                if C == REACH: 
                    chk_ids = list(chk_ids_list)
                    chk_ids += ([START_CHK] * self.num_cameras + [START_CHK+1] * self.num_cameras)
                    chk_ids = torch.as_tensor(chk_ids).to(dev)
                    tk_names = tk_names_lst + ['next_wrench'] * self.num_cameras + ['next_gripper'] * self.num_cameras
                    tk_ids = torch.as_tensor([self.policy.token_name_2_ids[n] for n in tk_names]).to(dev)
                    tk_ids = tk_ids.reshape(1, -1, 1).repeat(num_samples, 1, 1)

                    wrench_pts = self.renderer.points3d_to_screen2d(batch['target_wrench'][:, None])[:, 0]
                    wrench_hm = generate_heatmap_from_screen_pts(wrench_pts.reshape(-1, 2), (self.image_size, self.image_size), sigma=1.5, thres_sigma_times=3)
                    wrench_hm = wrench_hm.reshape(num_samples, self.num_cameras, -1)

                    gripper_pts = self.renderer.points3d_to_screen2d(batch['target_gripper'][:, None])[:, 0]
                    gripper_hm = generate_heatmap_from_screen_pts(gripper_pts.reshape(-1, 2), (self.image_size, self.image_size), sigma=1.5, thres_sigma_times=3)
                    gripper_hm = gripper_hm.reshape(num_samples, self.num_cameras, -1)

                    this_tk_vals = arp.cat_uneven_blc_tensors(tk_vals[indices], wrench_pts, gripper_pts) 
                    tks = torch.cat((this_tk_vals, tk_ids), dim=-1)

                    _loss_dict = self.policy.compute_loss(tks, chk_ids, contexts={
                        'visual-featmap': feats[indices].flatten(0, 1),
                        'visual-tokens': visual_tokens[indices],

                        'smooth-heatmap-wrench': wrench_hm.flatten(0, 1),
                        'smooth-heatmap-gripper': gripper_hm.flatten(0, 1),
                        
                        'prompt-features': global_feat[indices]
                    }, skip_tokens=[self.policy.token_name_2_ids['command'], ])
                    add_to_losses(_loss_dict)

                elif C == ADJUST: # adjust_direction
                    chk_ids = list(chk_ids_list)
                    chk_ids += [START_CHK,] * self.num_cameras
                    chk_ids = torch.as_tensor(chk_ids).to(dev)
                    tk_names = tk_names_lst + ['next_wrench_fine'] * self.num_cameras
                    tk_ids = torch.as_tensor([self.policy.token_name_2_ids[n] for n in tk_names]).to(dev)
                    tk_ids = tk_ids.reshape(1, -1, 1).repeat(num_samples, 1, 1)

                    target_pts = self.renderer.points3d_to_screen2d(batch['target_refined_wrench'][:, None])[:, 0]
                    hm = generate_heatmap_from_screen_pts(target_pts.reshape(-1, 2), (self.image_size, self.image_size), sigma=1.5, thres_sigma_times=3)
                    hm = hm.reshape(num_samples, self.num_cameras, -1)

                    this_tk_vals = arp.cat_uneven_blc_tensors(tk_vals[indices], target_pts) 
                    tks = torch.cat((this_tk_vals, tk_ids), dim=-1)

                    _loss_dict = self.policy.compute_loss(tks, chk_ids, contexts={
                        'visual-featmap': feats[indices].flatten(0, 1),
                        'visual-tokens': visual_tokens[indices],
                        'smooth-heatmap-wrench': hm.flatten(0, 1),
                        'prompt-features': global_feat[indices]
                    }, skip_tokens=[self.policy.token_name_2_ids['command'], ])
                    add_to_losses(_loss_dict)
                else:
                    chk_ids = list(chk_ids_list)
                    chk_ids += ([START_CHK,] * self.num_cameras * 6)
                    chk_ids = torch.as_tensor(chk_ids).to(dev)
                    tk_names = tk_names_lst + ['next_gripper_traj'] * self.num_cameras * 6
                    tk_ids = torch.as_tensor([self.policy.token_name_2_ids[n] for n in tk_names]).to(dev)
                    tk_ids = tk_ids.reshape(1, -1, 1).repeat(num_samples, 1, 1)

                    target_pts = self.renderer.points3d_to_screen2d(batch['target_gripper_traj'])
                    hm = generate_heatmap_from_screen_pts(target_pts.reshape(-1, 2), (self.image_size, self.image_size), sigma=1.5, thres_sigma_times=3)
                    hm = hm.reshape(num_samples, 6 * self.num_cameras, -1)
                    target_pts = target_pts.flatten(1, 2)

                    this_tk_vals = arp.cat_uneven_blc_tensors(tk_vals[indices], target_pts) 
                    tks = torch.cat((this_tk_vals, tk_ids), dim=-1)
                    
                    _loss_dict = self.policy.compute_loss(tks, chk_ids, contexts={
                        'visual-featmap': feats[indices].repeat(1, 6, 1, 1, 1).flatten(0, 1),
                        'visual-tokens': visual_tokens[indices],
                        'smooth-heatmap-gripper': hm.flatten(0, 1),
                        'prompt-features': global_feat[indices]
                    }, skip_tokens=[self.policy.token_name_2_ids['command'], ])
                    add_to_losses(_loss_dict)

            # endregion

            loss_dict = {k: sum(v) / len(v) for k, v in loss_dict.items()}
            loss_dict['total_loss'] = sum(loss_dict.values())
            return loss_dict

        with autocast(enabled=True):
            loss_dict = forward()
        
        if backprop:
            self.learning['optimizer'].zero_grad(set_to_none=True)
            self.learning['scaler'].scale(loss_dict['total_loss']).backward()
            self.learning['scaler'].step(self.learning['optimizer'])
            self.learning['scaler'].update()

            self.learning['lr_scheduler'].step()

        loss_dict = {k: v.item() for k, v in loss_dict.items()}

        return loss_dict
    
    
    
    def predict_command(self, batch):
        """
        {  
            'points': tensor Nx3
            'colors': tensor Nx3,
            'curr_wrench': tensor 3,
            'curr_gripper': tensor 3
        } 
        """
        dev = batch['points'][0].device
        batch_size = 1
        REACH, ADJUST, ROTATE = 0, 1, 2
        COMMANDS = ['reach', 'adjust', 'rotate']
        scene_bounds = torch.tensor(SCENE_BOUNDS).to(dev)

        pcd, color = batch['points'], batch['colors']
        pcd, color = clamp_pc_in_bound(pcd[None, ...], color[None, ...], SCENE_BOUNDS)
        pcd, color = pcd[0], color[0]

        ctrl_pts = [batch['curr_wrench'][None], batch['curr_gripper'][None]]
        ctrl_pts = torch.cat(ctrl_pts)
        ctrl_pts_cube, _ = place_pc_in_cube(pcd, ctrl_pts, with_mean_or_bounds=False, scene_bounds=SCENE_BOUNDS)
        curr_wrench = ctrl_pts_cube[:1]
        curr_gripper = ctrl_pts_cube[1:]
        pcd_cube, _ = place_pc_in_cube(pcd, with_mean_or_bounds=False, scene_bounds=SCENE_BOUNDS)

        color = (color - 0.5) * 2
        max_pc = torch.max(torch.abs(pcd_cube))
        img = self.renderer(pcd_cube, torch.cat((pcd_cube / max_pc, color), dim=-1)).unsqueeze(0)
        img = img.permute(0, 1, 4, 2, 3) # [1, 3, 7, 224, 224]
        img = torch.cat((img, self.pixel_loc.to(dev).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)), dim=2)
        img = img.flatten(0, 1)

        feats = self.patchify(img)
        feats = feats.view(
                batch_size,
                self.num_cameras,
                self.hidden_dim,
                self.num_tokens_hw,
                self.num_tokens_hw,
            ).transpose(1, 2)

        feats = rearrange(feats, "b d ... -> b ... d")
        orig_shape = feats.shape # (bs, cameras, h, w, hidden_dim)
        feats = rearrange(feats, "b ... d -> b (...) d")
        global_feat, feats = self.vit(feats) # b, ..., d
        global_feat = global_feat[:, None, :]
        feats = feats.reshape(*orig_shape)
        feats = rearrange(feats, "b ... d -> b d ...")  # (bs, hidden_dim, cameras, h, w)
        feats = feats.transpose(1, 2) # (bs, cameras, hidden_dim, h, w)

        tk_names = ['curr_wrench'] * self.num_cameras + ['curr_gripper'] * self.num_cameras + ['prompt-features']
        tk_vals = arp.cat_uneven_blc_tensors(self.renderer.points3d_to_screen2d(curr_wrench[:, None])[:, 0], 
                                                self.renderer.points3d_to_screen2d(curr_gripper[:, None])[:, 0], 
                                                torch.zeros(batch_size, 1, 1, device=dev))
        tk_ids = torch.as_tensor([self.policy.token_name_2_ids[n] for n in tk_names]).to(dev)
        tk_ids = tk_ids.reshape(1, -1, 1).repeat(batch_size, 1, 1)
        prompt_tks = torch.cat((tk_vals, tk_ids), dim=-1) 
        
        result = self.policy.generate(prompt_tks, 
            future_tk_chk_ids=[{'chk_id': prompt_tks.size(1), 'tk_id': self.policy.token_name_2_ids['command']}], 
            contexts={
                'prompt-features': global_feat,
                'visual-tokens': None
            })

        cmd = result[0, -1, 0].item()
        return COMMANDS[int(cmd)]

    
    
    def predict_detailed_actions(self, batch):
        """
        {  
            'points': tensor Nx3
            'colors': tensor Nx3,
            'curr_wrench': tensor 3,
            'curr_gripper': tensor 3,
            'command': str
        } 
        """
        dev = batch['points'][0].device
        batch_size = 1
        cmd = batch['command']
        REACH, ADJUST, ROTATE = 0, 1, 2
        COMMANDS = ['reach', 'adjust', 'rotate']
        scene_bounds = torch.tensor(SCENE_BOUNDS).to(dev)

        pcd, color = batch['points'], batch['colors']
        pcd, color = clamp_pc_in_bound(pcd[None, ...], color[None, ...], SCENE_BOUNDS)
        pcd, color = pcd[0], color[0]

        ctrl_pts = [batch['curr_wrench'][None], batch['curr_gripper'][None]]
        ctrl_pts = torch.cat(ctrl_pts)
        ctrl_pts_cube, rev_trans = place_pc_in_cube(pcd, ctrl_pts, with_mean_or_bounds=False, scene_bounds=SCENE_BOUNDS)
        curr_wrench = ctrl_pts_cube[:1]
        curr_gripper = ctrl_pts_cube[1:]
        pcd_cube, _ = place_pc_in_cube(pcd, with_mean_or_bounds=False, scene_bounds=SCENE_BOUNDS)

        color = (color - 0.5) * 2
        max_pc = torch.max(torch.abs(pcd_cube))
        img = self.renderer(pcd_cube, torch.cat((pcd_cube / max_pc, color), dim=-1)).unsqueeze(0)

        # for visualization
        pil_images = [img[0, 0, :, :, 3:6].permute(2, 0, 1), img[0, 1, :, :, 3:6].permute(2, 0, 1)]
        pil_images = [im / 2 + 0.5 for im in pil_images]
        pil_images = [to_pil_image(im) for im in pil_images]        

        img = img.permute(0, 1, 4, 2, 3) # [1, 3, 7, 224, 224]
       
        img = torch.cat((img, self.pixel_loc.to(dev).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)), dim=2)
        img = img.flatten(0, 1)

        feats = self.patchify(img)
        feats = feats.view(
                batch_size,
                self.num_cameras,
                self.hidden_dim,
                self.num_tokens_hw,
                self.num_tokens_hw,
            ).transpose(1, 2)

        feats = rearrange(feats, "b d ... -> b ... d")
        orig_shape = feats.shape # (bs, cameras, h, w, hidden_dim)
        feats = rearrange(feats, "b ... d -> b (...) d")
        global_feat, feats = self.vit(feats) # b, ..., d
        global_feat = global_feat[:, None, :]
        visual_tokens = feats
        feats = feats.reshape(*orig_shape)
        feats = rearrange(feats, "b ... d -> b d ...")  # (bs, hidden_dim, cameras, h, w)
        feats = feats.transpose(1, 2)

        tk_names = ['curr_wrench'] * self.num_cameras + ['curr_gripper'] * self.num_cameras + ['prompt-features'] + ['command']

        all_commands = torch.as_tensor([COMMANDS.index(cmd)]).reshape(batch_size, -1, 1).to(dev)
        tk_vals = arp.cat_uneven_blc_tensors(self.renderer.points3d_to_screen2d(curr_wrench[:, None])[:, 0], 
                                                self.renderer.points3d_to_screen2d(curr_gripper[:, None])[:, 0], 
                                                torch.zeros(batch_size, 1, 1, device=dev),
                                                all_commands)
        tk_ids = torch.as_tensor([self.policy.token_name_2_ids[n] for n in tk_names]).to(dev)
        tk_ids = tk_ids.reshape(1, -1, 1).repeat(batch_size, 1, 1)
        prompt_tks = torch.cat((tk_vals, tk_ids), dim=-1) 
        START_CHK = prompt_tks.size(1)

        spatial_logits_buffer = []
        points_buffer = []
        def sample_callback(lst_of_spatial_logits):
            lst_of_spatial_logits = [a[:, None] if len(a.shape) == 3 else a for a in lst_of_spatial_logits]
            spatial_logits_buffer.extend(lst_of_spatial_logits)
            hm_logits = torch.cat([a for a in lst_of_spatial_logits], dim=1)
            hm = F.softmax(hm_logits.flatten(2), dim=2)
            hm = hm.view(-1, self.num_cameras, self.image_size, self.image_size)
            pred_pt = [self.renderer.get_most_likely_point_3d(hm[i : i + 1]) for i in range(len(hm))]
            spatial_point = torch.cat(pred_pt, 0) # bs, 3
            screen_points = self.renderer.points3d_to_screen2d(spatial_point[:, None, :])
            screen_points = screen_points[:, 0]
            spatial_point_back = rev_trans(spatial_point)
            points_buffer.append({
                '3d_normalized': spatial_point,
                '3d': spatial_point_back,
                '2d': screen_points
            })
            return screen_points.reshape(batch_size, -1, 2)

        if cmd == 'reach':
            future_tk_chk_ids = [{'chk_id': START_CHK, 'tk_id': self.policy.token_name_2_ids['next_wrench']}] * self.num_cameras + \
                                [{'chk_id': START_CHK+1, 'tk_id': self.policy.token_name_2_ids['next_gripper']}] * self.num_cameras
                
            self.policy.generate(prompt_tks, 
                    future_tk_chk_ids=future_tk_chk_ids, 
                    contexts={
                        'prompt-features': global_feat,
                        'visual-tokens': visual_tokens,
                        'visual-featmap': feats.flatten(0, 1)
                }, sample=True, sample_function={frozenset({START_CHK, START_CHK+1}): sample_callback})
            
            return {
                'pil_images': pil_images,
                'wrench_points': points_buffer[0],
                'gripper_points': points_buffer[1]
            } 

        elif cmd == 'adjust':
            future_tk_chk_ids = [{'chk_id': START_CHK, 'tk_id': self.policy.token_name_2_ids['next_wrench_fine']}] * self.num_cameras
            self.policy.generate(prompt_tks, 
                    future_tk_chk_ids=future_tk_chk_ids, 
                    contexts={
                        'prompt-features': global_feat,
                        'visual-tokens': visual_tokens,
                        'visual-featmap': feats.flatten(0, 1)
                }, sample=True, sample_function={START_CHK: sample_callback})
            return {
                'pil_images': pil_images,
                'wrench_points': points_buffer[-1]
            }
        else:
            future_tk_chk_ids = [{'chk_id': START_CHK, 'tk_id': self.policy.token_name_2_ids['next_gripper_traj']}] * 6 * self.num_cameras
            self.policy.generate(prompt_tks, 
                    future_tk_chk_ids=future_tk_chk_ids, 
                    contexts={
                        'prompt-features': global_feat,
                        'visual-tokens': visual_tokens,
                        'visual-featmap': feats.repeat(1, 6, 1, 1, 1).flatten(0, 1)
                }, sample=True, sample_function={START_CHK: sample_callback})
            return {
                'pil_images': pil_images,
                'gripper_points': points_buffer[-1]
            }