from math import ceil
from omegaconf import DictConfig
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat
from typing import Optional
import torch
import torchvision
import numpy as np
import clip
from scipy.spatial.transform import Rotation
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.optim import Lamb, GradualWarmupScheduler
from utils.structure import ActResult
import utils.math3d as math3d
from utils.clip import clip_encode_text
from preprocess import CubePointCloudRenderer, preprocess_images_in_batch, \
    flatten_img_pc_to_points, clamp_pc_in_bound, place_pc_in_cube, generate_heatmap_from_screen_pts, \
    apply_se3_augmentation

from utils.layers import (
    Conv2DBlock,
    Conv2DUpsampleBlock,
    PreNorm,
    Attention,
    DenseBlock,
    FeedForward
)

class MultiViewTransformer(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        renderer: Optional[CubePointCloudRenderer]=None,
    ):
        super().__init__()
        self.depth = cfg.depth
        self.img_feat_dim = cfg.img_feat_dim
        self.img_size = cfg.img_size
        self.add_proprio = cfg.add_proprio
        self.proprio_dim = cfg.proprio_dim
        self.add_lang = cfg.add_lang
        self.lang_dim = cfg.lang_dim
        self.lang_len = cfg.lang_len
        self.im_channels = cfg.im_channels
        self.img_patch_size = cfg.img_patch_size
        self.final_dim = cfg.final_dim
        self.attn_dropout = cfg.attn_dropout
        self.decoder_dropout = cfg.decoder_dropout
        self.self_cross_ver = cfg.self_cross_ver
        self.add_corr = cfg.add_corr
        self.add_pixel_loc = cfg.add_pixel_loc
        self.add_depth = cfg.add_depth
        self.pe_fix = cfg.pe_fix
        # print(f"MultiViewTransformer: {vars(self)}")

        assert renderer is not None
        self.renderer = renderer
        self.num_cameras = self.renderer.num_cameras

        # patchified input dimensions
        spatial_size = self.img_size // self.img_patch_size  # 220 / 11 = 20

        if self.add_proprio:
            # 64 img features + 64 proprio features
            self.input_dim_before_seq = self.im_channels * 2
        else:
            self.input_dim_before_seq = self.im_channels

        # learnable positional encoding
        if self.add_lang:
            self.lang_emb_dim, self.lang_max_seq_len = self.lang_dim, self.lang_len
        else:
            self.lang_emb_dim, self.lang_max_seq_len = 0, 0

        if self.pe_fix:
            num_pe_token = spatial_size**2 * self.num_cameras
        else:
            num_pe_token = self.lang_max_seq_len + (spatial_size**2 * self.num_cameras)

        self.pos_encoding = nn.Parameter(
            torch.randn(
                1,
                num_pe_token,
                self.input_dim_before_seq,
            )
        )

        inp_img_feat_dim = self.img_feat_dim

        if self.add_corr:
            inp_img_feat_dim += 3

        if self.add_pixel_loc:
            inp_img_feat_dim += 3
            self.pixel_loc = torch.zeros(
                (self.num_cameras, 3, self.img_size, self.img_size)
            )
            self.pixel_loc[:, 0, :, :] = (
                torch.linspace(-1, 1, self.num_cameras).unsqueeze(-1).unsqueeze(-1)
            )
            self.pixel_loc[:, 1, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(-1)
            )
            self.pixel_loc[:, 2, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(0)
            )

        if self.add_depth:
            inp_img_feat_dim += 1

        # img input preprocessing encoder
        self.input_preprocess = Conv2DBlock(
            inp_img_feat_dim,
            self.im_channels,
            kernel_sizes=1,
            strides=1,
            norm=None,
            activation=cfg.activation,
        )
        inp_pre_out_dim = self.im_channels

        if self.add_proprio:
            # proprio preprocessing encoder
            self.proprio_preprocess = DenseBlock(
                self.proprio_dim,
                self.im_channels,
                norm="group",
                activation=cfg.activation,
            )

        self.patchify = Conv2DBlock(
            inp_pre_out_dim,
            self.im_channels,
            kernel_sizes=self.img_patch_size,
            strides=self.img_patch_size,
            norm="group",
            activation=cfg.activation,
            padding=0,
        )

        # lang preprocess
        if self.add_lang:
            self.lang_preprocess = DenseBlock(
                self.lang_emb_dim,
                self.im_channels * 2,
                norm="group",
                activation=cfg.activation,
            )

        self.fc_bef_attn = DenseBlock(
            self.input_dim_before_seq,
            cfg.attn_dim,
            norm=None,
            activation=None,
        )
        self.fc_aft_attn = DenseBlock(
            cfg.attn_dim,
            self.input_dim_before_seq,
            norm=None,
            activation=None,
        )

        get_attn_attn = lambda: PreNorm(
            cfg.attn_dim,
            Attention(
                cfg.attn_dim,
                heads=cfg.attn_heads,
                dim_head=cfg.attn_dim_head,
                dropout=cfg.attn_dropout,
            ),
        )

        get_attn_ff = lambda: PreNorm(cfg.attn_dim, FeedForward(cfg.attn_dim))
        # self-attention layers
        self.layers = nn.ModuleList([])
        attn_depth = self.depth

        for _ in range(attn_depth):
            self.layers.append(nn.ModuleList([get_attn_attn(), get_attn_ff()]))

        self.up0 = Conv2DUpsampleBlock(
            self.input_dim_before_seq,
            self.im_channels,
            kernel_sizes=self.img_patch_size,
            strides=self.img_patch_size,
            norm=None,
            activation=cfg.activation,
        )

        final_inp_dim = self.im_channels + inp_pre_out_dim

        # final layers
        self.final = Conv2DBlock(
            final_inp_dim,
            self.im_channels,
            kernel_sizes=3,
            strides=1,
            norm=None,
            activation=cfg.activation,
        )

        self.trans_decoder = Conv2DBlock(
            self.final_dim,
            1,
            kernel_sizes=3,
            strides=1,
            norm=None,
            activation=None,
        )

        out_size = cfg.feat_dim
        feat_fc_dim = 0
        feat_fc_dim += self.input_dim_before_seq
        feat_fc_dim += self.final_dim

        self.feat_fc = nn.Sequential(
            nn.Linear(self.num_cameras * feat_fc_dim, feat_fc_dim),
            nn.ReLU(),
            nn.Linear(feat_fc_dim, feat_fc_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_fc_dim // 2, out_size),
        )

    def forward(
        self,
        img,
        proprio=None,
        lang_emb=None,
        **kwargs,
    ):
        """
        :param img: tensor of shape (bs, num_img, img_feat_dim, h, w)
        :param proprio: tensor of shape (bs, priprio_dim)
        :param lang_emb: tensor of shape (bs, lang_len, lang_dim)
        """

        bs, num_img, img_feat_dim, h, w = img.shape
        num_pat_img = h // self.img_patch_size
        assert num_img == self.num_cameras
        # assert img_feat_dim == self.img_feat_dim
        assert h == w == self.img_size

        img = img.view(bs * num_img, img_feat_dim, h, w)
        # preprocess
        # (bs * num_img, im_channels, h, w)
        d0 = self.input_preprocess(img)

        # (bs * num_img, im_channels, h, w) ->
        # (bs * num_img, im_channels, h / img_patch_strid, w / img_patch_strid) patches
        ins = self.patchify(d0)
        # (bs, im_channels, num_img, h / img_patch_strid, w / img_patch_strid) patches
        ins = (
            ins.view(
                bs,
                num_img,
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
        if self.add_lang: # ! language as extra sequence
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
        if self.self_cross_ver == 0:
            # self-attention layers
            for self_attn, self_ff in self.layers:
                x = self_attn(x) + x
                x = self_ff(x) + x

        elif self.self_cross_ver == 1:
            lx, imgx = x[:, :num_lang_tok], x[:, num_lang_tok:]

            # within image self attention
            imgx = imgx.reshape(bs * num_img, num_pat_img * num_pat_img, -1)
            for self_attn, self_ff in self.layers[: len(self.layers) // 2]:
                imgx = self_attn(imgx) + imgx
                imgx = self_ff(imgx) + imgx

            imgx = imgx.view(bs, num_img * num_pat_img * num_pat_img, -1)
            x = torch.cat((lx, imgx), dim=1)
            # cross attention
            for self_attn, self_ff in self.layers[len(self.layers) // 2 :]:
                x = self_attn(x) + x
                x = self_ff(x) + x

        else:
            assert False

        # append language features as sequence
        if self.add_lang:
            # throwing away the language embeddings
            x = x[:, num_lang_tok:]
        x = self.fc_aft_attn(x)

        # reshape back to orginal size
        x = x.view(bs, *ins_orig_shape[1:-1], x.shape[-1])  # [B, num_img, np, np, 128]
        x = rearrange(x, "b ... d -> b d ...")  # [B, 128, num_img, np, np]

        feat = []
        _feat = torch.max(torch.max(x, dim=-1)[0], dim=-1)[0]
        _feat = _feat.view(bs, -1)
        feat.append(_feat)

        x = (
            x.transpose(1, 2)
            .clone()
            .view(
                bs * self.num_cameras, self.input_dim_before_seq, num_pat_img, num_pat_img
            )
        )

        u0 = self.up0(x)
        u0 = torch.cat([u0, d0], dim=1)
        u = self.final(u0)

        # translation decoder
        trans = self.trans_decoder(u).view(bs, self.num_cameras, h, w)

        hm = F.softmax(trans.detach().view(bs, self.num_cameras, h * w), 2).view(
            bs * self.num_cameras, 1, h, w
        )

        _feat = torch.sum(hm * u, dim=[2, 3])
        _feat = _feat.view(bs, -1)
        feat.append(_feat)
        feat = torch.cat(feat, dim=-1)
        feat = self.feat_fc(feat)
        out = {"screen_logits": trans, "feature": feat}
        return out

    def reset(self):
        self.renderer.reset()


class PolicyNetwork(nn.Module):
    def __init__(self, cfg: DictConfig, env_cfg: DictConfig, render_device="cuda:0"):
        super().__init__()
        self.env_cfg = env_cfg

        # for verifying the input
        self.img_feat_dim = cfg.img_feat_dim
        self.add_proprio = cfg.add_proprio
        self.proprio_dim = cfg.proprio_dim
        self.add_lang = cfg.add_lang
        if self.add_lang:
            self.lang_emb_dim, self.lang_max_seq_len = cfg.lang_dim, cfg.lang_len
        else:
            self.lang_emb_dim, self.lang_max_seq_len = 0, 0

        self.renderer: CubePointCloudRenderer = CubePointCloudRenderer(
            device=render_device,
            img_size=(cfg.img_size, cfg.img_size),
            with_depth=cfg.add_depth,
        )
        self.num_cameras = self.renderer.num_cameras
        self.proprio_dim = cfg.proprio_dim
        self.img_size = cfg.img_size

        self.mvt1 = MultiViewTransformer(cfg, renderer=self.renderer)
    
    def verify(
        self,
        pc,
        img_feat,
        proprio,
        lang_emb,
    ):
        bs = len(pc)
        assert bs == len(img_feat)

        for _pc, _img_feat in zip(pc, img_feat):
            np, x1 = _pc.shape
            np2, x2 = _img_feat.shape
            assert np == np2
            assert x1 == 3
            assert x2 == self.img_feat_dim

        if self.add_proprio:
            bs3, x3 = proprio.shape
            assert bs == bs3
            assert x3 == self.proprio_dim, "Does not support proprio of shape {proprio.shape}"
        else:
            assert proprio is None, "Invalid input for proprio={proprio}"

        if self.add_lang:
            bs4, x4, x5 = lang_emb.shape
            assert bs == bs4
            assert x4 == self.lang_max_seq_len, "Does not support lang_emb of shape {lang_emb.shape}"
            assert  x5 == self.lang_emb_dim, "Does not support lang_emb of shape {lang_emb.shape}"
        else:
            assert (lang_emb is None) or torch.all(lang_emb == 0)
    

    def render(self, pc, img_feat, augmentation=0):
        mvt = self.mvt1
        with torch.no_grad():
            if mvt.add_corr:
                img = [
                    self.renderer(
                        _pc,
                        torch.cat((_pc, _img_feat), dim=-1),
                    ).unsqueeze(0)
                    for (_pc, _img_feat) in zip(
                        pc, img_feat
                    )
                ]
            else:
                img = [
                    self.renderer(
                        _pc,
                        _img_feat,
                    ).unsqueeze(0)
                    for (_pc, _img_feat) in zip(
                        pc, img_feat
                    )
                ]

            img = torch.cat(img, 0)
            img = img.permute(0, 1, 4, 2, 3)

            # image augmentation
            if augmentation != 0 and self.training:
                stdv = augmentation * torch.rand(1, device=img.device)
                # values in [-stdv, stdv]
                noise = stdv * ((2 * torch.rand(*img.shape, device=img.device)) - 1)
                img = torch.clamp(img + noise, -1, 1)

            if mvt.add_pixel_loc:
                bs = img.shape[0]
                pixel_loc = mvt.pixel_loc.to(img.device)
                img = torch.cat(
                    (img, pixel_loc.unsqueeze(0).repeat(bs, 1, 1, 1, 1)), dim=2
                )
        return img

    def forward(
        self,
        pc,
        img_feat,
        proprio=None,
        lang_emb=None,
        augmentation=0,
        **kwargs,
    ):
        """
        :param pc: list of tensors, each tensor of shape (num_points, 3)
        :param img_feat: list tensors, each tensor of shape
            (bs, num_points, img_feat_dim)
        :param proprio: tensor of shape (bs, priprio_dim)
        :param lang_emb: tensor of shape (bs, lang_len, lang_dim)
        """

        self.verify(pc, img_feat, proprio, lang_emb)
        img = self.render(pc, img_feat, augmentation=augmentation)
        out = self.mvt1(img=img, proprio=proprio, lang_emb=lang_emb, **kwargs)
        return out
    
    def reset(self):
        self.renderer.reset()


class Policy:
    def __init__(self, network: nn.Module, rvt_cfg: DictConfig, log_dir=""):
        env_cfg = network.env_cfg
        self._network = network
        self._num_rotation_classes = rvt_cfg.num_rotation_classes
        self._rotation_resolution = 360 / self._num_rotation_classes
        self._lr = rvt_cfg.lr
        self._image_resolution = [env_cfg.image_size, env_cfg.image_size]
        self._lambda_weight_l2 = rvt_cfg.lambda_weight_l2
        self._transform_augmentation = rvt_cfg.transform_augmentation
        self._place_with_mean = rvt_cfg.place_with_mean
        self._transform_augmentation_xyz = torch.from_numpy(
            np.array(rvt_cfg.transform_augmentation_xyz)
        )
        self._transform_augmentation_rpy = rvt_cfg.transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = self._rotation_resolution
        self._optimizer_type = rvt_cfg.optimizer_type
        self.gt_hm_sigma = rvt_cfg.gt_hm_sigma
        self.augmentation_ratio = rvt_cfg.augmentation_ratio
        self.add_rgc_loss = rvt_cfg.add_rgc_loss
        self.add_lang = rvt_cfg.add_lang
        self.log_dir = log_dir
        self.warmup_steps = rvt_cfg.warmup_steps
        self.lr_cos_dec = rvt_cfg.lr_cos_dec
        self.cos_dec_max_step = rvt_cfg.cos_dec_max_step
        self.scene_bounds = env_cfg.scene_bounds
        self.cameras = env_cfg.cameras
        self.move_pc_in_bound = rvt_cfg.move_pc_in_bound

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        if isinstance(self._network, DistributedDataParallel):
            self._net_mod = self._network.module
        else:
            self._net_mod = self._network

        self.num_all_rot = self._num_rotation_classes * 3

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device

        if self._optimizer_type == "lamb":
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

    # copied from per-act and removed the translation part
    def get_one_hot_expert_gt_actions(
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

    def get_logits(self, out, dims, only_pred=False):
        """
        :param out: output of mvt
        :param dims: tensor dimensions (bs, nc, h, w)
        :param only_pred: some speedupds if the q values are meant only for
            prediction
        :return: tuple of trans_q, rot_q, grip_q and coll_q that is used for
            training and preduction
        """
        bs, nc, h, w = dims
        assert isinstance(only_pred, bool)
        q_trans = out["screen_logits"].view(bs, nc, h * w).transpose(1, 2) # (bs, h*w, nc)
        if not only_pred:
            q_trans = q_trans.clone()

        # (bs, 218)
        rot_q = out["feature"].view(bs, -1)[:, 0 : self.num_all_rot]
        grip_q = out["feature"].view(bs, -1)[:, self.num_all_rot : self.num_all_rot + 2]
        # (bs, 2)
        collision_q = out["feature"].view(bs, -1)[:, self.num_all_rot + 2 : self.num_all_rot + 4]
        return q_trans, rot_q, grip_q, collision_q

    def update(
        self,
        replay_sample: dict,
    ) -> dict:
        backprop = self._network.training
        assert replay_sample["gripper_pose"].shape[1:] == (7, )
        assert replay_sample["lang_goal_embs"].shape[1:] == (77, 512)
        assert replay_sample["low_dim_state"].shape[1:] == (self._net_mod.proprio_dim,)

        # sample
        action_grip = replay_sample["gripper_action"].int() # (b,) of int
        action_ignore_collisions = replay_sample["ignore_collisions"].view(-1, 1).int()  # (b, 1) of int
        action_gripper_pose = replay_sample["gripper_pose"]  # (b, 7)
        action_trans_con = action_gripper_pose[:, 0:3]  # (b, 3)
        # rotation in quaternion xyzw
        action_rot = action_gripper_pose[:, 3:7]  # (b, 4)
        lang_goal_embs = replay_sample["lang_goal_embs"].float()
        proprio = replay_sample["low_dim_state"]  # (b, 4/3)

        obs, pcd = preprocess_images_in_batch(replay_sample, self.cameras)

        with torch.no_grad():
            pc, img_feat = flatten_img_pc_to_points(obs, pcd)
            if self._transform_augmentation and backprop:
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
            wpt = [x[:3] for x in action_trans_con]

            wpt_local = []
            rev_trans = []
            for _pc, _wpt in zip(pc, wpt):
                a, b = place_pc_in_cube(_pc, _wpt,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                wpt_local.append(a.unsqueeze(0))
                rev_trans.append(b)

            wpt_local = torch.cat(wpt_local, axis=0)

            pc = [
                place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0]
                for _pc in pc
            ]

            bs = len(pc)
            nc = self._net_mod.num_cameras
            h = w = self._net_mod.img_size

        out = self._network(
            pc=pc,
            img_feat=img_feat,
            proprio=proprio,
            lang_emb=lang_goal_embs,
            augmentation=self.augmentation_ratio if backprop and (self.augmentation_ratio != 0) else 0,
        )

        # out_ref = torch.load('/home/xz653/Desktop/out-network-ref.pth')
        # out['feature'] = out_ref['feat']
        # out['screen_logits'] = out_ref['trans']

        q_trans, rot_q, grip_q, collision_q = self.get_logits(out, dims=(bs, nc, h, w))

        (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,  # (bs, 2)
            action_collision_one_hot,  # (bs, 2)
        ) = self.get_one_hot_expert_gt_actions(
            bs, action_rot, action_grip, action_ignore_collisions, device=self._device
        )
        action_trans = self.get_translation_action(
            wpt_local, dims=(bs, nc, h, w)
        )

        loss_log = {}
        trans_loss = self._cross_entropy_loss(q_trans, action_trans).mean()
        rot_loss_x = rot_loss_y = rot_loss_z = 0.0
        grip_loss = 0.0
        collision_loss = 0.0
        if self.add_rgc_loss:
            rot_loss_x = self._cross_entropy_loss(
                rot_q[:, 0 * self._num_rotation_classes : 1 * self._num_rotation_classes],
                action_rot_x_one_hot.argmax(-1),
            ).mean()
            rot_loss_y = self._cross_entropy_loss(
                rot_q[:, 1 * self._num_rotation_classes : 2 * self._num_rotation_classes],
                action_rot_y_one_hot.argmax(-1),
            ).mean()
            rot_loss_z = self._cross_entropy_loss(
                rot_q[:, 2 * self._num_rotation_classes : 3 * self._num_rotation_classes],
                action_rot_z_one_hot.argmax(-1),
            ).mean()
            grip_loss = self._cross_entropy_loss(grip_q, action_grip_one_hot.argmax(-1)).mean()
            collision_loss = self._cross_entropy_loss(collision_q, action_collision_one_hot.argmax(-1)).mean()

        total_loss = (
            trans_loss
            + rot_loss_x
            + rot_loss_y
            + rot_loss_z
            + grip_loss
            + collision_loss
        )
        if backprop:
            self._optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            self._optimizer.step()
            self._lr_sched.step()

        loss_log = {
            "total_loss": total_loss.item(),
            "trans_loss": trans_loss.item(),
            "rot_loss_x": rot_loss_x.item(),
            "rot_loss_y": rot_loss_y.item(),
            "rot_loss_z": rot_loss_z.item(),
            "grip_loss": grip_loss.item(),
            "collision_loss": collision_loss.item(),
            "lr": self._optimizer.param_groups[0]["lr"],
        }
        return loss_log

    @torch.no_grad()
    def act(
        self, step: int, observation: dict
    ) -> ActResult:
        if self.add_lang:
            lang_goal_tokens = observation.get("lang_goal_tokens", None).long()
            _, lang_goal_embs = clip_encode_text(self.clip_model, lang_goal_tokens)
            lang_goal_embs = lang_goal_embs.float()
        else:
            lang_goal_embs = (
                torch.zeros(observation["lang_goal_embs"].shape)
                .float()
                .to(self._device)
            )
        proprio = observation["low_dim_state"]

        obs, pcd = preprocess_images_in_batch(observation, self.cameras)
        pc, img_feat = flatten_img_pc_to_points(obs, pcd)

        pc, img_feat = clamp_pc_in_bound(
            pc, img_feat, self.scene_bounds, skip=not self.move_pc_in_bound
        )
        pc_new = []
        rev_trans = []
        for _pc in pc:
            a, b = place_pc_in_cube(
                _pc,
                with_mean_or_bounds=self._place_with_mean,
                scene_bounds=None if self._place_with_mean else self.scene_bounds,
            )
            pc_new.append(a)
            rev_trans.append(b)
        pc = pc_new

        bs = len(pc)
        nc = self._net_mod.num_cameras
        h = w = self._net_mod.img_size
        out = self._network(
            pc=pc,
            img_feat=img_feat,
            proprio=proprio,
            lang_emb=lang_goal_embs,
        )
        _, rot_q, grip_q, collision_q = self.get_logits(
            out, dims=(bs, nc, h, w), only_pred=True
        )
        pred_wpt, pred_rot_quat, pred_grip, pred_coll = self.derive_prediction(
            out, rot_q, grip_q, collision_q, rev_trans
        )
        continuous_action = np.concatenate(
            (
                pred_wpt[0].cpu().numpy(),
                pred_rot_quat[0],
                pred_grip[0].cpu().numpy(),
                pred_coll[0].cpu().numpy(),
            )
        )
        return ActResult(continuous_action)

    def derive_prediction(
        self,
        out,
        rot_q,
        grip_q,
        collision_q,
        rev_trans,
    ):
        bs = len(out["screen_logits"])
        h = w = self._net_mod.img_size
        nc = self._net_mod.num_cameras
        hm = F.softmax(out["screen_logits"].view(bs, nc, -1), dim=2)
        hm = hm.view(bs, nc, h, w)
        pred_pt = [self._network.renderer.get_most_likely_point_3d(hm[i : i + 1]) for i in range(bs)]
        pred_pt = torch.cat(pred_pt, 0)

        pred_pt_origin_coord = []
        for _pred_pt, _rev_trans in zip(pred_pt, rev_trans):
            pred_pt_origin_coord.append(_rev_trans(_pred_pt))
        pred_pt_origin_coord = torch.cat([x.unsqueeze(0) for x in pred_pt_origin_coord])

        pred_rot = torch.cat((
            rot_q[:, 0 * self._num_rotation_classes : 1 * self._num_rotation_classes].argmax(1, keepdim=True),
            rot_q[:, 1 * self._num_rotation_classes : 2 * self._num_rotation_classes].argmax(1, keepdim=True),
            rot_q[:, 2 * self._num_rotation_classes : 3 * self._num_rotation_classes].argmax(1, keepdim=True),
        ), dim=-1)
        pred_rot_quat = math3d.discrete_euler_to_quaternion(pred_rot.cpu(), self._rotation_resolution)
        pred_grip = grip_q.argmax(1, keepdim=True)
        pred_coll = collision_q.argmax(1, keepdim=True)
        return pred_pt_origin_coord, pred_rot_quat, pred_grip, pred_coll

    def get_translation_action(
        self,
        wpt_local, # this is groundtruth 3d point
        dims,
    ):
        bs, nc, h, w = dims
        wpt_img = self._net_mod.renderer.points3d_to_screen2d(wpt_local.unsqueeze(1))
        assert wpt_img.shape[1] == 1
        wpt_img = wpt_img.squeeze(1)  # (bs, num_img, 2)
        action_trans = generate_heatmap_from_screen_pts(
            wpt_img.reshape(-1, 2), #! just the winning points
            (h, w),
            sigma=self.gt_hm_sigma,
            thres_sigma_times=3,
        )
        action_trans = action_trans.view(bs, nc, h * w).transpose(1, 2).clone()
        return action_trans

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()

    def reset(self, **kwargs):
        self._network.reset()

    def load(self, model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        epoch = checkpoint.get("epoch", checkpoint.get("step", None))
        model = self._network
        optimizer = self._optimizer
        lr_sched = self._lr_sched
        if isinstance(model, DistributedDataParallel):
            model.module.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        else:
            print("WARNING: No optimizer_state in checkpoint" "KNOW WHAT YOU ARE DOING!!")
        if "lr_sched_state" in checkpoint:
            lr_sched.load_state_dict(checkpoint["lr_sched_state"])
        else:
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