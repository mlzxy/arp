import math
from copy import deepcopy
from collections import deque
from itertools import chain
from PIL import Image
from typing import Callable, Tuple, List, Dict

import einops
from argparse import Namespace
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.transforms.functional import to_pil_image
from torchvision.ops.misc import FrozenBatchNorm2d
from .configuration import ARPConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
from typing import Union, Optional
import torch.distributions as D
import arp
from lerobot.common.utils.nn import SinusoidalPositionEmbedding2d, generate_heatmap_from_screen_pts, to_red_heatmap, denorm_rgb


def segmented_range_list(start, end, segment_size):
    if (end - start) % segment_size != 0:
        end = start + int(math.ceil((end - start) / segment_size)) * segment_size
    multiple = (end - start) // segment_size
    lst = [start + i for i in range(multiple) for j in range(segment_size) ]
    return lst[:end - start]


class ARPPolicy(nn.Module, PyTorchModelHubMixin):
    name = "arp"

    def __init__(
        self,
        config: Optional[ARPConfig] = None,
        dataset_stats: Optional[Dict[str, Dict[str, Tensor]]] = None,
    ):
        super().__init__()
        if config is None:
            config = ARPConfig()
        self.config: ARPConfig = config

        self.normalize_inputs = Normalize(
            config.input_shapes, config.input_normalization_modes, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.model = AutoregressiveModel(config)
        self.expected_image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]
        self.reset()

    def reset(self):
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: Dict[str, Tensor]) -> Tensor:
        self.eval()
        batch = self.normalize_inputs(batch)
        batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
        if len(self._action_queue) == 0:
            actions = self.model(batch)[0][:, : self.config.n_action_steps]
            actions = self.unnormalize_outputs({"action": actions})["action"]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        batch = self.normalize_inputs(batch)
        batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
        batch = self.normalize_targets(batch)
        _, loss = self.model(batch)
        return {'loss_dict': loss}


class AutoregressiveModel(nn.Module):

    def __init__(self, config: ARPConfig):
        super().__init__()
        self.config = config

        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
            weights=config.pretrained_backbone_weights,
            norm_layer=FrozenBatchNorm2d,
        )
        self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer( d_model=config.dim_model,
                    dropout=config.dropout,
                    norm_first=config.pre_norm,
                    batch_first=False,
                    activation=config.feedforward_activation,
                    dim_feedforward=config.dim_feedforward,
                    nhead=config.n_heads), 
            num_layers=config.n_encoder_layers, 
            norm=nn.LayerNorm(config.dim_model)
        )
        
        # NOTE: policy definition
        self.guide_pts_downsample = config.guide_pts_downsample
        self.guide_pts_heatmap_sigma = config.guide_pts_heatmap_sigma
        arp_cfg = Namespace(**config.arp_cfg)
        action_token = arp.TokenType.make(name='action', is_continuous=True, dim=14, embedding='linear', predictor='gmm', 
                                          predictor_kwargs={'num_latents': config.num_latents})
        origin_stride = 32
        guide_token_right = arp.TokenType.make(name='guide-pt-right', is_continuous=True, dim=2, 
                                embedding='position_2d', predictor="upsample_from_2d_attn", 
                                predictor_kwargs={'attn_with': 'visual-featmap', 'upscale_ratio': origin_stride // self.guide_pts_downsample, 'label_name': 'smooth-heatmap-right', 'corr_dim': config.guide_pts_corr_dim})
        guide_token_left = arp.TokenType.make(name='guide-pt-left', is_continuous=True, dim=2, 
                                embedding='position_2d', predictor="upsample_from_2d_attn", 
                                predictor_kwargs={'attn_with': 'visual-featmap', 'upscale_ratio': origin_stride // self.guide_pts_downsample, 'label_name': 'smooth-heatmap-left', 'corr_dim': config.guide_pts_corr_dim})                                                             

        # self.linear_encoder_to_arp = nn.Linear(config.dim_model, arp_cfg.n_embd)
        self.policy = arp.AutoRegressivePolicy(arp.ModelConfig(
            n_embd=arp_cfg.n_embd,
            embd_pdrop=arp_cfg.embd_pdrop,
            max_seq_len=arp_cfg.max_seq_len,
            max_chunk_size=arp_cfg.max_seq_len,
            layers=[
                arp.LayerType.make(**arp_cfg.layer_cfg, condition_on='visual-tokens')
            ] * arp_cfg.num_layers,
            tokens=[
                arp.TokenType.make(name='state', is_control=True, is_continuous=True, dim=14, embedding='linear'),
                action_token,
                guide_token_left, guide_token_right
            ]
        ))

        # NOTE: encoder related
        self.encoder_robot_state_input_proj = nn.Linear(
            config.input_shapes["observation.state"][0], config.dim_model
        )
        self.encoder_img_feat_input_proj = nn.Conv2d(
            backbone_model.fc.in_features, config.dim_model, kernel_size=1
        )
        # Transformer encoder positional embeddings.
        self.encoder_robot_pos_embed = nn.Embedding(1, config.dim_model)
        self.encoder_cam_feat_pos_embed = SinusoidalPositionEmbedding2d(config.dim_model // 2)

        self.action_chunk_size = config.action_chunk_size if config.action_chunk_size > 0 else config.chunk_size
        self.guide_chunk_size = config.guide_chunk_size if config.guide_chunk_size > 0 else 1

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Union[Tuple[Tensor, Tensor], Tuple[None, None]]]:
        num_guide_points = self.config.num_guide_points
        batch_size, dev = batch["observation.images"].shape[0], batch["observation.images"].device
        H, W = batch["observation.images"].shape[-2:]
        all_cam_features = []
        all_cam_pos_embeds = []
        images = batch["observation.images"] # [8, 1, 3, 480, 640]

        for cam_index in range(images.shape[-4]): # 1
            cam_features = self.backbone(images[:, cam_index])["feature_map"]
            cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
            cam_features = self.encoder_img_feat_input_proj(cam_features)  # (B, C, h, w)
            all_cam_features.append(cam_features) # [8, 512, 15, 20]
            all_cam_pos_embeds.append(cam_pos_embed)

        encoder_in = torch.cat(all_cam_features, axis=-1)
        cam_pos_embed = torch.cat(all_cam_pos_embeds, axis=-1)

        robot_state_embed = self.encoder_robot_state_input_proj(batch["observation.state"])  # (B, C)
        encoder_in =  torch.cat([robot_state_embed[None, ...],  einops.rearrange(encoder_in, "b c h w -> (h w) b c")])
        pos_embed = torch.cat([self.encoder_robot_pos_embed.weight.unsqueeze(1), cam_pos_embed.flatten(2).permute(2, 0, 1)])

        encoder_out = self.encoder(encoder_in + pos_embed)
        encoder_out = encoder_out.permute(1, 0, 2)  # (B, S, C)
        # encoder_out = self.linear_encoder_to_arp(encoder_out)
        visual_featmap = encoder_out[:, 1:, :].permute(0, 2, 1).reshape(batch_size, -1, 15, 20) # (B, C, H, W) 

        actions, loss_dict = None, None       
        #region autoregressive policy
        tkname2id = self.policy.token_name_2_ids
        tk_ids = [tkname2id['state']] + [tkname2id['action']] * self.config.chunk_size \
            + [tkname2id['guide-pt-left']] * num_guide_points + \
            [tkname2id['guide-pt-right']] * num_guide_points

        # chk_ids = [0] 
        # action_chk_ids = segmented_range_list(1, 1 + self.config.chunk_size, self.action_chunk_size)
        # guide_chk_ids = segmented_range_list(1 + max(action_chk_ids), 1 + max(action_chk_ids) + num_guide_points * 2, self.guide_chunk_size)
        # chk_ids += (action_chk_ids + guide_chk_ids)
        chk_ids = [0] + [1] * (self.config.chunk_size + num_guide_points * 2) #list(range(1, 1 + num_guide_points * 2)) + [num_guide_points * 2 + 1] * self.config.chunk_size

        if self.training:
            visual_featmap = visual_featmap.unsqueeze(1).repeat(1, num_guide_points, 1, 1, 1).flatten(0, 1) 
            tk_is_pad_mask = torch.cat([batch['action_is_pad'], 
                        torch.full([batch_size, 1 + num_guide_points * 2], fill_value=False, device=dev)], dim=1)
            state = batch['observation.state'][:, None, :]

            guide_pts_left = batch['left_pts_2d'][:, :, -1] / self.guide_pts_downsample # B, L, 2
            guide_pts_right = batch['right_pts_2d'][:, :, -1] / self.guide_pts_downsample # B, L, 2

            guide_pts_left_lst, guide_pts_right_lst = [], []
            for i in range(batch_size):
                guide_pts_left_i = guide_pts_left[i][~batch['action_is_pad'][i]]
                guide_pts_right_i = guide_pts_right[i][~batch['action_is_pad'][i]]

                guide_pts_left_i = F.interpolate(guide_pts_left_i.permute(1, 0)[None, :, :], size=num_guide_points, mode='linear', align_corners=True).permute(0, 2, 1)
                guide_pts_right_i = F.interpolate(guide_pts_right_i.permute(1, 0)[None, :, :], size=num_guide_points, mode='linear', align_corners=True).permute(0, 2, 1)

                guide_pts_right_lst.append(guide_pts_right_i)
                guide_pts_left_lst.append(guide_pts_left_i)
            
            guide_pts_left = torch.cat(guide_pts_left_lst, dim=0).round()
            guide_pts_right = torch.cat(guide_pts_right_lst, dim=0).round()

            W //= self.guide_pts_downsample
            H //= self.guide_pts_downsample
            heatmap_right = generate_heatmap_from_screen_pts(guide_pts_right.flatten(0, 1), (H, W), 
                                            sigma=self.guide_pts_heatmap_sigma, thres_sigma_times=3).reshape(batch_size, num_guide_points, H, W)
            heatmap_left = generate_heatmap_from_screen_pts(guide_pts_left.flatten(0, 1), (H, W), 
                                            sigma=self.guide_pts_heatmap_sigma, thres_sigma_times=3).reshape(batch_size, num_guide_points, H, W)
            
            # NOTE: DEBUG CODE
            # hm_img = to_pil_image(to_red_heatmap(heatmap_right[0]).sum(dim=0))
            # hm_img = hm_img.resize((640, 480))
            # img = to_pil_image(denorm_rgb(batch["observation.images"][0, 0]))
            # Image.blend(img, hm_img, 0.5).save('./outputs/test.jpg')

            tk_vals = arp.cat_uneven_blc_tensors(batch['observation.state'][:, None, :], batch['action'], guide_pts_left, guide_pts_right) 
            tk_ids = torch.as_tensor(tk_ids).to(dev)[None, :, None].repeat(batch_size, 1, 1)
            tks = torch.cat([tk_vals, tk_ids], dim=-1) 
            chk_ids = torch.as_tensor(chk_ids, device=dev)[None, :]
            loss_dict = self.policy.compute_loss(tks, chk_ids, contexts={ 'visual-tokens': encoder_out, 
                        'visual-featmap': visual_featmap, 
                        'smooth-heatmap-right': heatmap_right.flatten(0, 1), 
                        'smooth-heatmap-left': heatmap_left.flatten(0, 1) }, valid_tk_mask=~tk_is_pad_mask)
        else:
            state = batch['observation.state'][:, None, :]
            prompt_tk_vals = state
            prompt_tk_ids =  torch.full([batch_size, 1, 1], fill_value=tkname2id['state']).to(dev) # control (state)
            prompt_tks = torch.cat([prompt_tk_vals, prompt_tk_ids], dim=-1)
            future_tk_reg_ids = [{'tk_id': tk_id, 'chk_id': chk_id} for tk_id, chk_id in zip(tk_ids[1:101], chk_ids[1:101])]
            pred_tks = self.policy.generate(prompt_tks, future_tk_reg_ids, 
                                    contexts={ 'visual-tokens': encoder_out, 'visual-featmap': visual_featmap }, sample=self.config.sample)
            actions = pred_tks[:, :self.config.n_action_steps, :-1].reshape(batch_size, self.config.n_action_steps, 14) 
        #endregion #######################
    
        if actions is not None:
            actions = actions[:, :self.config.n_action_steps_eval]

        return actions, loss_dict

