#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://arxiv.org/abs/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

import math
from collections import deque
from itertools import chain
from typing import Callable, List, Tuple, Dict
from copy import deepcopy
import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
from typing import Optional, Union
from lerobot.common.policies.aloha_diffusion_policy.configuration import Config as ACTConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
import lerobot.common.policies.aloha_diffusion_policy.tinydiffp as dfp



class Policy(nn.Module, PyTorchModelHubMixin):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://arxiv.org/abs/2304.13705, code: https://github.com/tonyzhaozh/act)
    """

    name = "act"

    def __init__(
        self,
        config: Optional[ACTConfig] = None,
        dataset_stats: Optional[Dict[str, Dict[str, Tensor]]] = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__()
        if config is None:
            config = ACTConfig()
        self.config: ACTConfig = config
        self.normalize_inputs = Normalize(
            config.input_shapes, config.input_normalization_modes, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.model = ACT(config)
        self.expected_image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_momentum is not None:
            self._ensembled_actions = None
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad
    def select_action(self, batch: Dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()
        batch = self.normalize_inputs(batch)
        batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)

        # If we are doing temporal ensembling, keep track of the exponential moving average (EMA), and return
        # the first action.
        if self.config.temporal_ensemble_momentum is not None:
            actions = self.model(batch)[0]  # (batch_size, chunk_size, action_dim)
            actions = self.unnormalize_outputs({"action": actions})["action"]
            if self._ensembled_actions is None:
                # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
                # time step of the episode.
                self._ensembled_actions = actions.clone()
            else:
                # self._ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
                # the EMA update for those entries.
                alpha = self.config.temporal_ensemble_momentum
                self._ensembled_actions = alpha * self._ensembled_actions + (1 - alpha) * actions[:, :-1]
                # The last action, which has no prior moving average, needs to get concatenated onto the end.
                self._ensembled_actions = torch.cat([self._ensembled_actions, actions[:, -1:]], dim=1)
            # "Consume" the first action.
            action, self._ensembled_actions = self._ensembled_actions[:, 0], self._ensembled_actions[:, 1:]
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.model(batch)[0][:, : self.config.n_action_steps]

            # TODO(rcadene): make _forward return output dictionary?
            actions = self.unnormalize_outputs({"action": actions})["action"]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
        batch = self.normalize_targets(batch)
        _, loss_dict = self.model(batch)
        return loss_dict


class ACT(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config
        self.use_input_state = "observation.state" in config.input_shapes

        # Backbone for image feature extraction.
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
            weights=config.pretrained_backbone_weights,
            norm_layer=FrozenBatchNorm2d,
        )
        self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = ACTEncoder(config)

        self.noise_scheduler = dfp.DDPMScheduler(
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            beta_start=0.0001,
            clip_sample=True,
            num_train_timesteps=100,
            prediction_type='sample', #'epsilon',
            variance_type='fixed_small'
        )

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, robot_state, image_feature_map_pixels].
        if self.use_input_state:
            self.encoder_robot_state_input_proj = nn.Linear(
                config.input_shapes["observation.state"][0], config.dim_model
            )
        self.encoder_img_feat_input_proj = nn.Conv2d(
            backbone_model.fc.in_features, config.dim_model, kernel_size=1
        )
        # Transformer encoder positional embeddings.
        self.encoder_robot_pos_embed = nn.Embedding(1, config.dim_model)
        self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        self.diff_model = dfp.TransformerForDiffusion(
            input_dim=config.output_shapes["action"][0],
            output_dim=config.output_shapes["action"][0],
            horizon=config.chunk_size,
            n_obs_steps=1 + 15 * 20,
            cond_dim=config.dim_model,
            n_layer=4,
            n_head=8,
            n_emb=512,
            p_drop_emb=0.1,
            p_drop_attn=0.1,
            causal_attn=False,
            time_as_cond=True,
            obs_as_cond=True,
            n_cond_layers=0
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.diff_model.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Union[Tuple[Tensor, Tensor], Tuple[None, None]]]:
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        `batch` should have the following structure:

        {
            "observation.state": (B, state_dim) batch of robot states.
            "observation.images": (B, n_cameras, C, H, W) batch of images.
            "action" (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.
        """
        if self.training:
            assert (
                "action" in batch
            ), "actions must be provided when using the variational objective in training mode."

        batch_size = batch["observation.images"].shape[0]
        device = batch["observation.images"].device
        dtype = batch["observation.images"].dtype

        all_cam_features = []
        all_cam_pos_embeds = []
        images = batch["observation.images"] # [8, 1, 3, 480, 640]
        for cam_index in range(images.shape[-4]): # 1
            cam_features = self.backbone(images[:, cam_index])["feature_map"]
            cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
            cam_features = self.encoder_img_feat_input_proj(cam_features)  # (B, C, h, w)
            all_cam_features.append(cam_features) # [8, 512, 15, 20]
            all_cam_pos_embeds.append(cam_pos_embed)
        # Concatenate camera observation feature maps and positional embeddings along the width dimension.
        encoder_in = torch.cat(all_cam_features, axis=-1)
        cam_pos_embed = torch.cat(all_cam_pos_embeds, axis=-1)

        # Get positional embeddings for robot state and latent.
        if self.use_input_state:
            robot_state_embed = self.encoder_robot_state_input_proj(batch["observation.state"])  # (B, C)

        encoder_in = torch.cat(
            [
                robot_state_embed[None, ...],
                einops.rearrange(encoder_in, "b c h w -> (h w) b c"),
            ]
        )
        pos_embed = torch.cat(
            [
                self.encoder_robot_pos_embed.weight.unsqueeze(1),
                cam_pos_embed.flatten(2).permute(2, 0, 1),
            ],
            axis=0,
        )

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in, pos_embed=pos_embed) # [302, 8, 512]
        encoder_out = encoder_out.transpose(0, 1) # [8, 302, 512]

        actions, loss_dict = None, {}

        if self.training:
            trajectory = batch["action"]
            noise = torch.randn(trajectory.shape, device=trajectory.device)
            bsz = trajectory.shape[0]
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (bsz,), device=trajectory.device
            ).long()
            noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
            pred = self.diff_model(noisy_trajectory, timesteps, encoder_out)
            pred_type = self.noise_scheduler.config.prediction_type # epsilon
            if pred_type == 'epsilon':
                target = noise
            elif pred_type == 'sample':
                target = trajectory
            else:
                raise ValueError(f"Unsupported prediction type {pred_type}") 
            loss_Dict["loss"] = F.mse_loss(pred, target, reduction='mean')
        else:
            trajectory = torch.randn(size=(batch_size, self.config.chunk_size, self.config.output_shapes["action"][0]), dtype=dtype, device=device)
            trajectory = dfp.conditional_sample(self.diff_model, self.noise_scheduler, trajectory, encoder_out, num_inference_steps=100)
            actions = trajectory
        
        return actions, loss_dict




class ACTEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(config.n_encoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Optional[Tensor]  = None, key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # note: [0] to select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed



def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
