from typing import Dict, Tuple
import math
import torch
import einops
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
import diffusion_policy.common.act as act


class VAE_ACT_ImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            # task params
            horizon, 
            n_action_steps, 
            n_obs_steps,
            # image
            crop_shape=(76, 76),
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # arch
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

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

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        dim_model = 512
        
        layer_kwargs = dict(n_heads=4, dropout=0.1, dim_feedforward=4 * dim_model, feedforward_activation="relu")

        self.latent_dim = latent_dim  = 32
        self.obs_encoder = obs_encoder
        self.model = nn.ModuleDict({
            'obs_feature_to_hidden': nn.Linear(obs_feature_dim, dim_model),
            'vae_encoder': act.ACTEncoder(dim_model, 4, pre_norm=False, layer_kwargs=layer_kwargs),
            'encoder': act.ACTEncoder(dim_model, 4, pre_norm=False, layer_kwargs=layer_kwargs),
            'decoder': act.ACTDecoder(dim_model, 4, pre_norm=False, layer_kwargs=layer_kwargs),
            'action_head': nn.Linear(dim_model, action_dim),
            'encoder_latent_input_proj': nn.Linear(latent_dim, dim_model),
            'vae_encoder_action_input_proj': nn.Linear(action_dim, dim_model),
            'vae_encoder_cls_embed': nn.Embedding(1, dim_model),
            'vae_encoder_latent_output_proj': nn.Linear(dim_model, latent_dim * 2),
            
            'encoder_pos_embed': nn.Embedding(n_obs_steps + 1, dim_model),
            'decoder_pos_embed': nn.Embedding(horizon, dim_model),
            'decoder_input_tokens': nn.Embedding(horizon, dim_model)
        })
        self.register_buffer(
                "vae_encoder_pos_enc",
                act.create_sinusoidal_pos_embedding(horizon + 1, dim_model).unsqueeze(0),
        )

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps


    def predict_action(self, batch_or_obs_dict: Dict[str, torch.Tensor], training=False) -> Dict[str, torch.Tensor]:
        nobs = batch_or_obs_dict['obs'] if 'obs' in batch_or_obs_dict else batch_or_obs_dict
        batch_size = len(nobs['image'])
        dev = nobs['image'].device

        nobs = self.normalizer.normalize(nobs)
        horizon = self.horizon
        To = self.n_obs_steps

        if training:
            nactions = self.normalizer['action'].normalize(batch_or_obs_dict['action'])
            assert horizon == nactions.shape[1]
            cls_embed = einops.repeat(
                self.model['vae_encoder_cls_embed'].weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            action_embed = self.model['vae_encoder_action_input_proj'](nactions) 
            vae_encoder_input = torch.cat([cls_embed, action_embed], axis=1)
            pos_embed = self.vae_encoder_pos_enc

            cls_token_out = self.model['vae_encoder'](
                vae_encoder_input.permute(1, 0, 2), # [8, 102, 512]
                pos_embed=pos_embed.permute(1, 0, 2),
            )[0] 
            latent_pdf_params = self.model['vae_encoder_latent_output_proj'](cls_token_out)

            mu = latent_pdf_params[:, : self.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu) 
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.latent_dim], dtype=torch.float32).to(dev)
        
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, T, Do
        cond = nobs_features.reshape(batch_size, To, -1)
        cond = self.model['obs_feature_to_hidden'](cond)
        latent_embed = self.model['encoder_latent_input_proj'](latent_sample)
        
        encoder_in = torch.cat([latent_embed[None, ...], cond.permute(1, 0, 2)])
        pos_emb = self.model['encoder_pos_embed'].weight[:, None, :]
        encoder_out = self.model['encoder'](encoder_in, pos_embed=pos_emb) 
        decoder_in = self.model['decoder_input_tokens'].weight[:, None, :].repeat(1, batch_size, 1)
        decoder_out = self.model['decoder'](decoder_in, encoder_out, encoder_pos_embed=pos_emb,
            decoder_pos_embed=self.model['decoder_pos_embed'].weight.unsqueeze(1))
        decoder_out = decoder_out.transpose(0, 1)
        naction_pred = self.model['action_head'](decoder_out) 
        
        if training:
            kld_loss = act.compute_kld_loss(mu, log_sigma_x2)
            l1_loss = F.l1_loss(naction_pred, nactions)
            return l1_loss + kld_loss 
        else:
            action_pred = self.normalizer['action'].unnormalize(naction_pred)
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
            result = {
                'action': action,
                'action_pred': action_pred
            }
            return result


    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            transformer_weight_decay: float, 
            obs_encoder_weight_decay: float,
            learning_rate: float, 
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = [dict(params=self.model.parameters(),
            weight_decay=transformer_weight_decay)]
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def compute_loss(self, batch):
        return self.predict_action(batch, training=True)
