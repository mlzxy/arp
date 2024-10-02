from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class ARPConfig:
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100
    n_action_steps_eval: int = -1

    input_shapes: Dict[str, List[int]] = field(
        default_factory=lambda: {
            "observation.images.top": [3, 480, 640],
            "observation.state": [14],
        }
    )
    output_shapes: Dict[str, List[int]] = field(
        default_factory=lambda: {
            "action": [14],
        }
    )

    # Normalization / Unnormalization
    input_normalization_modes: Dict[str, str] = field(
        default_factory=lambda: {
            "observation.images.top": "mean_std",
            "observation.state": "mean_std",
        }
    )
    output_normalization_modes: Dict[str, str] = field(
        default_factory=lambda: {
            "action": "mean_std",
        }
    )

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: Optional[str] = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    dropout: float = 0.1

    num_latents: int = 1
    num_guide_points: int = 10
    guide_pts_downsample: int = 1
    guide_pts_corr_dim: int = 64
    guide_pts_heatmap_sigma: float = 1.5

    arp_cfg: dict = field(default_factory=lambda: {})
    sample: bool = False

    guide_chunk_size: int = -1
    action_chunk_size: int = -1

    def __post_init__(self):
        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
  
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )
        
        if self.n_action_steps_eval == -1:
            self.n_action_steps_eval = self.n_action_steps

        self.arp_cfg['max_seq_len'] = 1 + self.chunk_size + self.num_guide_points * 2


