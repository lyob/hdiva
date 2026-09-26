from dataclasses import dataclass, fields
from typing import Literal, Optional, Tuple

project_dir = "/mnt/home/blyo1/hdiva"


@dataclass
class IMAE_CelebA_64_Training_Config:
    """model"""

    model_name: str = "ddpm_celeba_color_64"

    """dataset parameters"""
    image_dim: int = 64
    dataset_name: str = "celeba_color"
    dataset_size: int = 162770

    """Denoiser parameters"""
    timestep_dist: str = "uniform"
    num_timesteps: int = 1000
    beta_minmax: tuple = (1e-4, 2e-2)
    denoiser_target: str = "noise"  # "image", "residual", "noise"
    sigma_minmax: tuple = (0.0001, 0.9999)
    noise_schedule: str = "cosine_in_alpha_bar"  # 'linear_in_beta', 'cosine_in_alpha_bar'
    denoiser_type: str = "unet"

    denoiser_act_fn: str = "relu"
    num_channels: int = 3  # 3 for color images, 1 for grayscale images
    num_kernels: int = 256
    kernel_size: int = 3
    padding: int = 1
    bias: bool = False
    time_embedding_method: str = "as_input"  # as_input, as_channel
    time_channels: int = 64
    num_blocks: int = 3  # number of downsampling/upsampling blocks in UNet
    num_enc_conv: int = 2  # number of conv layers in each downsampling block
    num_mid_conv: int = 2  # number of conv layers in middle block
    num_dec_conv: int = 2  # number of conv layers in each upsampling
    pool_window: int = 2

    """infnet"""
    latent_dim: int = 512
    num_kernels_rec: int = 32
    activation_rec: str = "relu"  # gelu or relu
    bias_rec: bool = False
    num_blocks_rec: int = 3
    num_enc_conv_rec: int = 2
    num_mid_conv_rec: int = 2
    pool_window_rec: int = 2
    kernel_size_rec: int = 3
    padding_rec: int = 1
    downsample_in_mid_block_rec: bool = False

    """diva"""
    weighted_MSE: bool = False
    reduction: str = "mean"  # "mean" or "sum"

    """kl"""
    kl_weight_min: float = 1e-11
    kl_weight_max: float = 1e-11
    kl_annealing_epochs: int = 200
    kl_annealing_schedule: str = "cosine"  # "linear" or "cosine" or "linear_in_log"

    """training"""
    num_epochs: int = 500
    train_batch_size_per_gpu: int = 512
    lr_schedule: str = "cosine"
    lr_init: float = 4e-3
    lr_final: float = 2e-3
    lr_num_warmup_epochs: int = 1000
    seed: int = 43

    """pretraining"""
    resume_from_checkpoint: bool = False  # master switch for using checkpoint
    use_pretrained_denoiser_only: bool = False  # use a pretrained ddpm or denoiser to initialize the denoiser, rather than use the entire pretrained diva model
    train_infnet_only: bool = False  # if true, freeze the denoiser weights and only train the infnet
    pretrained_project_name: str = "diva_manual_celeba_color_64"
    pretrained_model_num: int = 3   
    pretrained_artifact_id: str = "v29"

    precision: str = "32"
    project_dir: str = project_dir
    model_checkpoint_dir: str = f"{project_dir}/c_training/lightning_checkpoints"
    data_cache_dir: str = f"{project_dir}/a_datasets/celeba_color/"

    """cluster"""
    strategy: str = "ddp"  # "ddp" or "deepspeed_stage_2"
    # strategy: str = "deepspeed_stage_2"  # "ddp" or "deepspeed_stage_2"
    # zero_opt_stage: int = 2
    num_nodes: int = 2
    num_gpus_per_node: int = 4

    """logging"""
    log_every_n_steps: int = 10
    checkpoint_every_n_epochs: int = 50

    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})
