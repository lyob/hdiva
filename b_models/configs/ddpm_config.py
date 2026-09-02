from dataclasses import dataclass, fields
from typing import Optional, Tuple

project_dir = "/mnt/home/blyo1/hdiva"


@dataclass
class CelebA_64_Config:
    """dataset parameters"""

    image_dim: int = 64
    dataset_name: str = "celeba_color"
    dataset_size: int = 162770
    data_cache_dir: str = f"{project_dir}/a_datasets/celeba_color/"


@dataclass
class DSprites_64_Config:
    """dataset parameters"""

    image_dim: int = 64
    dataset_name: str = "dsprites"
    dataset_size: int = 737280
    data_cache_dir: str = f"{project_dir}/a_datasets/dsprites/"


@dataclass
class CircleInSquareDatasetConfig:
    """circle in square dataset parameters"""

    image_dim: int = 64
    dataset_name: str = "circle_in_square"
    dataset_size: int = 100000
    data_cache_dir: str = f"{project_dir}/a_datasets/circle_in_square/"


@dataclass
class SOSDatasetConfig:
    """sos dataset parameters"""

    image_dim: int = 28
    dataset_name: str = "sos"
    dataset_size: int = 100000
    sigma_x: float = 0.8
    sigma_y: float = 1.5
    holdout_center: bool = False
    data_cache_dir: str = f"{project_dir}/a_datasets/sos/"


@dataclass
class DiskDatasetConfig:
    """disk dataset parameters"""

    image_dim: int = 28
    dataset_name: str = "simple_disks"
    dataset_size: int = 100000  # 1e5
    data_cache_dir: str = f"{project_dir}/a_datasets/disks/"
    foreground: float = 1.
    background: float = 0.
    outer_radius: int = 8
    transition_width: int = 2
    dataset_seed: int = 42
    cx_range: Optional[Tuple[float, float]] = (10, 18)
    cy_range: Optional[Tuple[float, float]] = None


@dataclass
class DenoiserConfig:
    """Denoiser parameters"""

    timestep_dist: str = "uniform"
    num_timesteps: int = 100
    beta_minmax: tuple = (1e-4, 2e-2)
    sigma_minmax: tuple = (0.0001, 0.9999)
    # noise_schedule: 'linear_in_alpha_bar', 'cosine_in_alpha_bar', 'inverted_cosine_in_alpha_bar'
    noise_schedule: str = "linear_in_alpha_bar"
    parameterization: str = "noise"  # "image", "velocity", "noise"
    denoiser_type: str = "unet"
    denoiser_act_fn: str = "relu"
    num_channels: int = 1  # 3 for color images, 1 for grayscale images
    num_kernels: int = 32
    kernel_size: int = 3
    padding: int = 1
    bias: bool = False
    norm: str = "gn"  # "bn" for batch norm, "gn" for group norm
    time_embedding_method: str = "as_input"  # as_input, as_channel
    time_channels: int = 64
    num_blocks: int = 3  # number of downsampling/upsampling blocks in UNet
    num_enc_conv: int = 2  # number of conv layers in each downsampling block
    num_mid_conv: int = 2  # number of conv layers in middle block
    num_dec_conv: int = 2  # number of conv layers in each upsampling
    pool_window: int = 2


@dataclass
class DDPMConfig:
    """sami loss"""

    weighted_mse: bool = False
    reduction: str = "sum"  # "mean" or "sum"


@dataclass
class TrainingConfig:
    """training"""

    num_epochs: int = 2000
    train_batch_size_per_gpu: int = 512
    lr_schedule: str = "cosine"
    lr_init: float = 4e-3
    lr_final: float = 2e-3
    lr_num_warmup_epochs: int = 10000  # this is actually the number of steps
    precision: str = "bf16-mixed"
    seed: int = 43


@dataclass
class PretrainingConfig:
    """pretraining"""

    resume_from_checkpoint: bool = False  # master switch for using checkpoint
    pretrained_project_name: str = "ddpm_dsprites"
    pretrained_model_num: int = 2
    pretrained_artifact_id: str = "v29"

    precision: str = "32"
    project_dir: str = project_dir
    checkpoint_epoch: int = 0
    model_checkpoint_dir: str = f"{project_dir}/c_training/lightning_checkpoints"


@dataclass
class ClusterConfig:
    """cluster and logging"""

    # cluster
    strategy: str = "ddp"  # "ddp" or "deepspeed_stage_2"
    num_nodes: int = 3
    num_gpus_per_node: int = 4

    # logging
    log_every_n_steps: int = 10
    checkpoint_every_n_epochs: int = 500


@dataclass
class DDPM_SOS_Training_Config(
    SOSDatasetConfig,
    DenoiserConfig,
    DDPMConfig,
    TrainingConfig,
    PretrainingConfig,
    ClusterConfig,
):
    """model"""

    model_name: str = "ddpm_sos"

    # Overrides for DenoiserConfig
    num_timesteps: int = 100
    num_kernels: int = 32
    time_channels: int = 28  # Must match image_dim for UNet time embedding concatenation
    num_blocks: int = 2  # number of downsampling/upsampling blocks in UNet
    num_enc_conv: int = 2  # number of conv layers in each downsampling block
    num_mid_conv: int = 2  # number of conv layers in middle block
    num_dec_conv: int = 2  # number of conv layers in each upsampling
    pool_window: int = 2
    parameterization: str = "velocity"
    noise_schedule: str = "cosine_in_alpha_bar"

    # Overrides for DDPM Config
    weighted_mse: bool = True

    # Overrides for TrainingConfig
    num_epochs: int = 12000
    train_batch_size_per_gpu: int = 2000

    lr_init: float = 8e-3
    lr_final: float = 3e-4
    lr_num_warmup_epochs: int = 15000
    seed: int = 42
    precision: str = "32"

    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})

@dataclass
class DDPM_Simple_Disk_Training_Config(
    DiskDatasetConfig,
    DenoiserConfig,
    DDPMConfig,
    TrainingConfig,
    PretrainingConfig,
    ClusterConfig,
):
    """model"""

    model_name: str = "ddpm_simple_disk"

    # Overrides for DenoiserConfig
    num_timesteps: int = 100
    num_kernels: int = 32
    time_channels: int = 28  # Must match image_dim for UNet time embedding concatenation
    num_blocks: int = 2  # number of downsampling/upsampling blocks in UNet
    num_enc_conv: int = 2  # number of conv layers in each downsampling block
    num_mid_conv: int = 2  # number of conv layers in middle block
    num_dec_conv: int = 2  # number of conv layers in each upsampling
    pool_window: int = 2
    parameterization: str = "noise"  # velocity1, noise, velocity2
    noise_schedule: str = "cosine_in_alpha_bar"

    # Overrides for DDPM Config
    weighted_mse: bool = False

    # Overrides for TrainingConfig
    num_epochs: int = 12000
    train_batch_size_per_gpu: int = 2000

    lr_init: float = 8e-3
    lr_final: float = 3e-4
    lr_num_warmup_epochs: int = 30000
    seed: int = 42
    precision: str = "32"

    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})


@dataclass
class DDPM_CIS_Training_Config(
    CircleInSquareDatasetConfig,
    DenoiserConfig,
    DDPMConfig,
    TrainingConfig,
    PretrainingConfig,
    ClusterConfig,
):
    """model"""

    model_name: str = "ddpm_cis"

    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})


@dataclass
class DDPM_CelebA_64_Training_Config(
    CelebA_64_Config,
    DenoiserConfig,
    DDPMConfig,
    TrainingConfig,
    PretrainingConfig,
    ClusterConfig,
):
    """model"""

    model_name: str = "ddpm_celeba_color_64"

    """Denoiser parameters"""
    timestep_dist: str = "uniform"
    num_timesteps: int = 1000
    beta_minmax: tuple = (1e-4, 2e-2)
    sigma_minmax: tuple = (0.0001, 0.9999)
    noise_schedule: str = "cosine_in_alpha_bar"  # 'linear_in_beta', 'cosine_in_alpha_bar'
    denoiser_type: str = "unet"
    # denoiser_block_out_channels: list[int] = field(default_factory=lambda: [128, 128, 256, 256, 512, 512])

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

    """diva"""
    weighted_MSE: bool = False
    reduction: str = "mean"  # "mean" or "sum"

    """training"""
    num_epochs = 50
    train_batch_size_per_gpu: int = 512
    lr_schedule: str = "cosine"
    lr_init: float = 1e-3
    lr_final: float = 1e-4
    lr_num_warmup_epochs: int = 50
    seed = 43
    train_infnet_only: bool = False
    train_denoiser_from_scratch: bool = True
    precision: str = "32"
    data_cache_dir: str = "/mnt/home/blyo1/ceph/projects/diva/datasets/celeba_color/"  # path to cache the dataset

    """cluster"""
    strategy: str = "ddp"  # "ddp" or "deepspeed_stage_2"
    # zero_opt_stage: int = 2
    num_gpus_per_node: int = 4
    num_nodes: int = 2

    """logging"""
    log_every_n_steps: int = 10
    checkpoint_every_n_epochs: int = 10

    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})


@dataclass
class SelectConfig(DDPM_Simple_Disk_Training_Config):
    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})
