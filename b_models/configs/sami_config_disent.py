from dataclasses import dataclass, fields
from typing import Optional, Tuple

project_dir = "/mnt/home/blyo1/hdiva"


@dataclass
class DspritesDatasetConfig:
    """dsprites dataset parameters"""

    image_dim: int = 64
    dataset_name: str = "dsprites"
    dataset_size: int = 737280
    data_cache_dir: str = f"{project_dir}/a_datasets/dsprites/"

@dataclass
class DenoiserConfig:
    """Denoiser parameters"""
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
class DDPMConfig():
    timestep_dist: str = "uniform"  # uniform, linearly_increasing
    num_timesteps: int = 1000
    beta_minmax: tuple = (1e-4, 2e-2)
    sigma_minmax: tuple = (0.0001, 0.9999)
    noise_schedule: str = "cosine_in_alpha_bar"  # cosine_in_alpha_bar or linear_in_alpha_bar
    denoiser_type: str = "unet"
    parameterization: str = "velocity"  # velocity or noise or image


@dataclass
class UNetInfNetConfig:
    """infnet (UNet-like)"""

    infnet_type: str = "half_unet"
    latent_dim: int = 10
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


@dataclass
class ConvNetInfNetConfig:
    """infnet (ConvNet)"""

    # encoder model
    infnet_type: str = "convnet"
    convnet_type: str = "complex"
    latent_dim: int = 4
    num_layers_rec: int = 5
    num_kernels_rec: int = 32
    kernel_size_rec: int = 3
    stride_rec: int = 2
    padding_rec: int = 1
    activation_rec: str = "gelu"  # gelu or relu
    norm_rec: str = "bn"  # "bn" for batch norm, "gn" for group norm
    bias_rec: bool = True
    adaptive_avg_pool_output_size: int = 1
    time_embedding_method_rec: str = "as_input"  # none, as_input, per_layer, film


@dataclass
class SAMIConfig:
    """sami loss"""

    model_name: str = "sami_c64c"
    reduction: str = "sum"  # "mean" or "sum"
    weighted_mse: bool = False
    weighted_rate: bool = False

    """beta"""
    rate_type: str = "grad"  # "grad", "norm", "cumulative", "kl"
    beta_init: float = 1e-5
    beta_final: float = 1e-5
    beta_wait_epochs: int = 1000
    beta_annealing_epochs: int = 1000
    beta_annealing_schedule: str = "cosine" # "linear" or "cosine" or "linear_in_log"

@dataclass
class TrainingConfig:
    """training"""

    num_epochs: int = 80000
    train_batch_size_per_gpu: int = 4000

    lr_schedule: str = "cosine"
    lr_init: float = 1e-3
    lr_final: float = 1e-4
    lr_num_warmup_epochs: int = 10000  # this is actually the number of steps
    encoder_lr_init: float = 1e-3
    encoder_lr_intermed: float = 3e-5
    encoder_lr_final: float = 3e-5
    encoder_wait_epochs: int = 0
    encoder_warmup_epochs: int = 50000
    encoder_convergence_epochs: int = 0
    precision: str = "32"  # "32" or "bf16-mixed"
    seed: int = 43
    optimizer: str = "adam"


@dataclass
class PretrainingConfig:
    """pretraining"""

    resume_from_checkpoint: bool = True  # master switch for using checkpoint
    checkpoint_epoch: int | str = "last"

    use_pretrained_denoiser_only: bool = True  # use a pretrained ddpm or denoiser to initialize the denoiser, rather than use the entire pretrained diva model
    train_infnet_only: bool = False  # if true, freeze the denoiser weights and only train the infnet
    pretrained_model_name: str = "ddpm_simple_disk"
    pretrained_model_number: int = 6
    pretrained_artifact_id: str = "v23"

    precision: str = "32"
    project_dir: str = project_dir
    model_checkpoint_dir: str = f"{project_dir}/c_training/lightning_checkpoints"


@dataclass
class ClusterConfig:
    """cluster and logging"""

    # cluster
    strategy: str = "ddp"  # "ddp" or "deepspeed_stage_2"
    num_nodes: int = 1
    num_gpus_per_node: int = 2

    # logging
    log_every_n_steps: int = 10
    checkpoint_every_n_epochs: int = 2000


@dataclass
class SAMI_ConvNet_dSprites_Training_Config(
    DspritesDatasetConfig,
    DenoiserConfig,
    ConvNetInfNetConfig,
    SAMIConfig,
    TrainingConfig,
    PretrainingConfig,
    ClusterConfig,
):
    """model"""

    model_name: str = "sami_convnet_dsprites"

    # Overrides for DenoiserConfig
    timestep_dist: str = "uniform"
    num_kernels: int = 64

    # Overrides for ConvNetInfNetConfig
    convnet_type: str = "complex"
    num_layers_rec: int = 3
    num_kernels_rec: int = 32

    # Overrides for SAMIConfig
    kl_weight_min: float = 1e-11
    kl_weight_max: float = 1e-10
    kl_annealing_epochs: int = 20000

    # Overrides for TrainingConfig
    num_epochs: int = 700
    train_batch_size_per_gpu: int = 1000
    lr_init: float = 5e-3
    lr_final: float = 3e-3
    lr_num_warmup_epochs: int = 27000
    encoder_lr_init: float = 2e-3
    encoder_lr_intermed: float = 5e-3
    encoder_lr_final: float = 4e-3
    encoder_wait_epochs: int = 6000
    encoder_warmup_epochs: int = 10000
    encoder_convergence_epochs: int = 20000
    seed: int = 42

    # Overrides for PretrainingConfig
    resume_from_checkpoint: bool = False
    use_pretrained_denoiser_only: bool = False
    checkpoint_epoch: str | None = None

    # Overrides for ClusterConfig
    num_nodes: int = 1

    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})

@dataclass
class Config(
    SAMI_ConvNet_dSprites_Training_Config,
):
    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})
