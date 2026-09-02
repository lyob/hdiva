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
class BumpDatasetConfig:
    """bump dataset parameters"""

    image_dim: int = 28
    dataset_name: str = "bump"
    dataset_size: int = 100000
    holdout_center: bool = False
    data_cache_dir: str = f"{project_dir}/a_datasets/bump/"

@dataclass
class DiskDatasetConfig:
    """disk dataset parameters"""

    image_dim: int = 28
    dataset_name: str = "simple_disks"
    dataset_size: int = 100000  # 1e5
    data_cache_dir: str = f"{project_dir}/a_datasets/simple_disks/"
    foreground: float = 1.
    background: float = 0.
    outer_radius: int = 8
    transition_width: int = 2
    cx_range: Optional[Tuple[float, float]] = (10, 18)
    cy_range: Optional[Tuple[float, float]] = None
    dataset_seed: int = 42

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
class CircleInSquareDatasetConfig:
    """circle in square dataset parameters"""

    image_dim: int = 28  # 28 or 64, the rest of the parameters are scaled accordingly in the data generation code
    dataset_name: str = "circle_in_square"
    dataset_size: int = 200000
    data_cache_dir: str = f"{project_dir}/a_datasets/circle_in_square/"


@dataclass
class DenoiserConfig:
    """Denoiser parameters"""

    timestep_dist: str = "uniform"  # uniform, linearly_increasing
    num_timesteps: int = 1000
    beta_minmax: tuple = (1e-4, 2e-2)
    sigma_minmax: tuple = (0.0001, 0.9999)
    noise_schedule: str = "cosine_in_alpha_bar"  # cosine_in_alpha_bar or linear_in_alpha_bar
    denoiser_type: str = "unet"
    parameterization: str = "velocity"  # velocity or noise or image

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
    bias_rec: bool = False
    adaptive_avg_pool_output_size: int = 1


@dataclass
class MLPInfNetConfig:
    """infnet (MLP)"""

    # encoder model
    infnet_type: str = "mlp"
    hidden_dim: int = 256
    num_layers_rec: int = 2
    latent_dim: int = 2
    activation_rec: str = "relu"  # gelu or relu
    bias_rec: bool = False


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
    num_nodes: int = 3
    num_gpus_per_node: int = 4

    # logging
    log_every_n_steps: int = 10
    checkpoint_every_n_epochs: int = 2000


@dataclass
class SAMI_Manual_dSprites_Training_Config(
    DspritesDatasetConfig,
    DenoiserConfig,
    UNetInfNetConfig,
    SAMIConfig,
    TrainingConfig,
    PretrainingConfig,
    ClusterConfig,
):
    """model"""

    model_name: str = "sami_dsprites"

    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})


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
class SAMI_Bump_Training_Config(
    # BumpDatasetConfig,
    SOSDatasetConfig,
    DenoiserConfig,
    ConvNetInfNetConfig,
    # MLPInfNetConfig,
    SAMIConfig,
    TrainingConfig,
    PretrainingConfig,
    ClusterConfig,
):
    """model"""

    model_name: str = "sami_bump"

    # Overrides for DenoiserConfig
    num_kernels: int = 64
    time_channels: int = 28
    num_blocks: int = 2  # number of downsampling/upsampling blocks in UNet
    num_enc_conv: int = 2  # number of conv layers in each downsampling block
    num_mid_conv: int = 2  # number of conv layers in middle block
    num_dec_conv: int = 2  # number of conv layers in each upsampling
    pool_window: int = 2

    # Overrides for ConvNetInfNetConfig# encoder model
    infnet_type: str = "convnet"
    convnet_type: str = "complex"
    latent_dim: int = 4
    num_layers_rec: int = 4
    num_kernels_rec: int = 64
    kernel_size_rec: int = 3
    stride_rec: int = 2
    padding_rec: int = 1
    activation_rec: str = "relu"  # gelu or relu
    bias_rec: bool = False

    convnet_type: str = "complex"
    latent_dim: int = 2
    num_layers_rec: int = 2
    kernel_size_rec: int = 3
    stride_rec: int = 1
    num_kernels_rec: int = 64
    adaptive_avg_pool_output_size: int = 7

    # Overrides for SAMIConfig
    use_kl: bool = False
    kl_weight_min: float = 1e-11
    kl_weight_max: float = 1e-10
    kl_annealing_epochs: int = 2000

    # Overrides for TrainingConfig
    num_epochs: int = 3000
    fraction_unconditional_init: float = 0.1
    fraction_unconditional_final: float = 0
    fraction_unconditional_annealing_epochs: int = 400
    train_batch_size_per_gpu: int = 1000
    lr_init: float = 6e-3
    lr_final: float = 3e-3
    lr_num_warmup_epochs: int = 15000
    encoder_lr_init: float = 2e-3
    encoder_lr_intermed: float = 3e-3
    encoder_lr_final: float = 3e-3
    encoder_wait_epochs: int = 4000
    encoder_warmup_epochs: int = 10000
    encoder_convergence_epochs: int = 20000
    seed: int = 42
    precision: str = "bf16-mixed"

    # Overrides for PretrainingConfig
    resume_from_checkpoint: bool = False
    use_pretrained_denoiser_only: bool = False
    checkpoint_epoch: str | None = None

    # Overrides for ClusterConfig
    strategy: str = "ddp_find_unused_parameters_true"
    num_nodes: int = 1

    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})


@dataclass
class SAMI_SOS_Training_Config(
    SOSDatasetConfig,
    DenoiserConfig,
    ConvNetInfNetConfig,
    SAMIConfig,
    TrainingConfig,
    PretrainingConfig,
    ClusterConfig,
):
    """model"""

    model_name: str = "sami_sos"

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

    # Overrides for ConvNetInfNetConfig
    infnet_type: str = "convnet"
    convnet_type: str = "complex"
    latent_dim: int = 2
    num_layers_rec: int = 3
    num_kernels_rec: int = 32
    kernel_size_rec: int = 4
    stride_rec: int = 2
    padding_rec: int = 1
    activation_rec: str = "gelu"  # gelu or relu
    norm_rec: str = "gn"  # "bn" for batch norm, "gn" for group norm
    num_groups_rec: int = 16  # only used if norm_rec is "gn"
    bias_rec: bool = True
    adaptive_avg_pool_output_size: int = 2

    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})

@dataclass
class SAMI_Simple_Disk_Training_Config(
    DiskDatasetConfig,
    DenoiserConfig,
    ConvNetInfNetConfig,
    SAMIConfig,
    TrainingConfig,
    PretrainingConfig,
    ClusterConfig,
):
    """model"""

    model_name: str = "sami_simple_disk"

    # Overrides for DenoiserConfig
    num_timesteps: int = 100
    num_kernels: int = 32
    time_channels: int = 28  # Must match image_dim for UNet time embedding concatenation
    num_blocks: int = 2  # number of downsampling/upsampling blocks in UNet
    num_enc_conv: int = 2  # number of conv layers in each downsampling block
    num_mid_conv: int = 2  # number of conv layers in middle block
    num_dec_conv: int = 2  # number of conv layers in each upsampling
    pool_window: int = 2
    parameterization: str = "velocity1"
    noise_schedule: str = "cosine_in_alpha_bar"

    # Overrides for SAMIConfig
    reduction: str = "sum"  # "mean" or "sum"
    weighted_mse: bool = False
    weighted_rate: bool = False
    rate_type: str = "grad"  # "grad", "norm", "cumulative", "kl"
    beta_init: float = 1e-3
    beta_final: float = 1e-3
    beta_wait_epochs: int = 10000
    beta_annealing_epochs: int = 20000
    beta_annealing_schedule: str = "cosine" # "linear" or "cosine" or "linear_in_log"
    

    # Overrides for ConvNetInfNetConfig
    infnet_type: str = "convnet"
    convnet_type: str = "complex"
    latent_dim: int = 2
    num_layers_rec: int = 3
    num_kernels_rec: int = 32
    kernel_size_rec: int = 4
    stride_rec: int = 2
    padding_rec: int = 1
    activation_rec: str = "gelu"  # gelu or relu
    norm_rec: str = "gn"  # "bn" for batch norm, "gn" for group norm
    num_groups_rec: int = 8  # only used if norm_rec is "gn"
    bias_rec: bool = False
    adaptive_avg_pool_output_size: int = 3

    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})


@dataclass
class SAMI_CIS_Training_Config(
    CircleInSquareDatasetConfig,
    DenoiserConfig,
    ConvNetInfNetConfig,
    SAMIConfig,
    TrainingConfig,
    PretrainingConfig,
    ClusterConfig,
):
    """model"""

    model_name: str = "sami_cis"

    # Overrides for DenoiserConfig
    num_timesteps: int = 100
    num_blocks: int = 2  # number of downsampling/upsampling blocks in UNet
    denoiser_type: str = "unet"
    parameterization: str = "noise"
    noise_schedule: str = "cosine_in_alpha_bar"

    # ConvNetInfNetConfig
    infnet_type: str = "convnet"
    convnet_type: str = "complex"
    latent_dim: int = 2
    num_layers_rec: int = 4
    num_kernels_rec: int = 32
    kernel_size_rec: int = 3
    stride_rec: int = 2
    padding_rec: int = 1
    activation_rec: str = "gelu"  # gelu or relu
    norm_rec: str = "gn"  # "bn" for batch norm, "gn" for group norm
    num_groups_rec: int = 8  # only used if norm_rec is "gn"
    bias_rec: bool = False
    adaptive_avg_pool_output_size: int = 1

    # Overrides for SAMIConfig
    use_kl: bool = True

    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})


@dataclass
class SAMI_CIS_28_Training_Config(
    CircleInSquareDatasetConfig,
    DenoiserConfig,
    ConvNetInfNetConfig,
    SAMIConfig,
    TrainingConfig,
    PretrainingConfig,
    ClusterConfig,
):
    """model"""

    model_name: str = "sami_cis"

    # Overrides for CircleInSquareDatasetConfig
    image_dim: int = 28  # 28 or 64, the rest of the parameters are scaled accordingly in the data generation code
    dataset_size: int = 200000

    # Overrides for ConvNetInfNetConfig
    infnet_type: str = "convnet"
    convnet_type: str = "complex"
    latent_dim: int = 4
    num_layers_rec: int = 4
    num_kernels_rec: int = 32
    kernel_size_rec: int = 3
    stride_rec: int = 2
    padding_rec: int = 1
    activation_rec: str = "gelu"  # gelu or relu
    norm_rec: str = "gn"  # "bn" for batch norm, "gn" for group norm
    num_groups_rec: int = 8  # only used if norm_rec is "gn"
    bias_rec: bool = False
    adaptive_avg_pool_output_size: int = 1

    # Overrides for DenoiserConfig
    num_timesteps: int = 100
    denoiser_type: str = "unet"
    parameterization: str = "noise"
    noise_schedule: str = "cosine_in_alpha_bar"

    # Overrides for SAMIConfig
    use_kl: bool = True

    # Overrides for TrainingConfig
    num_epochs: int = 12000
    train_batch_size_per_gpu: int = 2000

    use_kl: bool = True
    kl_weight_min: float = 1e-8
    kl_weight_max: float = 1
    kl_annealing_epochs: int = 5000

    lr_init: float = 1e-2
    lr_final: float = 5e-3
    lr_num_warmup_epochs: int = 15000
    encoder_lr_init: float = 6e-3
    encoder_lr_intermed: float = 3e-3
    encoder_lr_final: float = 3e-3
    encoder_wait_epochs: int = 5000
    encoder_warmup_epochs: int = 10000
    encoder_convergence_epochs: int = 20000
    seed: int = 42
    precision: str = "32"

    # Overrides for ClusterConfig
    num_nodes: int = 2
    num_gpus_per_node: int = 4

    # logging
    log_every_n_steps: int = 10
    checkpoint_every_n_epochs: int = 1000

    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})


@dataclass
class SelectConfig(
    # SAMI_Bump_Training_Config,
    # SAMI_Manual_dSprites_Training_Config,
    # SAMI_SOS_Training_Config,
    SAMI_Simple_Disk_Training_Config,
    # SAMI_CIS_Training_Config,
    # SAMI_CIS_28_Training_Config,
):
    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})
