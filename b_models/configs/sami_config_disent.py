from dataclasses import dataclass, fields
from typing import Literal, Optional, Tuple

project_dir = "/mnt/home/blyo1/hdiva"


@dataclass
class Dsprites_64_Config:
    """dsprites dataset parameters"""

    image_dim: int = 64
    dataset_size: int = 737280
    data_cache_dir: str = f"{project_dir}/a_datasets/dsprites/"

@dataclass
class DenoiserConfig:
    """Denoiser parameters"""
    denoiser_type: str = "unet"
    denoiser_act_fn: str = "relu"
    num_channels: int = 1  # 3 for color images, 1 for grayscale images
    num_kernels: int = 32
    kernel_size: int = 3
    padding: int = 1
    bias: bool = False
    norm: Literal["gn", "bn"] = "gn"  # "bn" for batch norm, "gn" for group norm
    time_embedding_method: Literal["as_input", "as_channel"] = "as_input"
    time_channels: int = 64
    num_blocks: int = 3  # number of downsampling/upsampling blocks in UNet
    num_enc_conv: int = 2  # number of conv layers in each downsampling block
    num_mid_conv: int = 2  # number of conv layers in middle block
    num_dec_conv: int = 2  # number of conv layers in each upsampling
    pool_window: int = 2

@dataclass
class DDPMConfig(DenoiserConfig):
    parameterization: Literal["velocity", "noise", "image"] = "noise"
    timestep_dist: Literal["uniform", "linearly_increasing", "exponentially_increasing", "hump", "linear_then_uniform"] = "linear_then_uniform"
    num_timesteps: int = 200
    noise_schedule: Literal["cosine_in_alpha_bar", "linear_in_alpha_bar"] = "cosine_in_alpha_bar" 
    beta_minmax: tuple = (1e-4, 2e-2)
    sigma_minmax: tuple = (0.0001, 0.9999)
    weighted_mse: bool = False
    reduction: Literal["mean", "sum"] = "mean"


@dataclass
class UNetInfNetConfig:
    """infnet (UNet-like)"""
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
    latent_dim: int = 8
    num_layers_rec: int = 3
    num_kernels_rec: int = 32
    kernel_size_rec: int = 3
    stride_rec: int = 2
    padding_rec: int = 1
    activation_rec: Literal["gelu", "relu"] = "relu" 
    norm_rec: Literal["bn", "gn"] = "gn"  # "bn" for batch norm, "gn" for group norm
    num_groups_rec: int = 8
    bias_rec: bool = True
    adaptive_avg_pool_output_size: int = 3
    time_embedding_method_rec: Literal["none", "as_input", "per_layer", "film"] = "film"
    logvar_init_bias: float = -2.0


@dataclass
class ResNetInfNetConfig(ConvNetInfNetConfig):
    """infnet (ResNet): strided downsampling trunk + stride-1 residual blocks.

    Depth comes from `num_res_blocks_rec` rather than from more downsampling stages, so
    the spatial map stays large and the guidance score stays small while the receptive
    field still grows. With `res_zero_init=True` each residual branch starts at alpha=0,
    so the network begins as its downsampling trunk and `logvar_init_bias` does not need
    retuning when depth changes.
    """
    # latent_dim: int = 8
    # num_layers_rec: int = 3  # stride-2 downsampling stages (64 -> 32 -> 16 -> 8)
    # num_kernels_rec: int = 32
    # kernel_size_rec: int = 3
    # stride_rec: int = 2
    # padding_rec: int = 1
    # activation_rec: Literal["gelu", "relu"] = "gelu"  
    # norm_rec: Literal["bn", "gn"] = "gn"  # "bn" for batch norm, "gn" for group norm
    # num_groups_rec: int = 8
    # bias_rec: bool = True
    # adaptive_avg_pool_output_size: int = 3
    # time_embedding_method_rec: Literal["none", "as_input", "per_layer", "film"] = "film"
    # logvar_init_bias: float = -4.0
    
    res_zero_init: bool = True  # start alpha at 0: identity at init, depth grows in
    num_res_blocks_rec: int = 2  # stride-1 residual blocks at the final resolution


@dataclass
class SAMIConfig:
    """sami loss"""

    model_name: str = "sami_c64c"
    weighted_rate: bool = False

    """beta"""
    rate_type: Literal["norm", "grad", "kl"] = "kl" 
    beta_init: float = 1e-8
    beta_final: float = 1e-8
    beta_wait_epochs: int = 1000
    beta_annealing_epochs: int = 1000
    beta_annealing_schedule: Literal["linear", "cosine", "linear_in_log"] = "cosine"

@dataclass
class TrainingConfig:
    """training"""

    num_epochs: int = 1200
    train_batch_size_per_gpu: int = 500

    lr: float = 1e-3
    encoder_lr: float = 1e-3  # only gets its own param group when it differs from lr
    precision: Literal["32", "bf16-mixed"] = "32"
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

    project_dir: str = project_dir
    model_checkpoint_dir: str = f"{project_dir}/c_training/lightning_checkpoints"


@dataclass
class ClusterConfig:
    """cluster and logging"""

    # cluster
    strategy: str = "ddp"  # "ddp" or "deepspeed_stage_2"
    num_nodes: int = 1
    num_gpus_per_node: int = 4

    # logging
    log_every_n_steps: int = 10
    checkpoint_every_n_epochs: int = 100

@dataclass
class DSprites_Training_Config(
    Dsprites_64_Config,
    DDPMConfig,
    SAMIConfig,
    # ConvNetInfNetConfig,
    ResNetInfNetConfig,
    TrainingConfig,
    PretrainingConfig,
    ClusterConfig,
):

    # dataset
    dataset_name: str = "dsprites"
    dataset_config: Dsprites_64_Config = Dsprites_64_Config()

    # infnet
    infnet_type: Literal["half_unet", "convnet", "resnet"] = "convnet" 
    convnet_type: Literal["simple", "complex"] = "complex"
    # unet_config: UNetInfNetConfig = UNetInfNetConfig()
    # resnet_config: ResNetInfNetConfig = ResNetInfNetConfig()
    # convnet_config: ConvNetInfNetConfig = ConvNetInfNetConfig()

    # IMAE config
    model_name: str = "imae_dsprites"

    # pretraining config
    resume_from_checkpoint: bool = False  # master switch for using checkpoint
    checkpoint_epoch: int | str = "last"

    use_pretrained_denoiser_only: bool = True  # use a pretrained ddpm or denoiser to initialize the denoiser, rather than use the entire pretrained diva model
    train_infnet_only: bool = False  # if true, freeze the denoiser weights and only train the infnet
    pretrained_model_name: str = "ddpm_dsprites"
    pretrained_model_number: int = 6
    pretrained_artifact_id: str = "v4"
    
    # # pretraining config
    # resume_from_checkpoint: bool = True  # master switch for using checkpoint
    # checkpoint_epoch: int | str = "last"

    # use_pretrained_denoiser_only: bool = False  # use a pretrained ddpm or denoiser to initialize the denoiser, rather than use the entire pretrained diva model
    # train_infnet_only: bool = False  # if true, freeze the denoiser weights and only train the infnet
    # pretrained_model_name: str = "imae_dsprites"
    # pretrained_model_number: int = 7
    # pretrained_artifact_id: str = "v3"

    project_dir: str = project_dir
    model_checkpoint_dir: str = f"{project_dir}/c_training/lightning_checkpoints"

    # cluster
    strategy: str = "ddp"  # "ddp" or "deepspeed_stage_2"
    num_nodes: int = 1
    num_gpus_per_node: int = 3

    # logging
    log_every_n_steps: int = 10
    checkpoint_every_n_epochs: int = 100


    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})

@dataclass
class Config(
    DSprites_Training_Config,
):
    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})
