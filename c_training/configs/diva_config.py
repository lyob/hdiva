from dataclasses import dataclass, fields


@dataclass
class DiVA_Manual_CelebA_64_Training_Config:
    """model"""

    model_name: str = "diva_manual_celeba_color_64"

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
    noise_schedule: str = (
        "cosine_in_alpha_bar"  # 'linear_in_beta', 'cosine_in_alpha_bar'
    )
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

    resume_from_checkpoint: bool = False  # master switch for using checkpoint
    use_pretrained_denoiser_only: bool = False  # use a pretrained ddpm or denoiser to initialize the denoiser, rather than use the entire pretrained diva model
    train_infnet_only: bool = (
        False  # if true, freeze the denoiser weights and only train the infnet
    )

    precision: str = "32"
    model_checkpoint_dir: str = "/mnt/home/blyo1/hdiva/c_training/lightning_checkpoints"
    data_cache_dir: str = (
        "/mnt/home/blyo1/hdiva/a_datasets/celeba_color/"  # path to cache the dataset
    )

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


@dataclass
class DiVA_Manual_dSprites_Training_Config:
    """model"""

    model_name: str = "diva_dsprites"

    """dataset parameters"""
    image_dim: int = 64
    dataset_name: str = "dsprites"
    dataset_size: int = 737280

    """Denoiser parameters"""
    timestep_dist: str = "uniform"
    num_timesteps: int = 1000
    beta_minmax: tuple = (1e-4, 2e-2)
    denoiser_target: str = "noise"  # "image", "residual", "noise"
    sigma_minmax: tuple = (0.0001, 0.9999)
    noise_schedule: str = (
        "cosine_in_alpha_bar"  # 'linear_in_beta', 'cosine_in_alpha_bar'
    )
    denoiser_type: str = "unet"

    denoiser_act_fn: str = "relu"
    num_channels: int = 1  # 3 for color images, 1 for grayscale images
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

    """diva"""
    weighted_MSE: bool = False
    reduction: str = "mean"  # "mean" or "sum"

    """kl"""
    kl_weight_min: float = 1e-13
    kl_weight_max: float = 1e-13
    kl_annealing_epochs: int = 200
    kl_annealing_schedule: str = "cosine"  # "linear" or "cosine" or "linear_in_log"

    """training"""
    num_epochs: int = 300
    train_batch_size_per_gpu: int = 512
    lr_schedule: str = "cosine"
    lr_init: float = 4e-3
    lr_final: float = 1e-3
    lr_num_warmup_epochs: int = 10000  # this is actually the number of steps
    encoder_lr_init: float = 1e-4
    encoder_lr_final: float = 4e-4
    encoder_warmup_epochs: int = 8000
    encoder_convergence_epochs: int = 5000
    seed: int = 43

    resume_from_checkpoint: bool = False  # master switch for using checkpoint
    use_pretrained_denoiser_only: bool = False  # use a pretrained ddpm or denoiser to initialize the denoiser, rather than use the entire pretrained diva model
    train_infnet_only: bool = (
        False  # if true, freeze the denoiser weights and only train the infnet
    )

    precision: str = "32"
    model_checkpoint_dir: str = "/mnt/home/blyo1/hdiva/c_training/lightning_checkpoints"
    data_cache_dir: str = (
        "/mnt/home/blyo1/hdiva/a_datasets/dsprites/"  # path to cache the dataset
    )

    """cluster"""
    strategy: str = "ddp"  # "ddp" or "deepspeed_stage_2"
    # strategy: str = "deepspeed_stage_2"  # "ddp" or "deepspeed_stage_2"
    # zero_opt_stage: int = 2
    num_nodes: int = 5
    num_gpus_per_node: int = 4

    """logging"""
    log_every_n_steps: int = 10
    checkpoint_every_n_epochs: int = 10

    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})


@dataclass
class DiVA_ConvNet_dSprites_Training_Config:
    """model"""

    model_name: str = "diva_convnet_dsprites"

    """dataset parameters"""
    image_dim: int = 64
    dataset_name: str = "dsprites"
    dataset_size: int = 737280

    """Denoiser parameters"""
    timestep_dist: str = "uniform"
    num_timesteps: int = 1000
    beta_minmax: tuple = (1e-4, 2e-2)
    denoiser_target: str = "noise"  # "image", "residual", "noise"
    sigma_minmax: tuple = (0.0001, 0.9999)
    noise_schedule: str = (
        "cosine_in_alpha_bar"  # 'linear_in_beta', 'cosine_in_alpha_bar'
    )
    denoiser_type: str = "unet"

    denoiser_act_fn: str = "relu"
    num_channels: int = 1  # 3 for color images, 1 for grayscale images
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
    # encoder model
    latent_dim: int = 10
    num_layers_rec: int = 2
    num_kernels_rec: int = 32
    kernel_size_rec: int = 4
    stride_rec: int = 2
    padding_rec: int = 1
    activation_rec: str = "relu"  # gelu or relu
    bias_rec: bool = False

    """diva"""
    weighted_MSE: bool = False
    reduction: str = "mean"  # "mean" or "sum"

    """kl"""
    kl_weight_min: float = 1e-14
    kl_weight_max: float = 1e-14
    kl_annealing_epochs: int = 200
    kl_annealing_schedule: str = "cosine"  # "linear" or "cosine" or "linear_in_log"

    """training"""
    num_epochs: int = 300
    train_batch_size_per_gpu: int = 512
    lr_schedule: str = "cosine"
    lr_init: float = 1e-3
    lr_final: float = 1e-3
    lr_num_warmup_epochs: int = 1  # this is actually the number of steps

    encoder_lr_init: float = 4e-3
    encoder_lr_final: float = 1e-3
    encoder_warmup_epochs: int = 0
    encoder_convergence_epochs: int = 20000
    seed: int = 43

    resume_from_checkpoint: bool = True  # master switch for using checkpoint
    use_pretrained_denoiser_only: bool = True  # use a pretrained ddpm or denoiser to initialize the denoiser, rather than use the entire pretrained diva model
    train_infnet_only: bool = (
        False  # if true, freeze the denoiser weights and only train the infnet
    )
    pretrained_project_name: str = "diva_convnet_dsprites"
    pretrained_model_num: int = 2
    pretrained_artifact_id: str = "v29"

    precision: str = "32"
    model_checkpoint_dir: str = "/mnt/home/blyo1/hdiva/c_training/lightning_checkpoints"
    data_cache_dir: str = (
        "/mnt/home/blyo1/hdiva/a_datasets/dsprites/"  # path to cache the dataset
    )

    """cluster"""
    strategy: str = "ddp"  # "ddp" or "deepspeed_stage_2"
    # strategy: str = "deepspeed_stage_2"  # "ddp" or "deepspeed_stage_2"
    # zero_opt_stage: int = 2
    num_nodes: int = 5
    num_gpus_per_node: int = 4

    """logging"""
    log_every_n_steps: int = 10
    checkpoint_every_n_epochs: int = 10

    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})
