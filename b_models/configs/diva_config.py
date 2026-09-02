from dataclasses import dataclass, fields

project_dir = "/mnt/home/blyo1/hdiva"


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


@dataclass
class DiVA_Manual_dSprites_Training_Config:
    """model"""

    model_name: str = "diva_dsprites"

    """dataset parameters"""
    image_dim: int = 64
    dataset_name: str = "dsprites"
    dataset_size: int = 737280

    """Denoiser parameters"""
    timestep_dist: str = "linearly_increasing"
    num_timesteps: int = 1000
    beta_minmax: tuple = (1e-4, 2e-2)
    denoiser_target: str = "noise"  # "image", "residual", "noise"
    sigma_minmax: tuple = (0.0001, 0.9999)
    noise_schedule: str = "cosine_in_alpha_bar"  # 'linear_in_beta', 'cosine_in_alpha_bar'
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
    kl_weight_min: float = 1e-16
    kl_weight_max: float = 1e-14
    kl_annealing_epochs: int = 300
    kl_annealing_schedule: str = "cosine"  # "linear" or "cosine" or "linear_in_log"

    """training"""
    num_epochs: int = 400
    train_batch_size_per_gpu: int = 512
    lr_schedule: str = "cosine"
    lr_init: float = 4e-3
    lr_final: float = 2e-3
    lr_num_warmup_epochs: int = 10000  # this is actually the number of steps
    encoder_lr_init: float = 2e-4
    encoder_lr_intermed: float = 2e-4
    encoder_lr_final: float = 3e-4
    encoder_wait_epochs: int = 0
    encoder_warmup_epochs: int = 8000
    encoder_convergence_epochs: int = 5000
    seed: int = 43

    """pretraining"""
    resume_from_checkpoint: bool = True  # master switch for using checkpoint
    use_pretrained_denoiser_only: bool = True  # use a pretrained ddpm or denoiser to initialize the denoiser, rather than use the entire pretrained diva model
    train_infnet_only: bool = False  # if true, freeze the denoiser weights and only train the infnet
    pretrained_project_name: str = "ddpm_dsprites"
    pretrained_model_num: int = 2
    pretrained_artifact_id: str = "v29"

    precision: str = "32"
    project_dir: str = project_dir
    model_checkpoint_dir: str = f"{project_dir}/c_training/lightning_checkpoints"
    data_cache_dir: str = f"{project_dir}/a_datasets/dsprites/"

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
class DiVA_Manual_dSprites_Training_Config_Alternating_Phases:
    """model"""

    model_name: str = "diva_dsprites_alternating"

    """dataset parameters"""
    image_dim: int = 64
    dataset_name: str = "dsprites"
    dataset_size: int = 737280

    """Denoiser parameters"""
    timestep_dist: str = "linearly_increasing"
    num_timesteps: int = 1000
    beta_minmax: tuple = (1e-4, 2e-2)
    denoiser_target: str = "noise"  # "image", "residual", "noise"
    sigma_minmax: tuple = (0.0001, 0.9999)
    noise_schedule: str = "cosine_in_alpha_bar"  # 'linear_in_beta', 'cosine_in_alpha_bar'
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
    latent_dim: int = 7
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
    kl_weight_min: float = 1e-16
    kl_weight_max: float = 1e-16
    kl_annealing_epochs: int = 300
    kl_annealing_schedule: str = "cosine"  # "linear" or "cosine" or "linear_in_log"

    """training"""
    num_epochs: int = 500
    train_batch_size_per_gpu: int = 512
    lr_schedule: str = "cosine"
    seed: int = 43

    """alternating phases"""
    phase_a_duration_epochs: int = 500
    phase_b_duration_epochs: int = 100
    starting_phase: str = "a"  # "a" or "b"

    lr_init: float = 1e-3

    # Phase A (Encoder High, Denoiser Low)
    phase_a_encoder_lr: float = 4e-3
    phase_a_denoiser_lr: float = 1e-4
    phase_a_timestep_dist: str = "linearly_increasing"

    # Phase B (Encoder Low, Denoiser High)
    phase_b_encoder_lr: float = 1e-4
    phase_b_denoiser_lr: float = 3e-3
    phase_b_timestep_dist: str = "uniform"

    """pretraining"""
    resume_from_checkpoint: bool = True  # master switch for using checkpoint
    use_pretrained_denoiser_only: bool = True  # use a pretrained ddpm or denoiser to initialize the denoiser, rather than use the entire pretrained diva model
    train_infnet_only: bool = False  # if true, freeze the denoiser weights and only train the infnet
    pretrained_project_name: str = "ddpm_dsprites"
    pretrained_model_num: int = 2
    pretrained_artifact_id: str = "v29"

    precision: str = "32"
    project_dir: str = project_dir
    model_checkpoint_dir: str = f"{project_dir}/c_training/lightning_checkpoints"
    data_cache_dir: str = f"{project_dir}/a_datasets/dsprites/"

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
    noise_schedule: str = "cosine_in_alpha_bar"  # 'linear_in_beta', 'cosine_in_alpha_bar'
    denoiser_type: str = "unet"

    denoiser_act_fn: str = "relu"
    num_channels: int = 1  # 3 for color images, 1 for grayscale images
    num_kernels: int = 64
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
    convnet_type: str = "complex"
    latent_dim: int = 7
    num_layers_rec: int = 3
    num_kernels_rec: int = 32
    kernel_size_rec: int = 3
    stride_rec: int = 2
    padding_rec: int = 1
    activation_rec: str = "relu"  # gelu or relu
    bias_rec: bool = False

    """diva"""
    weighted_MSE: bool = False
    reduction: str = "mean"  # "mean" or "sum"

    """kl"""
    kl_weight_min: float = 1e-11
    kl_weight_max: float = 1e-10
    kl_annealing_epochs: int = 20000
    kl_annealing_schedule: str = "cosine"  # "linear" or "cosine" or "linear_in_log"

    """training"""
    num_epochs: int = 500
    train_batch_size_per_gpu: int = 800
    lr_schedule: str = "cosine"

    lr_init: float = 5e-3
    lr_final: float = 1e-3
    lr_num_warmup_epochs: int = 27000  # this is actually the number of steps

    encoder_lr_init: float = 1e-3
    encoder_lr_intermed: float = 5e-3
    encoder_lr_final: float = 1e-3
    encoder_wait_epochs: int = 10000
    encoder_warmup_epochs: int = 500
    encoder_convergence_epochs: int = 20000
    seed: int = 42

    """pretraining"""
    resume_from_checkpoint: bool = False  # master switch for using checkpoint
    use_pretrained_denoiser_only: bool = False  # use a pretrained ddpm or denoiser to initialize the denoiser, rather than use the entire pretrained diva model
    train_infnet_only: bool = False  # if true, freeze the denoiser weights and only train the infnet

    """pretrained model"""
    pretrained_project_name: str = "ddpm_dsprites"
    pretrained_model_num: int = 2
    pretrained_artifact_id: str = "v29"
    checkpoint_epoch: str | None = None

    precision: str = "32"
    project_dir: str = project_dir
    model_checkpoint_dir: str = f"{project_dir}/c_training/lightning_checkpoints"
    data_cache_dir: str = f"{project_dir}/a_datasets/dsprites"

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
class DiVA_ConvNet_dSprites_Training_Config_Alternating_Phases:
    """model"""

    model_name: str = "diva_convnet_dsprites_alternating"

    """dataset parameters"""
    image_dim: int = 64
    dataset_name: str = "dsprites"
    dataset_size: int = 737280

    """Denoiser parameters"""
    timestep_dist: str = "linearly_increasing"
    num_timesteps: int = 1000
    beta_minmax: tuple = (1e-4, 2e-2)
    denoiser_target: str = "noise"  # "image", "residual", "noise"
    sigma_minmax: tuple = (0.0001, 0.9999)
    noise_schedule: str = "cosine_in_alpha_bar"  # 'linear_in_beta', 'cosine_in_alpha_bar'
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
    num_kernels_rec: int = 48
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
    num_epochs: int = 500
    train_batch_size_per_gpu: int = 512
    lr_schedule: str = "cosine"
    seed: int = 43

    """alternating phases"""
    phase_a_duration_epochs: int = 400
    phase_b_duration_epochs: int = 200
    starting_phase: str = "a"  # "a" or "b"

    lr_init: float = 1e-3

    # Phase A (Encoder High, Denoiser Low)
    phase_a_encoder_lr: float = 2e-3
    phase_a_denoiser_lr: float = 1e-5
    phase_a_timestep_dist: str = "linearly_increasing"

    # Phase B (Encoder Low, Denoiser High)
    phase_b_encoder_lr: float = 1e-5
    phase_b_denoiser_lr: float = 2e-3
    phase_b_timestep_dist: str = "uniform"

    """pretraining"""
    resume_from_checkpoint: bool = True  # master switch for using checkpoint
    use_pretrained_denoiser_only: bool = True  # use a pretrained ddpm or denoiser to initialize the denoiser, rather than use the entire pretrained diva model
    train_infnet_only: bool = False  # if true, freeze the denoiser weights and only train the infnet

    """pretrained model"""
    pretrained_project_name: str = "ddpm_dsprites"
    pretrained_model_num: int = 2
    pretrained_artifact_id: str = "v29"

    precision: str = "32"
    project_dir: str = project_dir
    model_checkpoint_dir: str = f"{project_dir}/c_training/lightning_checkpoints"
    data_cache_dir: str = f"{project_dir}/a_datasets/dsprites"

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
