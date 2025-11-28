from dataclasses import dataclass, fields, field

@dataclass
class DDPM_CelebA_64_Training_Config:
    '''model'''
    model_name: str = "ddpm_celeba_color_64"

    '''dataset parameters'''
    image_dim: int = 64
    dataset_name: str = "celeba_color"
    dataset_size: int = 162770
    
    '''Denoiser parameters'''
    timestep_dist: str = 'uniform'
    num_timesteps: int = 1000
    beta_minmax: tuple = (1e-4, 2e-2)
    denoiser_target: str = "noise"  # "image", "residual", "noise"
    sigma_minmax: tuple = (0.0001, .9999)
    noise_schedule: str = 'cosine_in_alpha_bar'  # 'linear_in_beta', 'cosine_in_alpha_bar'
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
    
    '''diva'''
    weighted_MSE: bool = False
    reduction: str = "mean"  # "mean" or "sum"

    '''training'''
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

    '''cluster'''
    strategy: str = "ddp"  # "ddp" or "deepspeed_stage_2"
    # zero_opt_stage: int = 2
    num_gpus_per_node: int = 4
    num_nodes: int = 2

    '''logging'''
    log_every_n_steps: int = 10
    checkpoint_every_n_epochs: int = 10

    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})


@dataclass
class DDPM_dSprites_Training_Config:
    '''model'''
    model_name: str = "ddpm_dsprites"

    '''dataset parameters'''
    image_dim: int = 64
    dataset_name: str = "dsprites"
    dataset_size: int = 737280
    
    '''Denoiser parameters'''
    timestep_dist: str = 'uniform'
    num_timesteps: int = 1000
    beta_minmax: tuple = (1e-4, 2e-2)
    denoiser_target: str = "noise"  # "image", "residual", "noise"
    sigma_minmax: tuple = (0.0001, .9999)
    noise_schedule: str = 'cosine_in_alpha_bar'  # 'linear_in_beta', 'cosine_in_alpha_bar'
    denoiser_type: str = "unet"
    # denoiser_block_out_channels: list[int] = field(default_factory=lambda: [128, 128, 256, 256, 512, 512])

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
    
    '''diva'''
    weighted_MSE: bool = False
    reduction: str = "mean"  # "mean" or "sum"

    '''training'''
    num_epochs = 300
    train_batch_size_per_gpu: int = 512
    lr_schedule: str = "cosine"
    lr_init: float = 3e-3
    lr_final: float = 2e-3
    lr_num_warmup_epochs: int = 20000
    seed = 43
    precision: str = "32"
    model_checkpoint_dir: str = "/mnt/home/blyo1/hdiva/c_training/lightning_checkpoints"
    data_cache_dir: str = "/mnt/home/blyo1/ceph/projects/hdiva/a_datasets/dsprites/"  # path to cache the dataset
    resume_from_checkpoint: bool = True
    checkpoint_model_num: int = 1
    checkpoint_epoch: int|None = None

    '''cluster'''
    strategy: str = "ddp"  # "ddp" or "deepspeed_stage_2"
    # zero_opt_stage: int = 2
    num_nodes: int = 5
    num_gpus_per_node: int = 4

    '''logging'''
    log_every_n_steps: int = 10
    checkpoint_every_n_epochs: int = 10

    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})