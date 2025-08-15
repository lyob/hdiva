from dataclasses import dataclass

@dataclass
class DiVA_CelebA_64_Training_Config:
    '''general model parameters'''
    model_name: str = "diva_celeba_color_64"
    image_size: int = 64
    in_channels: int = 3
    d_latent: int = 512

    '''dataset parameters'''
    dataset_name: str = "celeba_color"
    # dataset_size: int = 100000
    dataset_size: int = 162770
    
    '''Denoiser parameters'''
    beta_schedule: str = "squaredcos_cap_v2"  # the noise distribution. "linear" or "sigmoid" or "squaredcos_cap_v2"
    # beta_schedule: str = "linear"  # the noise distribution. "linear" or "sigmoid"
    denoiser_size: str = "large"  # small or large
    timestep_dist: str = 'uniform'
    num_timesteps: int = 1000
    denoiser_type: str = "unet"
    denoiser_block_out_channels: list = (128, 128, 256, 256, 512, 512)
    # denoiser_block_out_channels: list = (256, 256, 362, 362, 512, 512)
    # denoiser_block_out_channels: list = (128, 256, 256, 512)
    denoiser_act_fn: str = "relu"
    denoiser_layers_per_block: int = 2  # how many ResNet layers to use per UNet block
    
    '''infnet'''
    # "linear" or "half_unet" or "half_unet_no_time" or "half_unet_no_time_no_attn"
    infnet_type: str = "half_unet_no_time_no_attn"
    # infnet_block_out_channels: list = (64, 64, 128, 128)
    infnet_block_out_channels: list = (128, 128, 256, 256, 256, 128)
    # infnet_block_out_channels: list = (64, 64, 64, 64)
    kl_weight_min: float = 1e-10
    kl_weight_max: float = 1e-8
    kl_annealing_epochs: int = 25
    kl_annealing_schedule: str = "linear_in_log"  # "linear" or "cosine" or "linear_in_log"
    kl_reduction: str = "mean"  # "mean" or "sum"
    infnet_act_fn: str = "gelu"  # "silu" or "relu" or "gelu"

    '''training'''
    num_epochs = 1000
    train_batch_size_per_gpu: int = 250
    learning_rate_init: float = 1e-3
    learning_rate_final: float = 1e-4
    num_warmup_steps: int = 2000

    seed = 43
    # strategy: str = "deepspeed_stage_2"  # "ddp" or "deepspeed_stage_2"
    strategy: str = "ddp"  # "ddp" or "deepspeed_stage_2"
    zero_opt_stage: int = 2
    num_gpus_per_node: int = 4
    num_nodes: int = 4
    train_infnet_only: bool = False
    train_denoiser_from_scratch: bool = True
    precision: str = "32"
    gradient_clipping: bool = False

    '''logging'''
    log_every_n_steps: int = 10
    checkpoint_every_n_epochs: int = 10