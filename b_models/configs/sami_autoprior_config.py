from dataclasses import dataclass, fields

project_dir = "/mnt/home/blyo1/hdiva"


@dataclass
class SAMI_AutoPrior_CelebA_64_Training_Config:
    '''SAMI (infomax) fine-tuning of the pretrained autoprior_celeba_64:v77 model.

    Architecture fields use the autoprior's names (image_dims, latent_dims, ...) because the
    autoprior modules read them directly from this config; values match the v77 run.
    '''
    '''general model parameters'''
    model_name: str = "sami_autoprior_celeba_64"
    image_dims: int = 64
    num_channels: int = 1
    latent_dims: int = 256

    '''dataset'''
    dataset_name: str = "celeba_old"
    trainset_size: int = 60000  # the autoprior trained on the first 60000 images; the rest is the test set

    '''denoiser (autoprior UNet, time as input)'''
    num_kernels: int = 128
    kernel_size: int = 3
    padding: int = 1
    bias: bool = False
    time_channels: int = 64
    num_blocks: int = 3
    num_enc_conv: int = 2
    num_mid_conv: int = 2
    num_dec_conv: int = 2
    pool_window: int = 2

    '''infnet (autoprior HalfUNet)'''
    num_kernels_rec: int = 64
    activation_rec: str = "relu"
    bias_rec: bool = False
    kernel_size_rec: int = 3
    padding_rec: int = 1
    num_blocks_rec: int = 3
    num_enc_conv_rec: int = 2
    num_mid_conv_rec: int = 2
    pool_window_rec: int = 2
    downsample_in_mid_block_rec: bool = False
    use_full_cov: bool = False
    epsilon: float = 1e-4

    '''diffusion'''
    noise_schedule: str = "cosine_in_alpha_bar"  # uses the autoprior's s=0.01 offset (see SAMI_AutoPrior)
    sigma_minmax: tuple = (0.0001, 0.9999)
    num_timesteps: int = 1000
    timestep_dist: str = "uniform"
    parameterization: str = "noise"  # the autoprior was trained with denoiser_target="noise"

    '''sami loss'''
    reduction: str = "sum"  # the autoprior summed the MSE over pixels and averaged over the batch
    weighted_mse: bool = False
    weighted_rate: bool = False
    rate_type: str = "norm"  # "grad", "norm", "cumulative", "kl" ("kl" with beta=1e-3 reproduces the autoprior objective)
    beta_init: float = 1e-5
    beta_final: float = 1e-3
    beta_wait_epochs: int = 0
    beta_annealing_epochs: int = 100
    beta_annealing_schedule: str = "constant"  # "constant", "linear", "cosine", "exponential"

    '''training'''
    num_epochs: int = 1000
    train_batch_size_per_gpu: int = 512  # the autoprior used a batch size of 512
    lr: float = 1e-3
    encoder_lr: float = 1e-3  # only gets its own param group when it differs from lr
    optimizer: str = "adam"
    precision: str = "32"
    seed: int = 43

    '''pretrained autoprior'''
    load_autoprior_weights: bool = True
    train_infnet_only: bool = False  # if true, freeze the denoiser and only train the infnet
    pretrained_entity: str = "blyo"
    pretrained_model_name: str = "autoprior_celeba_64"
    pretrained_artifact_id: str = "v77"

    '''resume from a lightning checkpoint of this project'''
    resume_from_checkpoint: bool = False
    pretrained_model_number: int = 1
    checkpoint_epoch: int | str = "last"

    '''paths'''
    project_dir: str = project_dir
    model_checkpoint_dir: str = f"{project_dir}/c_training/lightning_checkpoints"

    '''cluster'''
    strategy: str = "ddp"
    num_nodes: int = 1
    num_gpus_per_node: int = 4

    '''logging'''
    log_every_n_steps: int = 10
    checkpoint_every_n_epochs: int = 50

    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})
