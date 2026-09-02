from dataclasses import dataclass, fields

project_dir = "/mnt/home/blyo1/hdiva"


@dataclass
class DspritesDatasetConfig:
    """dsprites dataset parameters"""

    image_dim: int = 64
    dataset_name: str = "dsprites"
    dataset_size: int = 737280


@dataclass
class BumpDatasetConfig:
    """bump dataset parameters"""

    image_dim: int = 28
    dataset_name: str = "bump"
    dataset_size: int = 100000
    holdout_center: bool = False


@dataclass
class SOSDatasetConfig:
    """sos dataset parameters"""

    image_dim: int = 28
    dataset_name: str = "sos"
    dataset_size: int = 100000
    holdout_center: bool = False


@dataclass
class CircleInSquareDatasetConfig:
    """circle in square dataset parameters"""

    image_dim: int = 64
    dataset_name: str = "circle_in_square"
    dataset_size: int = 100000


@dataclass
class VAEConfig:
    """VAE parameters"""

    input_dim: int = 64
    input_channel: int = 1
    hidden_channels: tuple = (128, 128, 128)
    z_dim: int = 4
    num_blocks: int = 2

    weighted_MSE: bool = False
    reduction: str = "mean"  # "mean" or "sum"

    """kl weights"""
    use_kl: bool = True
    kl_weight: float = 1e-14
    kl_annealing_schedule: str = "cosine"
    kl_weight_max: float = 1
    kl_weight_min: float = 0.1
    kl_annealing_epochs: int = 2000


@dataclass
class TrainingConfig:
    """training"""

    num_epochs: int = 4000
    train_batch_size_per_gpu: int = 512
    lr_schedule: str = "cosine"
    # lr_init: float = 4e-3
    # lr_final: float = 1e-3
    lr_init: float = 6e-3
    lr_final: float = 1e-3
    lr_num_warmup_epochs: int = 20000  # this is actually the number of steps
    precision: str = "bf16-mixed"
    seed: int = 43


@dataclass
class PretrainingConfig:
    """pretraining"""

    resume_from_checkpoint: bool = False  # master switch for using checkpoint
    use_pretrained_denoiser_only: bool = False  # use a pretrained ddpm or denoiser to initialize the denoiser, rather than use the entire pretrained diva model
    train_infnet_only: bool = False  # if true, freeze the denoiser weights and only train the infnet
    pretrained_project_name: str = "ddpm_dsprites"
    pretrained_model_num: int = 2
    pretrained_artifact_id: str = "v29"

    precision: str = "32"
    project_dir: str = project_dir
    model_checkpoint_dir: str = f"{project_dir}/c_training/lightning_checkpoints"
    data_cache_dir: str = f"{project_dir}/a_datasets/dsprites/"


@dataclass
class ClusterConfig:
    """cluster and logging"""

    # cluster
    strategy: str = "ddp"  # "ddp" or "deepspeed_stage_2"
    num_nodes: int = 1
    num_gpus_per_node: int = 2

    # logging
    log_every_n_steps: int = 10
    checkpoint_every_n_epochs: int = 100


@dataclass
class VAE_CIS_Training_Config(
    CircleInSquareDatasetConfig,
    VAEConfig,
    TrainingConfig,
    PretrainingConfig,
    ClusterConfig,
):
    """model"""

    model_name: str = "vae_cis"

    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})


@dataclass
class SelectConfig(
    VAE_CIS_Training_Config,
):
    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})
