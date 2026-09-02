import json
from dataclasses import dataclass, fields
from typing import Literal, Optional, Tuple, get_type_hints

import numpy as np
import torch

project_dir = "/mnt/home/blyo1/hdiva"

@dataclass
class GSMDatasetConfig:
    """GSM dataset parameters"""

    input_dim: int = 2
    latent_dim_y: int = 2
    latent_dim_z: int = 2

    dataset_name: str = "gsm"
    dataset_size: int = 20_000
    sigma_z: float = 0.5
    lambdas: Tuple[float, float] = (1.0, 0.25)
    A: Optional[np.ndarray] = None
    sigma_x: float = 0.1
    seed: int = 0
    data_cache_dir: str = f"{project_dir}/a_datasets/gsm/"

@dataclass
class RingDatasetFlatConfig:
    """Ring dataset parameters"""

    input_dim: int = 2
    latent_dim_y: int = 2
    latent_dim_z: int = 2

    dataset_name: str = "ring_flat"

    n: int = 20_000
    m: int = 4
    kappa: float = 1.0
    r0: float = 2.0
    sigma: float = 0.15
    seed: int | None = 0
    return_latents: bool = False
    data_cache_dir: str = f"{project_dir}/a_datasets/ring/"

@dataclass
class RingDatasetConfig:
    """Ring dataset parameters"""

    input_dim: int = 2
    latent_dim_y: int = 3
    latent_dim_z: int = 3

    dataset_name: str = "ring"

    n: int = 20_000
    m: int = 4
    kappa: float = 1.0
    kappa_z: float = 0.0
    r0: float = 2.0
    sigma: float = 0.15
    seed: int | None = 0
    return_latents: bool = False
    data_cache_dir: str = f"{project_dir}/a_datasets/ring/"

@dataclass
class HierarchicalDiskDatasetConfig:
    """Image dataset parameters"""

    input_dim: int = 32
    latent_dim_y: int = 2
    latent_dim_z: int = 1

    dataset_name: str = "hierarchical_disks"

    num_samples: int = 10_000  # 3e4
    grid_size: int = 28
    foreground: float = 1.
    background: float = 0.
    outer_radius: int = 6
    transition_width: int = 2
    dataset_seed: int = 42
    a: float = 1.0
    b: float = 1.0
    sigma: float = 0.5
    n_std: float = 3.0
    data_cache_dir: str = f"{project_dir}/a_datasets/hierarchical_disks/"


@dataclass
class HierarchicalTwoDiskDatasetConfig:
    """Image dataset parameters"""

    input_dim: int = 32
    latent_dim_y: int = 6
    latent_dim_z: int = 2

    dataset_name: str = "h2disks"

    # num_samples: int = 10_000  # 3e4
    # grid_size: int = 28
    # background: float = 0.
    # mu_I: float = 0.5
    # sigma_b: float = 0.14
    # sigma_y: float = 0.07
    # I_min: float = 0.05
    # I_max: float = 1.0
    # outer_radius: int = 5
    # transition_width: int = 2
    # dataset_seed: int = 42


    num_samples: int = 10_000
    grid_size: int = 28
    mu_I: float = 0.6
    background: float = 0.0
    sigma_d: float = 4.0
    sigma_eps: float = 1.5
    intensity_gain: float = 0.04
    sigma_eta: float = 0.04
    I_min: float = 0.1
    I_max: float = 1.0
    outer_radius: float = 5.5
    transition_width: int = 2
    dataset_seed: int = 42

    data_cache_dir: str = f"{project_dir}/a_datasets/h2disks/"

@dataclass
class DenoiserConfig:
    """denoiser parameters"""

    denoiser_x_type: str = "mlp"
    denoiser_y_type: str = "mlp"
    # bias: bool = False
    activation: str = "silu"  # silu or gelu

    hidden_dim_x: int = 128
    num_blocks_x: int = 4
    time_dim_x: int = 64

    hidden_dim_y: int = 128
    num_blocks_y: int = 4
    time_dim_y: int = 64

@dataclass
class DenoiserConvConfig:
    """denoiser parameters"""

    denoiser_x_type: str = "conv"
    denoiser_y_type: str = "mlp"
    # bias: bool = False

    num_channels_den_x: int = 64
    num_layers_den_x: int = 4
    time_dim_den_x: int = 64
    activation_den_x: Literal["relu", "gelu", "silu"] = "relu"
    bias_den_x: bool = True

    hidden_dim_den_y: int = 12
    num_blocks_den_y: int = 3
    time_dim_den_y: int = 12
    activation_den_y: str = "gelu"  # silu or gelu



@dataclass
class DDPMConfig:
    """DDPM parameters"""

    noise_schedule_x: str = "cosine_in_alpha_bar"  # cosine_in_alpha_bar or linear_in_alpha_bar
    noise_schedule_y: str = "cosine_in_alpha_bar"  # cosine_in_alpha_bar or linear_in_alpha_bar
    sigma_minmax_x: tuple = (0.0001, 0.9999)
    sigma_minmax_y: tuple = (0.0001, 0.9999)
    timestep_dist_x: str = "uniform"  # uniform, linearly_increasing
    timestep_dist_y: str = "uniform"  # uniform, linearly_increasing
    num_timesteps_x: int = 100
    num_timesteps_y: int = 100
    parameterization: str = "noise"  # velocity or noise or image
    reduction: str = "mean"  # mean or sum
    weighted_mse: bool = False


@dataclass
class RecNetConfig:
    """recnet (ConvNet)"""

    # encoder model
    recnet_type: str = "convnet"
    num_layers_rec_x_to_y: int = 3
    num_layers_rec_y_to_z: int = 3
    hidden_dim_rec: int = 128
    activation_rec: str = "silu"  # gelu or relu
    bias_rec: bool = True

@dataclass
class RecNetConvConfig:
    """recnet (ConvNet)"""

    # encoder model
    recnet_type: str = "convnet"

    in_channels: int = 1
    num_channels_rec: int = 64
    max_channels_rec: int = 128
    num_groups_rec: int = 8
    num_layers_rec_x_to_y: int = 2
    num_layers_rec_y_to_z: int = 2
    activation_rec: str = "gelu"  # gelu or relu
    bias_rec: bool = True
    # EMA momentum for the running y-latent scale (unit-variance normalization of y
    # before the y-diffusion). Higher = slower to track drift. See RecNetConv.normalize_y.
    y_scale_momentum: float = 0.99
    adaptive_avg_pool_output_size: int = 3


@dataclass
class HSAMIConfig:
    """sami loss"""

    model_name: str = "hsami_ring"
    reduction: str = "mean"  # "mean" or "sum"
    weighted_mse: bool = False
    weighted_rate: bool = False

    """beta"""
    rate_type: str = "grad"  # "grad", "norm", "cumulative", "kl"
    # overall constant for the r_eta^2 = rate_weight_scale/(2*alpha_bar) weight in
    # the "grad" (mutual-information) rate estimator. absorbs the derivation
    # constant + any uniform-t vs uniform-eta reweighting; tune if the rate term
    # is mis-scaled relative to the MSE terms.
    rate_weight_scale: float = 1.0
    beta_schedule: Literal["wait_then_anneal", "constant"] = "wait_then_anneal"
    beta_init: float = 1e-10
    beta_final: float = 1e-4
    beta_wait_epochs: int = 0
    beta_annealing_epochs: int = 2000
    beta_annealing_schedule: str = "cosine" # "linear" or "cosine" or "linear_in_log"

    """t_y curriculum: gradually open up the noisy-Y timestep range"""
    # floor on sqrt(alpha_bar_y) at step 0; e.g. 0.3 keeps amplification < 3x.
    # set to 0.0 to disable the curriculum entirely.
    t_y_min_sqrt_alpha_bar: float = 0.1
    # number of optimiser steps over which t_y_max grows from the capped value to
    # the full range.  ignored when t_y_min_sqrt_alpha_bar == 0.
    t_y_curriculum_steps: int = 120000

@dataclass
class TrainingConfig:
    """training"""

    num_epochs: int = 5000
    train_batch_size_per_gpu: int = 512

    encoder_lr_init: float = 2e-5
    encoder_lr_intermed: float = 2e-5
    encoder_lr_final: float = 2e-5
    encoder_wait_epochs: int = 1000
    encoder_warmup_epochs: int = 1000
    encoder_convergence_epochs: int = 0

    den_x_lr_schedule: str = "cosine"
    den_x_lr_init: float = 2e-5
    den_x_lr_final: float = 2e-5
    den_x_lr_num_warmup_epochs: int = 1000  # this is actually the number of steps
    
    den_y_lr_schedule: str = "cosine"
    den_y_lr_init: float = 2e-5
    den_y_lr_final: float = 2e-5
    den_y_lr_num_warmup_epochs: int = 1000  # this is actually the number of steps

    precision: str = "32"  # "32" or "bf16-mixed"
    training_seed: int = 43
    optimizer: str = "adam"

@dataclass
class PretrainingConfig:
    """pretraining"""

    resume_from_checkpoint: bool = False  # master switch for using checkpoint
    checkpoint_epoch: int | str = "last"

    # denoiser_x pretraining: unconditional DDPM on x before full joint training
    pretrain_denoiser_x: bool = True
    pretrain_denoiser_x_epochs: int = 1000

    use_pretrained_denoiser_x: bool = True  # use a pretrained ddpm or denoiser to initialize the denoiser, rather than use the entire pretrained diva model
    freeze_denoiser_x: bool = False  # if true, freeze the denoiser weights and only train the recnet

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
    checkpoint_every_n_epochs: int = 200

@dataclass
class HSAMI_Training_Config(
    GSMDatasetConfig,
    # RingDatasetFlatConfig,
    # RingDatasetConfig,
    DenoiserConfig,
    DDPMConfig,
    RecNetConfig,
    HSAMIConfig,
    TrainingConfig,
    PretrainingConfig,
    ClusterConfig,
):
    """model"""

    model_name: str = "hsami_gsm"

    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})

    def load_from_json(self, path: str):
        with open(path) as fp:
            d = json.load(fp)
        hints = get_type_hints(type(self))
        field_names = {f.name for f in fields(self)}
        for k, v in d.items():
            if k not in field_names:
                continue
            if isinstance(v, list):
                hint = hints.get(k)
                args = getattr(hint, "__args__", None)
                if hint is np.ndarray or (args and np.ndarray in args):
                    v = np.array(v)
                elif hint is tuple or getattr(hint, "__origin__", None) is tuple:
                    v = tuple(v)
            setattr(self, k, v)


@dataclass
class SelectConfig(
    HSAMI_Training_Config
):
    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})


@dataclass
class HSAMI_Ring_Training_Config(
    RingDatasetConfig,
    DenoiserConfig,
    DDPMConfig,
    RecNetConfig,
    HSAMIConfig,
    TrainingConfig,
    PretrainingConfig,
    ClusterConfig,
):
    """Training config for the wrapped-phase ring dataset."""

    model_name: str = "hsami_ring"

    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})

    def load_from_json(self, path: str):
        with open(path) as fp:
            d = json.load(fp)
        hints = get_type_hints(type(self))
        field_names = {f.name for f in fields(self)}
        for k, v in d.items():
            if k not in field_names:
                continue
            if isinstance(v, list):
                hint = hints.get(k)
                if hint is tuple or getattr(hint, "__origin__", None) is tuple:
                    v = tuple(v)
            setattr(self, k, v)

@dataclass
class HSAMI_Image_Training_Config(
    HierarchicalTwoDiskDatasetConfig,
    DenoiserConvConfig,
    DDPMConfig,
    RecNetConvConfig,
    HSAMIConfig,
    TrainingConfig,
    PretrainingConfig,
    ClusterConfig,
):
    """Training config for the image dataset."""

    model_name: str = "hsami_image"

    @classmethod
    def from_dict(cls, d: dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})

    def load_from_json(self, path: str):
        with open(path) as fp:
            d = json.load(fp)
        hints = get_type_hints(type(self))
        field_names = {f.name for f in fields(self)}
        for k, v in d.items():
            if k not in field_names:
                continue
            if isinstance(v, list):
                hint = hints.get(k)
                if hint is tuple or getattr(hint, "__origin__", None) is tuple:
                    v = tuple(v)
            setattr(self, k, v)
