from dataclasses import dataclass, fields, field
import json
from typing import Literal, Optional, Tuple, get_type_hints
import numpy as np

project_dir = "/mnt/home/blyo1/hdiva"


@dataclass
class RingDatasetConfig:
    """Ring dataset parameters"""

    num_samples: int = 20000
    input_dim: int = 2

    dataset_name: str = "ring"

    noise: float = 0.1
    seed: int | None = 0
    return_latents: bool = False
    data_cache_dir: str = f"{project_dir}/a_datasets/ring/"


@dataclass
class EllipseDatasetConfig:
    """Ring dataset parameters"""

    num_samples: int = 20000
    input_dim: int = 2

    dataset_name: str = "ellipse"

    noise: float = 0.0
    scales: Tuple[float, float] = (4.0, 1.0),
    rotation: float = 30.0,
    return_latents: bool = False
    data_cache_dir: str = f"{project_dir}/a_datasets/ellipse/"



@dataclass
class DenoiserConfig:
    """Denoiser parameters"""

    denoiser_type: str = "mlp"
    parameterization: str = "noise"  # velocity or noise or image

    denoiser_act_fn: str = "relu"
    bias: bool = False
    time_embedding_method: str = "as_input"  # as_input, per_layer, film
    hidden_dim: int = 32
    num_layers: int = 2
    time_channels: int = 32

@dataclass
class DDPMConfig:
    """DDPM parameters"""

    timestep_dist: str = "uniform"  # uniform, linearly_increasing
    num_timesteps: int = 100
    beta_minmax: tuple = (1e-4, 2e-2)
    sigma_minmax: tuple = (0.0001, 0.9999)
    noise_schedule: str = "cosine_in_alpha_bar"  # cosine_in_alpha_bar or linear_in_alpha_bar

@dataclass
class RecNetConfig:
    """recnet (ConvNet)"""

    # encoder model
    infnet_type: str = "mlp"
    num_layers_rec: int = 2
    hidden_dim_rec: int = 32
    activation_rec: str = "gelu"  # gelu or relu
    latent_dim: int = 2
    bias_rec: bool = True
    # conditions q(z|x_t) on the noise level; "none" collapses the latent to the prior
    time_embedding_method_rec: str = "as_input"  # none, as_input, per_layer, film

@dataclass
class SAMIConfig:
    """sami loss"""

    model_name: str = "sami_ellipse"
    reduction: str = "sum"  # "mean" or "sum"
    weighted_mse: bool = False
    weighted_rate: bool = False

    """beta"""
    rate_type: str = "norm"  # "grad", "norm", "cumulative", "kl"
    # rate_type: str = "norm"  # "grad", "norm", "cumulative", "kl"
    # a list trains one model per value, each saved with its own float beta_init
    beta_init: float | list[float] = field(default_factory=lambda: [2., 1.5, 1.2])
    # beta_init: float | list[float] = field(default_factory=lambda: [0.0001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.12, 0.15, 0.19, 0.2, 0.3])

@dataclass
class TrainingConfig:
    """training parameters"""

    seed: int = 0
    batch_size: int = 2000
    num_epochs: int = 8e2
    lr: float = 3e-4
    log_every_n_epochs: int = 100
    pretrained_denoiser: int | None = 1
    freeze_denoiser: bool = False


@dataclass
class Config(
    # RingDatasetConfig,
    EllipseDatasetConfig,
    DenoiserConfig,
    DDPMConfig,
    RecNetConfig,
    SAMIConfig,
    TrainingConfig,
):
    """model"""

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

