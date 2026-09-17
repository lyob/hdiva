from typing import Dict, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class EllipseDataset(Dataset):
    """An elongated Gaussian cloud generated from two independent factors.

    Axis scales are standard deviations; their squares are feature energies.
    Rotation is specified in degrees. Labels are the unscaled N(0, I) factors.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        noise: float = 0.0,
        scales: Tuple[float, float] = (4.0, 1.0),
        rotation: float = 30.0,
    ):
        self.n_samples = n_samples
        self.noise = noise
        self.scales = np.asarray(scales, dtype=float)
        self.rotation = rotation

        if self.scales.shape != (2,) or not np.all(
            np.isfinite(self.scales) & (self.scales > 0)
        ):
            raise ValueError("scales must contain two finite positive values")
        if not np.isfinite(noise) or noise < 0:
            raise ValueError("noise must be finite and nonnegative")
        if not np.isfinite(rotation):
            raise ValueError("rotation must be finite")

        self.factors = np.random.normal(size=(n_samples, 2))
        theta = np.deg2rad(rotation)
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)],
             [np.sin(theta), np.cos(theta)]]
        )
        # Accelerate's BLAS sets spurious FP flags on its SIMD tails, so large
        # matmuls warn about divide/overflow/invalid despite finite inputs.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            self.data = (self.factors * self.scales) @ rotation_matrix.T

        # Optional isotropic observation noise, separate from training corruption.
        if noise > 0:
            self.data += np.random.normal(scale=noise, size=self.data.shape)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        sample = self.data[idx]
        label = self.factors[idx]
        return {
            "data": torch.tensor(sample, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32)
        }
