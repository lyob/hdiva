from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from a_datasets.disks.disks import make_disk_vectorized


# Dataset Definition
class HierarchicalDiskDataset(Dataset):
    """Two-level latent hierarchy on top of the disk renderer.

    z   ~ N(0, 1)                          (top-level latent)
    c_x = a * z + eps_x,  eps_x ~ N(0, sigma^2)
    c_y = b * z + eps_y,  eps_y ~ N(0, sigma^2)

    (c_x, c_y) are unbounded, so candidates whose raw value falls outside
    +/- n_std marginal standard deviations are rejected and resampled; the
    accepted range is then linearly mapped onto the valid pixel-space disk
    centers before rendering.
    """
    def __init__(
        self,
        num_samples: int = 1000,
        grid_size: int = 28,
        foreground: float = 1.,
        background: float = 0.,
        outer_radius: int = 6,
        transition_width: int = 2,
        dataset_seed: int = 42,
        a: float = 1.0,
        b: float = 1.0,
        sigma: float = 0.5,
        n_std: float = 3.0,
    ):
        assert grid_size > 2 * outer_radius
        assert foreground >= 0 and foreground <= 1
        assert background >= 0 and background <= 1
        assert sigma > 0
        assert n_std > 0

        self.num_samples = num_samples
        self.grid_size = grid_size
        self.outer_radius = outer_radius
        self.transition_width = transition_width
        self.a = a
        self.b = b
        self.sigma = sigma
        self.n_std = n_std

        self.data_z = torch.ones(self.num_samples, dtype=torch.float32) * torch.nan
        self.data_x = torch.ones(self.num_samples, dtype=torch.float32) * torch.nan
        self.data_y = torch.ones(self.num_samples, dtype=torch.float32) * torch.nan
        self.targets = torch.ones(self.num_samples, 1, self.grid_size, self.grid_size, dtype=torch.float32) * torch.nan

        np.random.seed(dataset_seed)
        c_min = outer_radius
        c_max = self.grid_size - outer_radius
        self.cx_range = (c_min, c_max)
        self.cy_range = (c_min, c_max)

        # marginal std of the raw (unbounded) children under the generative model
        self.std_x = np.sqrt(a ** 2 + sigma ** 2)
        self.std_y = np.sqrt(b ** 2 + sigma ** 2)
        # raw window that gets kept (via rejection) and linearly mapped onto [c_min, c_max]
        self.raw_x_range = (-n_std * self.std_x, n_std * self.std_x)
        self.raw_y_range = (-n_std * self.std_y, n_std * self.std_y)

        for i in range(self.num_samples):
            z, c_x_raw, c_y_raw = self.sample_latents()
            c_x = self._raw_to_pixel(c_x_raw, self.raw_x_range, c_min, c_max)
            c_y = self._raw_to_pixel(c_y_raw, self.raw_y_range, c_min, c_max)

            random_disk, cx, cy = self.generate_image(
                c_x=c_x, c_y=c_y,
                foreground=foreground,
                background=background,
                img_size=self.grid_size,
                outer_radius=self.outer_radius,
                transition_width=self.transition_width
            )
            self.data_z[i] = z
            self.data_x[i] = cx
            self.data_y[i] = cy
            self.targets[i] = random_disk

    def sample_latents(self) -> Tuple[float, float, float]:
        """Rejection-sample (z, c_x_raw, c_y_raw) until both raw children fall
        within +/- n_std standard deviations of their marginal distribution."""
        x_lo, x_hi = self.raw_x_range
        y_lo, y_hi = self.raw_y_range
        while True:
            z = np.random.randn()
            eps_x = np.random.randn() * self.sigma
            eps_y = np.random.randn() * self.sigma
            c_x_raw = self.a * z + eps_x
            c_y_raw = self.b * z + eps_y

            if x_lo <= c_x_raw <= x_hi and y_lo <= c_y_raw <= y_hi:
                return z, c_x_raw, c_y_raw

    @staticmethod
    def _raw_to_pixel(raw_val: float, raw_range: Tuple[float, float], pixel_min: float, pixel_max: float) -> float:
        raw_lo, raw_hi = raw_range
        frac = (raw_val - raw_lo) / (raw_hi - raw_lo)
        return pixel_min + frac * (pixel_max - pixel_min)

    def generate_image(self, c_x: float, c_y: float, foreground: float = 1, background: float = 0, img_size: int = 32, outer_radius: int = 10, transition_width: int = 2):
        c_min = outer_radius
        c_max = img_size - outer_radius

        # Normalize to [-1, 1] using the per-axis ranges stored on the dataset
        cx_min, cx_max = self.cx_range
        cy_min, cy_max = self.cy_range
        cx = (c_x - cx_min) / (cx_max - cx_min) * 2 - 1
        cy = (c_y - cy_min) / (cy_max - cy_min) * 2 - 1

        assert c_x >= c_min and c_x <= c_max
        assert c_y >= c_min and c_y <= c_max

        disk = make_disk_vectorized(
            (img_size, img_size),
            outer_radius,
            transition_width,
            foreground=foreground,
            background=background,
            c_x=c_x,
            c_y=c_y,
        )

        return disk, cx, cy

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.targets[idx]
    # (
            # (
            #     # torch.tensor(self.data_z[idx], dtype=torch.float32),
            #     # torch.tensor(self.data_x[idx], dtype=torch.float32),
            #     # torch.tensor(self.data_y[idx], dtype=torch.float32),
            #     self.data_x[idx],
            #     self.data_y[idx],
            #     self.data_z[idx],
            # ),
            # torch.tensor(self.targets[idx], dtype=torch.float32).unsqueeze(0),
            
        # )
            


def load_hierarchical_disk_dataset(
    num_samples: int = int(2e3),
    img_size: int = 28,
    foreground: float = 1.,
    background: float = 0.,
    outer_radius: int = 8,
    transition_width: int = 2,
    dataset_seed: int = 42,
    a: float = 1.0,
    b: float = 1.0,
    sigma: float = 0.5,
    n_std: float = 3.0,
):
    """Create a hierarchical disk dataset with top-level latent z and children c_x, c_y."""

    dataset = HierarchicalDiskDataset(
        num_samples = num_samples,
        grid_size = img_size,
        foreground = foreground,
        background = background,
        outer_radius = outer_radius,
        transition_width = transition_width,
        dataset_seed = dataset_seed,
        a = a,
        b = b,
        sigma = sigma,
        n_std = n_std,
    )

    z, x, y = np.array(dataset.data_z), np.array(dataset.data_x), np.array(dataset.data_y)
    cpu_tensor = dataset.targets

    return cpu_tensor, (z, x, y)
