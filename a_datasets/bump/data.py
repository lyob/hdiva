import numpy as np
import torch
from torch.utils.data import Dataset


# Dataset Definition
class GaussianBumpDataset(Dataset):
    def __init__(
        self,
        num_samples=1000,
        grid_size=28,
        sigma=1.0,
        seed=42,
        scalar_encoding=False,
        holdout_center=False,
        only_sample_holdout=False,
    ):
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.sigma = sigma
        self.scalar_encoding = scalar_encoding
        self.holdout_center = holdout_center
        self.only_sample_holdout = only_sample_holdout
        self.data_x = []
        self.data_y = []
        self.targets = []

        if only_sample_holdout:
            assert not holdout_center, "if sampling only from center, do not hold out center"

        np.random.seed(seed)
        for _ in range(num_samples):
            if self.scalar_encoding:
                x, y = np.random.uniform(0, grid_size, size=2)  # Use continuous values for x and y
            else:
                x, y = np.random.randint(0, grid_size, size=2)

            # Skip samples that should be excluded based on holdout settings
            if self._check_holdout_center(x, y):
                continue

            # Generate gaussian encodings for non-scalar encoding
            if not self.scalar_encoding:
                gaussian_x = np.exp(-((np.arange(grid_size) - x) ** 2) / (2 * sigma**2))
                gaussian_y = np.exp(-((np.arange(grid_size) - y) ** 2) / (2 * sigma**2))

            xv, yv = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
            gaussian_image = np.exp(-((xv - x) ** 2 + (yv - y) ** 2) / (2 * sigma**2))

            if self.scalar_encoding:
                self.data_x.append(x)
                self.data_y.append(y)
            else:
                self.data_x.append(gaussian_x)
                self.data_y.append(gaussian_y)
            self.targets.append(gaussian_image)

    def _check_holdout_center(self, x, y):
        holdout_min = int(0.2 * self.grid_size) + 1
        holdout_max = int(0.8 * self.grid_size)
        if self.holdout_center and (holdout_min <= x <= holdout_max and holdout_min <= y <= holdout_max):
            return True
        if self.only_sample_holdout and not (holdout_min <= x <= holdout_max and holdout_min <= y <= holdout_max):
            return True
        return False

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            (torch.tensor(self.data_x[idx], dtype=torch.float32), torch.tensor(self.data_y[idx], dtype=torch.float32)),
            torch.tensor(self.targets[idx], dtype=torch.float32).unsqueeze(0),
        )


def load_bump_data(
    num_samples: int = 10000,
    grid_size: int = 28,
    sigma: float = 1.0,
    seed: int = 42,
    holdout_center: bool = False,
    only_sample_holdout: bool = False,
):
    """Load bump data by generating it from GaussianBumpDataset.

    Args:
        num_samples: Number of samples to generate.
        grid_size: Size of the grid (image dimensions).
        sigma: Standard deviation for the Gaussian bump.
        seed: Random seed for reproducibility.
        holdout_center: If True, exclude samples from the center region.
        only_sample_holdout: If True, only sample from the center region.

    Returns:
        Tensor of shape (num_samples, 1, grid_size, grid_size) containing the bump images.
    """
    dataset = GaussianBumpDataset(
        num_samples=num_samples,
        grid_size=grid_size,
        sigma=sigma,
        seed=seed,
        scalar_encoding=True,
        holdout_center=holdout_center,
        only_sample_holdout=only_sample_holdout,
    )
    x, y = np.array(dataset.data_x), np.array(dataset.data_y)

    # Stack all targets into a single tensor with shape (num_samples, 1, grid_size, grid_size)
    cpu_tensor = torch.stack([torch.tensor(t, dtype=torch.float32) for t in dataset.targets]).unsqueeze(1)
    return cpu_tensor, (x, y)
