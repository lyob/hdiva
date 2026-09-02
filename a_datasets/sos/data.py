import numpy as np
import torch
from torch.utils.data import Dataset


# Dataset Definition
class SOSDataset(Dataset):
    def __init__(
        self,
        num_samples=1000,
        grid_size=28,
        sigma_x=1.0,
        sigma_y=1.0,
        seed=42,
        holdout_center=False,
        only_sample_holdout=False,
    ):
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.holdout_center = holdout_center
        self.only_sample_holdout = only_sample_holdout
        self.data_x = []
        self.data_y = []
        self.targets = []

        if only_sample_holdout:
            assert not holdout_center, "if sampling only from center, do not hold out center"

        np.random.seed(seed)
        for _ in range(num_samples):
            x, y = np.random.uniform(0, grid_size, size=2)  # Use continuous values for x and y

            # Skip samples that should be excluded based on holdout settings
            if self._check_holdout_center(x, y):
                continue

            sos_image = self.generate_image(x, y, sigma_x, sigma_y, grid_size)

            self.data_x.append(x)
            self.data_y.append(y)
            self.targets.append(sos_image)

    def generate_image(self, x, y, sigma_x=1.0, sigma_y=1.0, grid_size=28):
        xv, yv = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
        sos_image = 0.5 * np.exp(-((xv - x) ** 2) / (4 * sigma_x**2)) + 0.5 * np.exp(
            -((yv - y) ** 2) / (4 * sigma_y**2)
        )
        return sos_image

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
            (
                torch.tensor(self.data_x[idx], dtype=torch.float32),
                torch.tensor(self.data_y[idx], dtype=torch.float32),
            ),
            torch.tensor(self.targets[idx], dtype=torch.float32).unsqueeze(0),
        )


def load_sos_data(
    num_samples: int = 10000,
    grid_size: int = 28,
    sigma_x: float = 1.0,
    sigma_y: float = 1.0,
    seed: int = 42,
    holdout_center: bool = False,
    only_sample_holdout: bool = False,
):
    """Load bump data by generating it from SOSDataset.

    Args:
        num_samples: Number of samples to generate.
        grid_size: Size of the grid (image dimensions).
        sigma: Standard deviation for the SOS.
        seed: Random seed for reproducibility.
        holdout_center: If True, exclude samples from the center region.
        only_sample_holdout: If True, only sample from the center region.

    Returns:
        Tensor of shape (num_samples, 1, grid_size, grid_size) containing the bump images.
    """
    dataset = SOSDataset(
        num_samples=num_samples,
        grid_size=grid_size,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        seed=seed,
        holdout_center=holdout_center,
        only_sample_holdout=only_sample_holdout,
    )
    x, y = np.array(dataset.data_x), np.array(dataset.data_y)

    # Stack all targets into a single tensor with shape (num_samples, 1, grid_size, grid_size)
    cpu_tensor = torch.stack([torch.tensor(t, dtype=torch.float32) for t in dataset.targets]).unsqueeze(1)
    return cpu_tensor, (x, y)
