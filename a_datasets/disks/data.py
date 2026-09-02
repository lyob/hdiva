import numpy as np
import torch
from torch.utils.data import Dataset
from a_datasets.disks.disks import make_disk_vectorized
from typing import Optional, Tuple, Union


# Dataset Definition
class SimpleDiskDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 1000,
        grid_size: int = 28,
        foreground: float = 1.,
        background: float = 0.,
        outer_radius: int = 6, 
        transition_width: int = 2,
        seed: int = 42,
        cx_range: Optional[Tuple[float, float]] = None,
        cy_range: Optional[Tuple[float, float]] = None,
    ):
        assert grid_size > 2 * outer_radius
        assert foreground >= 0 and foreground <= 1
        assert background >= 0 and background <= 1

        self.num_samples = num_samples
        self.grid_size = grid_size
        self.outer_radius = outer_radius
        self.transition_width = transition_width
        self.data_x = torch.ones(self.num_samples) * torch.nan
        self.data_y = torch.ones(self.num_samples) * torch.nan
        self.targets = torch.ones(self.num_samples, 1, self.grid_size, self.grid_size) * torch.nan

        np.random.seed(seed)
        default_c_min = outer_radius
        default_c_max = self.grid_size - outer_radius

        cx_min, cx_max = cx_range if cx_range is not None else (default_c_min, default_c_max)
        cy_min, cy_max = cy_range if cy_range is not None else (default_c_min, default_c_max)

        assert cx_min >= default_c_min and cx_max <= default_c_max, \
            f"cx_range ({cx_min}, {cx_max}) must be within ({default_c_min}, {default_c_max})"
        assert cy_min >= default_c_min and cy_max <= default_c_max, \
            f"cy_range ({cy_min}, {cy_max}) must be within ({default_c_min}, {default_c_max})"

        self.cx_range = (cx_min, cx_max)
        self.cy_range = (cy_min, cy_max)

        # dataset of uniformly sampled variables
        cxs = cx_min + np.random.rand(self.num_samples) * (cx_max - cx_min)
        cys = cy_min + np.random.rand(self.num_samples) * (cy_max - cy_min)

        for i, (cx, cy) in enumerate(zip(cxs, cys)):
            random_disk, cx, cy = self.generate_image(
                c_x=cx, c_y=cy, 
                foreground=foreground, 
                background=background, 
                img_size=self.grid_size, 
                outer_radius=self.outer_radius, 
                transition_width=self.transition_width
            )
            self.data_x[i] = cx
            self.data_y[i] = cy
            self.targets[i] = random_disk
    
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
        return (
            (
                torch.tensor(self.data_x[idx], dtype=torch.float32),
                torch.tensor(self.data_y[idx], dtype=torch.float32),
            ),
            torch.tensor(self.targets[idx], dtype=torch.float32).unsqueeze(0),
        )


def load_simple_disk_dataset(
    num_samples: int = int(2e3), 
    img_size: int = 28, 
    foreground: float = 1.,
    background: float = 0.,
    outer_radius: int = 8, 
    transition_width: int = 2,
    seed: int = 42,
    cx_range: Optional[Tuple[float, float]] = None,
    cy_range: Optional[Tuple[float, float]] = None,
):
    """Create a simple disk dataset."""

    dataset = SimpleDiskDataset(
        num_samples = num_samples,
        grid_size = img_size,
        foreground = foreground,
        background = background,
        outer_radius = outer_radius, 
        transition_width = transition_width,
        seed = seed,
        cx_range = cx_range,
        cy_range = cy_range,
    )

    x, y = np.array(dataset.data_x), np.array(dataset.data_y)
    # cpu_tensor = torch.stack([torch.tensor(t, dtype=torch.float32) for t in dataset.targets]).unsqueeze(1)
    cpu_tensor = dataset.targets

    return cpu_tensor, (x, y)