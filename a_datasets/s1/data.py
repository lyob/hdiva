from typing import Dict, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from a_datasets.disks.disks import make_disk_vectorized

class S1Dataset(Dataset):
    """a dataset of points that lie on the unit circle in 2D."""
    
    def __init__(self, n_samples: int = 1000, noise: float = 0.1):
        self.n_samples = n_samples
        self.noise = noise

        # Generate points on the unit circle
        angles = np.linspace(0, 2 * np.pi, n_samples)
        self.data = np.stack((np.cos(angles), np.sin(angles)), axis=1)

        # Add noise
        self.data += np.random.normal(scale=noise, size=self.data.shape)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        sample = self.data[idx]
        return {
            "data": torch.tensor(sample, dtype=torch.float32),
            "label": torch.tensor(0, dtype=torch.long)  # All points belong to the same class
        }