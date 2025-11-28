import numpy as np
import matplotlib.pylab as plt
import torch.nn as nn
import torch
import os
from typing import Any, Tuple, List, Union
from numpy.typing import NDArray
from torchvision import transforms
from a_datasets.custom_dataset_classes import CustomTensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from a_datasets.custom_dataset_classes import NpyDataset


def load_dsprites_data(data_dir: str = '/mnt/ceph/users/blyo1/projects/hdiva/a_datasets/dsprites') -> NDArray:
    dsprites_path = os.path.join(data_dir, 'dsprites_dataset.npz')
    dataset = np.load(dsprites_path, allow_pickle=True, encoding='latin1')
    data = dataset['imgs']
    cpu_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    return cpu_tensor

def get_dataloader(batch_size: int = 4, 
                   shuffle: bool = True, 
                   seed: int = 42):
    # normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,)),
    ])
    cpu_tensor = load_dsprites_data()
    train_dataset = CustomTensorDataset(cpu_tensor, transform)
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def npz_to_npy(data_dir: str = '/mnt/ceph/users/blyo1/projects/hdiva/a_datasets/dsprites'):
    import numpy as np
    dataset_dir = f"{data_dir}/a_datasets/dsprites"

    # 1. Load the compressed npz (loads keys, not data yet)
    npz_data = np.load(f"{dataset_dir}/dsprites_dataset.npz")
    print(npz_data.files)

    # 2. Extract the specific array you need and save as .npy
    # This step requires RAM for the full array temporarily. 
    # If you have strictly limited RAM, you may need a chunked read/write solution.
    np.save(f'{dataset_dir}/dataset_images.npy', npz_data['imgs'])
    np.save(f'{dataset_dir}/dataset_labels.npy', npz_data['latents_values'])

def get_dataloader_from_npy(batch_size: int = 4, 
                            shuffle: bool = True, 
                            seed: int = 42,
                            data_dir: str = '/mnt/ceph/users/blyo1/projects/hdiva/a_datasets/dsprites',
                            data_only: bool = False,
                            ):
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.unsqueeze(0)),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = NpyDataset(os.path.join(data_dir, 'dataset_images.npy'), 
                         os.path.join(data_dir, 'dataset_labels.npy'), 
                         transform=transform, 
                         data_only=data_only)

    generator = torch.Generator().manual_seed(seed) if shuffle else None
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=4, 
        generator=generator,
    )
    return dataloader