import torch
import numpy as np
import os
import time
from typing import Union, Tuple, List
import sysrsync
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from a_datasets.cats import get_cats_dataset
from a_datasets.custom_dataset_classes import *

def load_dataset_from_hf(dataset_type:str, with_label:bool=False, transform=None, dataset_size:int=0, batch_size:int=None, num_workers:int=4, shuffle:bool=True, img_dim:int=64):
    if dataset_type == "celeba":
        hf_name = "huggan/CelebA-faces-with-attributes"
        split = "train"
    elif dataset_type == "celeba_color":
        hf_name = "flwrlabs/celeba"
        split = "train"
        transform = transforms.Compose([
            transforms.CenterCrop(160),  # needed for right proportions
            transforms.Resize((img_dim, img_dim)), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
        ])
    elif dataset_type == "shapes3d":
        hf_name = "eurecom-ds/shapes3d"
        split = "train"
        transform = transforms.Compose([
            transforms.Resize((img_dim, img_dim)),  # Resize to 64x64
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    dataset = load_dataset(hf_name, split=split)
    transformed_dataset = CustomDataset(dataset, transform=transform, with_label=with_label)
    if dataset_size != 0:
        train_size = dataset_size
        test_size = len(transformed_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(transformed_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    else:
        train_dataset = transformed_dataset
        test_dataset = None

    if batch_size is None:
        loader = DataLoader(train_dataset, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    else:
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return train_dataset, loader



def prepare_dataset(dataset_name, model_name, img_dim, dataset_size, local_rank):
    if dataset_name == 'celeba' or dataset_name == 'celeba_color' or dataset_name == "shapes3d":
        if local_rank == 0:
            print(f"Loading {dataset_name} dataset with image dimension {img_dim} and dataset size {dataset_size}")
        architecture_name = f"{model_name}_{dataset_name}_{img_dim}"
        train_dataset, _ = load_dataset_from_hf(dataset_type=dataset_name, with_label=False, img_dim=img_dim, dataset_size=dataset_size)
    elif dataset_name == 'cats' or dataset_name == 'cat' and img_dim == 64:
        architecture_name = f"{model_name}_cats_64"
        train_dataset, test_dataset = get_cats_dataset(from_tmp=False)
        dataset_size = len(train_dataset)
        dataset_info = f"cats-{img_dim}x{img_dim}-{dataset_size}"
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized")
        
    return architecture_name, train_dataset, dataset_info


def move_celeba_data_to_tmp(dataset_resolution:int):
    # dataset_shape = int(dataset_name.split('-')[-2].split('x')[0])
    sysrsync.run(source=f'datasets/celeba/attribute_images_{dataset_resolution}x{dataset_resolution}.pt', 
                 destination=f'/tmp/attribute_images_{dataset_resolution}x{dataset_resolution}.pt')


def load_dataset(config):
    if config.dataset_name == "hdisks3":
        from a_datasets.hdisks3 import random_two_disk_dataset
        data = random_two_disk_dataset(
                    img_size=config.input_dim,
                    outer_radius=4,
                    transition_width=2,
                    d=10,
                    num_imgs=config.dataset_size)[0]
    elif config.dataset_name == 'hdisks3_stochastic':
        from a_datasets.hdisks3_stochastic import random_two_disk_dataset
        data = random_two_disk_dataset(
                    img_size=config.input_dim,
                    outer_radius=4,
                    transition_width=2,
                    d=10,
                    num_imgs=config.dataset_size)[0]
    else:
        raise ValueError(f"Dataset {config.dataset_name} not recognized")
    dataset = DiskDataset(data)
    dataloader = DataLoader(dataset, 
                    batch_size=config.train_batch_size_per_gpu, 
                    shuffle=True, 
                    num_workers=7,
                    generator=torch.Generator().manual_seed(config.seed))
    return dataloader