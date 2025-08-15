import torch
import numpy as np
import os
import time
from typing import Union, Tuple, List
import sysrsync
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from utils.cats import get_cats_dataset

class DataOnlyDataset(torch.utils.data.Dataset):
    '''makes it such that we can return only the data from the dataset and not the labels. For use with the cats dataset.'''
    def __init__(self, subset):
        self.subset = subset
        
    def __getitem__(self, index):
        # Get the (image, label) pair from the subset
        image, _ = self.subset[index]
        # Return only the image
        return image
    
    def __len__(self):
        return len(self.subset)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform, with_label=True):
        self.dataset = dataset
        self.transform = transform
        self.with_label = with_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = self.transform(item["image"])
        if not self.with_label:
            return image
        # Extract labels from the item, excluding the image key
        # Assuming the item is a dictionary with keys "image" and other attributes
        else:
            # Convert labels to integers if they are not already
            # Assuming item is a dictionary with keys "image" and other attributes
            if isinstance(item, dict):
                labels = {k: v for k, v in item.items() if k != "image"}
            elif isinstance(item, list):
                labels = {f"attr_{i}": v for i, v in enumerate(item) if i != 0}
            else:
                raise TypeError("Item must be a dictionary or a list")
            return image, labels

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


# ################### data loaders ####################
# def make_loader(train_set, test_set, args, num_workers=0):
#     trainloader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
#     testloader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
#     return trainloader, testloader

def move_celeba_data_to_tmp(dataset_resolution:int):
    # dataset_shape = int(dataset_name.split('-')[-2].split('x')[0])
    sysrsync.run(source=f'datasets/celeba/attribute_images_{dataset_resolution}x{dataset_resolution}.pt', 
                 destination=f'/tmp/attribute_images_{dataset_resolution}x{dataset_resolution}.pt')
