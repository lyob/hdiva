import torch
from torch.utils.data import Dataset
import numpy as np

class DataOnlyDataset(Dataset):
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


class CustomDataset(Dataset):
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
        

class DiskDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = lambda x: x * 2 - 1  # normalize to [-1, 1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Apply transformation if needed
        data = self.transform(self.data[idx])
        return data


class CustomTensorDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item = self.dataset[index]
        return self.transform(item)


class NpyDataset(Dataset):
    def __init__(self, data_path, label_path, transform=None, data_only=False):
        # mmap_mode='r' is the magic key here.
        # It opens the file in read-only mode without loading it to RAM.
        self.data = np.load(data_path, mmap_mode='r')
        self.labels = np.load(label_path, mmap_mode='r')
        self.transform = transform
        self.data_only = data_only

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Accessing self.data[idx] only reads that specific slice from disk.
        sample = self.data[idx]
        label = self.labels[idx]

        # Note: sample is currently a numpy memmap object. 
        # Convert to copy if you plan to mutate it, or just to tensor.
        sample = torch.from_numpy(np.array(sample)) # np.array() forces a copy into RAM
        sample = sample.float()
        label = torch.tensor(label)

        if self.transform:
            sample = self.transform(sample)

        if self.data_only:
            return sample
        else:
            return sample, label