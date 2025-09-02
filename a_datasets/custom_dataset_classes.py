import torch
from torch.utils.data import Dataset

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
    def __init__(self, tensor):
        self.tensor = tensor
    
    def __getitem__(self, index):
        return self.tensor[index]
    
    def __len__(self):
        return self.tensor.shape[0]