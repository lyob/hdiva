import os
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from a_datasets.custom_dataset_classes import DataOnlyDataset

# ------------------------------- cats dataset ------------------------------- #
def get_cats_dataset(train_split_percentage:float=0.8, from_tmp=False):
    if from_tmp:
        base_path = '/tmp'
    else:
        base_path = 'datasets/cat-dataset'
    
    data_path = os.path.join(base_path, 'cats')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = ImageFolder(data_path, transform=transform)
    
    train_size = int(train_split_percentage * len(dataset))  # 80% for training
    test_size = len(dataset) - train_size  # Remaining 20% for testing

    # Create indices for the splits
    train_indices = list(range(train_size))  # Indices for training set
    test_indices = list(range(train_size, len(dataset)))  # Indices for testing set

    # Create subsets for training and testing
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    train_dataset = DataOnlyDataset(train_dataset)
    test_dataset = DataOnlyDataset(test_dataset)

    print('train subset: ', len(train_dataset))
    return train_dataset, test_dataset