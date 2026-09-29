import os

import torch

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_celeba_old_data(trainset_size: int = 60000, split: str = "train", img_size: int = 64):
    """Load the grayscale celeba tensor the autoprior models were trained on.

    The file holds all 202599 images in filename order as float32 in [0, 1], shape (N, 1, H, W).
    The autoprior models were trained on the first `trainset_size` images; the rest are the test split.
    """
    data_path = os.path.join(DATA_DIR, f"attribute_images_{img_size}x{img_size}.pt")
    # mmap avoids reading the full 3.3GB file when we only need a slice of it
    dataset = torch.load(data_path, map_location="cpu", mmap=True)

    if split == "train":
        subset = dataset[:trainset_size]
    elif split == "test":
        subset = dataset[trainset_size:]
    else:
        raise ValueError(f"Unknown split: {split}, expected 'train' or 'test'")

    # clone so the slice owns its memory and doesn't keep the mmap'd file alive
    return subset.clone()
