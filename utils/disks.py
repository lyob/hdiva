import wandb
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt
from torch import Tensor
from PIL import Image
import gzip
from matplotlib.patches import Circle, Polygon
import random
import pywt

# ---------------------------------------------------------------------------- #
#                                 disk dataset                                 #
# ---------------------------------------------------------------------------- #

def load_disks_dataset_from_wandb():
    api = wandb.Api()
    entity='blyo'
    project_name='ddpm_conv'
    dataset_name = "disks-32x32-soft-50000"
    dataset_alias = 'v3'
    dataset_artifact = api.artifact(f'{entity}/{project_name}/{dataset_name}:{dataset_alias}')
    dataset_path = dataset_artifact.download()
    dataset_description = dataset_artifact.metadata
    dataset = torch.load(f'{dataset_path}/training_data')
    return dataset, dataset_description


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = lambda x: x * 2 - 1  # normalize to [-1, 1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Apply transformation if needed
        data = self.transform(self.data[idx])
        return data


def make_disk_vectorized(
    img_size: Union[int, Tuple[int, int], torch.Size],
    outer_radius: Optional[float] = None,
    transition_width: Optional[float] = None,
    foreground: Optional[float] = None,
    background: Optional[float] = None,
    c_x: Optional[float] = None,
    c_y: Optional[float] = None,
) -> Tensor:

    inner_radius = outer_radius - transition_width
    
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    assert len(img_size) == 2

    if outer_radius is None:
        outer_radius = (min(img_size) - 1) / 2

    if inner_radius is None:
        inner_radius = outer_radius / 2


    # Create a meshgrid for the x, y coordinates
    y, x = np.ogrid[:img_size[0], :img_size[1]]

    # Calculate the radial distance for each pixel from the center
    r = np.sqrt((y - c_y) ** 2 + (x - c_x) ** 2)

    # Initialize the mask as zeros
    mask = np.zeros(img_size)

    # Apply the conditions to the mask
    mask[r < inner_radius] = foreground  # Inside the inner radius, set to 1
    mask[r > outer_radius] = background  # Outside the outer radius, set to 0

    # Calculate the radial decay for pixels between inner and outer radius
    radial_decay = (r - inner_radius) / (outer_radius - inner_radius)

    # Apply the radial decay with the cosine function only for pixels between the radii
    mask[(r >= inner_radius) & (r <= outer_radius)] = background + (foreground - background) * (1 + np.cos(np.pi * radial_decay[(r >= inner_radius) & (r <= outer_radius)])) / 2

    return torch.tensor(mask, dtype=torch.float).unsqueeze(0)



def create_soft_edge_circle_dataset(n_samples=1000, img_size=(32, 32), outer_radius=10, transition_width=2):
    n_samples = int(n_samples)
    img_size = (32, 32)
    outer_radius = 6
    transition_width = 2
    foreground = 0.5
    
    # let's choose to vary the c_x, c_y, and foreground of the circle
    c_min = 6
    c_max = 32-6
    c_x = c_min + np.random.rand(n_samples) * (c_max - c_min)
    c_y = c_min + np.random.rand(n_samples) * (c_max - c_min)
    background = np.random.rand(n_samples)
    
    disks = []
    for i in range(n_samples):
        disk = make_disk_vectorized(
            img_size,
            outer_radius,
            transition_width,
            foreground=foreground,
            background=background[i],
            c_x=c_x[i],
            c_y=c_y[i],
        )
        disks.append(disk)
    # disks = torch.tensor(np.array(disks))
    disks = torch.stack(disks)
    
    return disks, c_x, c_y, background


def create_one_circle(c_x, c_y, background):
    """
    Create a single circle with the given parameters.
    Args:
        c_x (float): x-coordinate of the center of the circle.
        c_y (float): y-coordinate of the center of the circle.
        background (float): background value (0 to 1).
    Returns:
        disk (Tensor): The generated disk.
        c_x (float): x-coordinate of the center of the circle.
        c_y (float): y-coordinate of the center of the circle.
        background (float): background value (0 to 1).
    """
    img_size = (32, 32)
    outer_radius = 10
    transition_width = 2
    foreground = 0.5
    
    # let's choose to vary the c_x, c_y, and foreground of the circle
    c_min = 10
    c_max = 32-10
    
    assert c_x >= c_min and c_x <= c_max
    assert c_y >= c_min and c_y <= c_max
    assert background >= 0 and background <= 1

    disk = make_disk_vectorized(
        img_size,
        outer_radius,
        transition_width,
        foreground=foreground,
        background=background,
        c_x=c_x,
        c_y=c_y,
    )
    
    return disk, c_x, c_y, background


def create_one_circle_fractional_1(c_x, c_y, foreground=1, outer_radius=10, transition_width=2):
    '''both cx and cy are fractional values between -1 and 1. The background is a fn of cx and cy.'''
    # c_x = c_x * 32 + 16
    # c_y = c_y * 32 + 16
    
    img_size = (32, 32)
    
    # let's choose to vary the c_x, c_y, and foreground of the circle
    c_min = 10
    c_max = 32-10
    
    # cx = (c_x-16)/32
    # cy = (c_y-16)/32
    # background = (cx**2 + cy**2) * 30
    cx = (c_x - c_min)/(c_max-c_min)
    cy = (c_y - c_min)/(c_max-c_min)
    # print('c_x', c_x)
    # print('c_y', c_y)
    # print('cx', cx)
    # print('cy', cy)
    background = np.sqrt(cx**2 + cy**2) / np.sqrt(2)
    # print(background)
    
    assert c_x >= c_min and c_x <= c_max
    assert c_y >= c_min and c_y <= c_max
    assert background >= 0 and background <= 1
    
    disk = make_disk_vectorized(
        img_size,
        outer_radius,
        transition_width,
        foreground=foreground,
        background=background,
        c_x=c_x,
        c_y=c_y,
    )
    
    return disk, c_x, c_y, background

def create_one_circle_fractional_2(c_x, c_y, foreground=1, outer_radius=10, transition_width=2):
    '''both cx and cy are fractional values between 0 and 1. The background is a fn of cx and cy.'''
    img_size = (32, 32)
    
    # let's choose to vary the c_x, c_y, and foreground of the circle
    c_min = 10
    c_max = 32-10
    
    # cx = (c_x)/32
    # cy = (c_y)/32
    cx = (c_x - c_min) / (c_max - c_min) * 2 - 1
    cy = (c_y - c_min) / (c_max - c_min) * 2 - 1
    background = np.sqrt((cx**2 + cy**2)) / np.sqrt(2)
    # print(background)
    
    assert c_x >= c_min and c_x <= c_max
    assert c_y >= c_min and c_y <= c_max
    assert background >= -1 and background <= 1

    disk = make_disk_vectorized(
        img_size,
        outer_radius,
        transition_width,
        foreground=foreground,
        background=background,
        c_x=c_x,
        c_y=c_y,
    )
    
    return disk, c_x, c_y, background


# using Dataset
class CustomTensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor):
        self.tensor = tensor
    
    def __getitem__(self, index):
        return self.tensor[index]
    
    def __len__(self):
        return self.tensor.shape[0]


# ---------------------------------------------------------------------------- #
#                       create hierarchical disk dataset                       #
# ---------------------------------------------------------------------------- #

def make_two_disks_img(
    img_size: Union[int, Tuple[int, int], torch.Size],
    outer_radius: Optional[float] = None,
    transition_width: Optional[float] = None,
    d: Optional[float] = None,    # center-to-center distance
    id1: Optional[float] = None,  # intensity disk 1
    id2: Optional[float] = None,  # intensity disk 2
    ib: Optional[float] = None,   # background intensity
    cx: Optional[float] = None,  # midpoint x
    cy: Optional[float] = None,  # midpoint y
    orientation: Optional[float] = None,  # radians
) -> torch.Tensor:
    # Handle image size input
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    assert len(img_size) == 2

    assert d > 2 * outer_radius, "circles must not overlap. \n" \
    "make d larger than twice the radius of a disk."

    # Defaults
    if outer_radius is None:
        outer_radius = (min(img_size) - 1) / 2
    if transition_width is None:
        transition_width = outer_radius / 2
    inner_radius = outer_radius - transition_width
    if d is None:
        d = 2 * outer_radius + 2  # just separated by their diameters + margin
    if orientation is None:
        orientation = 0.0
    if cx is None:
        cx = img_size[1] / 2
    if cy is None:
        cy = img_size[0] / 2
    if ib is None:
        ib = 0.0
    if id1 is None:
        id1 = 1.0
    if id2 is None:
        id2 = 1.0

    # Compute centers of two disks
    dx = (d / 2) * np.sin(orientation)
    dy = (d / 2) * np.cos(orientation)
    c1_x, c1_y = cx - dx, cy - dy
    c2_x, c2_y = cx + dx, cy + dy

    # Meshgrid for pixel coordinates
    y, x = np.ogrid[:img_size[0], :img_size[1]]

    # Radial distances for each disk
    r1 = np.sqrt((y - c1_y) ** 2 + (x - c1_x) ** 2)
    r2 = np.sqrt((y - c2_y) ** 2 + (x - c2_x) ** 2)

    # Initialize image with background
    img = np.full(img_size, ib, dtype=np.float32)

    # --- Disk 1 mask ---
    mask1 = np.zeros(img_size, dtype=np.float32)
    mask1[r1 < inner_radius] = id1
    mask1[r1 > outer_radius] = ib
    in_band1 = (r1 >= inner_radius) & (r1 <= outer_radius)
    radial_decay1 = (r1 - inner_radius) / (outer_radius - inner_radius)
    mask1[in_band1] = ib + (id1 - ib) * (1 + np.cos(np.pi * radial_decay1[in_band1])) / 2

    # --- Disk 2 mask ---
    mask2 = np.zeros(img_size, dtype=np.float32)
    mask2[r2 < inner_radius] = id2
    mask2[r2 > outer_radius] = ib
    in_band2 = (r2 >= inner_radius) & (r2 <= outer_radius)
    radial_decay2 = (r2 - inner_radius) / (outer_radius - inner_radius)
    mask2[in_band2] = ib + (id2 - ib) * (1 + np.cos(np.pi * radial_decay2[in_band2])) / 2

    # Composite disks over background (disk 2 overwrites disk 1 in overlap)
    img = np.where(mask1 != ib, mask1, img)
    img = np.where(mask2 != ib, mask2, img)

    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)


def create_two_disk_img_using_global_vars(
        img_size, outer_radius, transition_width, d, 
        theta, delta_id, ib, cx, cy):
    # state the dependency of id on orientation
    avg_id = 0.5 + 0.3 * np.sin(theta)
    id1 = avg_id - delta_id
    id2 = avg_id + delta_id

    img = make_two_disks_img(
        img_size, outer_radius, transition_width, d,
        id1, id2, ib,
        cx, cy, theta)
    return img, avg_id, id1, id2


def random_two_disk_dataset(
        delta_id:float=.2,
        d:float=20, outer_radius:float=8, transition_width:float=2,
        img_size:int=64, num_imgs:int=1e3,
    ):
    '''The background is a fn of cx and cy.
    '''
    assert d >= 0 and d <= 32
    assert d > outer_radius * 2, 'd should be at least 2*outer_radius'
    assert transition_width >= 0, 'transition_width should be non-negative'
    assert outer_radius > 0, 'outer_radius should be positive'
    assert img_size > d, 'img_size should be greater than d'

    c_min = d/2 + outer_radius
    c_max = img_size - (d/2 + outer_radius)

    dataset = torch.empty((num_imgs, 1, img_size, img_size), dtype=torch.float32)
    for i in range(num_imgs):
        # ------------------------------ global factors ------------------------------ #
        # uniform sampling of cluster center position
        cx = c_min + np.random.rand() * (c_max - c_min)
        cy = c_min + np.random.rand() * (c_max - c_min)

        # uniform sampling of background intensity
        ib = np.random.rand()
        assert ib >= 0 and ib <= 1

        # uniform sampling of orientation between 0 and 2pi
        theta = np.random.rand() * 2 * np.pi

        # ------------------------------- local factors ------------------------------ #
        img, avg_id, id1, id2 = create_two_disk_img_using_global_vars(
            img_size, outer_radius, transition_width, d,
            theta, delta_id, ib, cx, cy)
        dataset[i] = img

    return dataset