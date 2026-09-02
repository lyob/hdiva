import numpy as np
import torch
from typing import Optional, Tuple, Union


# ---------------------------------------------------------------------------- #
#                       create hierarchical disk dataset                       #
# ---------------------------------------------------------------------------- #

def make_two_disks_img(
    img_size: Union[int, Tuple[int, int], torch.Size],
    outer_radius: float,
    transition_width: float,
    id1: float,  # intensity disk 1
    id2: float,  # intensity disk 2
    ib: float,   # background intensity
    cx_1: float, 
    cy_1: float,
    cx_2: float,
    cy_2: float, 
) -> torch.Tensor:
    # Handle image size input
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    assert len(img_size) == 2

    inner_radius = outer_radius - transition_width

    # Meshgrid for pixel coordinates
    y, x = np.ogrid[:img_size[0], :img_size[1]]

    # Radial distances for each disk
    r1 = np.sqrt((y - cy_1) ** 2 + (x - cx_1) ** 2)
    r2 = np.sqrt((y - cy_2) ** 2 + (x - cx_2) ** 2)

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


def convert_to_pixels(val:float, min_pixel:float, max_pixel:float):
    return min_pixel + (max_pixel - min_pixel) * val

def convert_to_01(val:float, min_pixel:float, max_pixel:float):
    return (val - min_pixel) / (max_pixel - min_pixel)

def define_local_vars(
        img_size, outer_radius, transition_width, d,
        theta, ib, cx_b, cy_b):
    '''do the computations in units of pixels'''

    cx_1 = cx_b - (d / 2) * np.sin(theta)
    cy_1 = cy_b - (d / 2) * np.cos(theta)
    cx_2 = cx_b + (d / 2) * np.sin(theta)
    cy_2 = cy_b + (d / 2) * np.cos(theta)

    cy_1_01 = convert_to_01(cy_1, outer_radius, img_size - outer_radius)
    cy_2_01 = convert_to_01(cy_2, outer_radius, img_size - outer_radius)

    cx_b_01 = convert_to_01(cx_b, outer_radius, img_size - outer_radius)
    cx_b_01 = convert_to_01(cx_b, outer_radius, img_size - outer_radius)

    id_1 = (1 - cy_1_01) * (1-np.abs(cx_b_01 - 0.5)**1)
    id_2 = (1 - cy_2_01) * (1-np.abs(cx_b_01 - 0.5)**1)

    img = make_two_disks_img(
        img_size, outer_radius, transition_width,
        id_1, id_2, ib,
        cx_1, cy_1, cx_2, cy_2)

    return img, float(cx_1), float(cy_1), float(cx_2), float(cy_2), float(id_1), float(id_2)


def random_two_disk_dataset(
        d:float=20, outer_radius:float=8, transition_width:float=2,
        img_size:int=64, num_imgs:int=1e3,
        cx:Optional[float]=None, cy:Optional[float]=None, ib:Optional[float]=None, theta:Optional[float]=None,
    ):
    '''The background is a fn of cx and cy.
    '''
    assert d >= 0 and d <= 32
    assert d > outer_radius * 2, 'd should be at least 2*outer_radius'
    assert transition_width >= 0, 'transition_width should be non-negative'
    assert outer_radius > 0, 'outer_radius should be positive'
    assert img_size > d, 'img_size should be greater than d'

    cx_init = cx
    cy_init = cy
    ib_init = ib
    theta_init = theta

    num_imgs = int(num_imgs)
    dataset = torch.empty((num_imgs, 1, img_size, img_size), dtype=torch.float32)
    cx_bs = torch.empty((num_imgs,), dtype=torch.float32)
    cy_bs = torch.empty((num_imgs,), dtype=torch.float32)
    thetas = torch.empty((num_imgs,), dtype=torch.float32)

    # make sure that the circles are fully within the image
    c_b_min = d/2 + outer_radius
    c_b_max = img_size - (d/2 + outer_radius)


    for i in range(num_imgs):
        # ------------------------------ global factors ------------------------------ #
        # uniform sampling of cluster center position
        cx_b = np.random.rand() if cx_init is None else cx_init
        cy_b = np.random.rand() if cy_init is None else cy_init

        # uniform sampling of background intensity
        ib = np.random.rand() if ib_init is None else ib_init
        assert ib >= 0 and ib <= 1

        # uniform sampling of orientation between 0 and 2pi
        if theta_init is None:
            theta = np.random.rand() * 2 * np.pi
        else:
            theta = theta_init

        # ------------------------------- local factors ------------------------------ #
        # # convert cluster coordinates into pixels
        cx_p_b = convert_to_pixels(cx_b, c_b_min, c_b_max)
        cy_p_b = convert_to_pixels(cy_b, c_b_min, c_b_max)

        img, cx_1, cy_1, cx_2, cy_2, id_1, id_2 = define_local_vars(
            img_size, outer_radius, transition_width, d,
            theta, ib, cx_p_b, cy_p_b)
        dataset[i] = img
        cx_bs[i] = cx_b
        cy_bs[i] = cy_b
        thetas[i] = theta

    return dataset, cx_bs, cy_bs, thetas





# ---------------------------------------------------------------------------- #
#                       hierarchical disk dataset number 3                     #
# ---------------------------------------------------------------------------- #

def make_two_disks_img_3(
    img_size: Union[int, Tuple[int, int], torch.Size],
    outer_radius: float,
    transition_width: float,
    id1: float,  # intensity disk 1
    id2: float,  # intensity disk 2
    ib: float,   # background intensity
    cx_1: float, 
    cy_1: float,
    cx_2: float,
    cy_2: float, 
) -> torch.Tensor:
    # Handle image size input
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    assert len(img_size) == 2

    inner_radius = outer_radius - transition_width

    # Meshgrid for pixel coordinates
    y, x = np.ogrid[:img_size[0], :img_size[1]]

    # Radial distances for each disk
    r1 = np.sqrt((y - cy_1) ** 2 + (x - cx_1) ** 2)
    r2 = np.sqrt((y - cy_2) ** 2 + (x - cx_2) ** 2)

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

def define_local_vars_3(
        img_size, outer_radius, transition_width, d,
        theta, ib, cx_b, cy_b, delta_id:float=0.2):
    '''do the computations in units of pixels'''

    # individual disk positions
    cx_1 = cx_b - (d / 2) * np.sin(theta)
    cy_1 = cy_b - (d / 2) * np.cos(theta)
    cx_2 = cx_b + (d / 2) * np.sin(theta)
    cy_2 = cy_b + (d / 2) * np.cos(theta)

    cy_1_01 = convert_to_01(cy_1, outer_radius, img_size - outer_radius)
    cy_2_01 = convert_to_01(cy_2, outer_radius, img_size - outer_radius)

    cx_b_01 = convert_to_01(cx_b, outer_radius, img_size - outer_radius)
    cx_b_01 = convert_to_01(cx_b, outer_radius, img_size - outer_radius)

    # individual disk intensities
    rescale = lambda x: (((x * 2 - 1) * .6) + 1) * .5  # scale to range 0.2 to 0.8
    ib_scaled = rescale(ib)
    id_1 = (1 - ib_scaled) + delta_id * np.cos(theta)
    id_2 = (1 - ib_scaled) - delta_id * np.cos(theta)
    # id_1 = (1 - cy_1_01) * (1-np.abs(cx_b_01 - 0.5)**1)
    # id_2 = (1 - cy_2_01) * (1-np.abs(cx_b_01 - 0.5)**1)

    img = make_two_disks_img(
        img_size, outer_radius, transition_width,
        id_1, id_2, ib,
        cx_1, cy_1, cx_2, cy_2)

    return img, float(cx_1), float(cy_1), float(cx_2), float(cy_2), float(id_1), float(id_2)


def random_two_disk_dataset_3(
        d:float=20, outer_radius:float=8, transition_width:float=2, delta_id:float=0.2,
        img_size:int=64, num_imgs:int=1e3,
        cx:Optional[float]=None, cy:Optional[float]=None, ib:Optional[float]=None, theta:Optional[float]=None,
    ):
    '''The background is a fn of cx and cy.
    '''
    assert d >= 0 and d <= 32
    assert d > outer_radius * 2, 'd should be at least 2*outer_radius'
    assert transition_width >= 0, 'transition_width should be non-negative'
    assert outer_radius > 0, 'outer_radius should be positive'
    assert img_size > d, 'img_size should be greater than d'

    cx_init = cx
    cy_init = cy
    ib_init = ib
    theta_init = theta

    num_imgs = int(num_imgs)
    dataset = torch.empty((num_imgs, 1, img_size, img_size), dtype=torch.float32)
    cx_bs = torch.empty((num_imgs,), dtype=torch.float32)
    cy_bs = torch.empty((num_imgs,), dtype=torch.float32)
    thetas = torch.empty((num_imgs,), dtype=torch.float32)

    # make sure that the circles are fully within the image
    c_b_min = d/2 + outer_radius
    c_b_max = img_size - (d/2 + outer_radius)


    for i in range(num_imgs):
        # ------------------------------ global factors ------------------------------ #
        # uniform sampling of cluster center position
        cx_b = np.random.rand() if cx_init is None else cx_init
        cy_b = np.random.rand() if cy_init is None else cy_init

        # uniform sampling of background intensity
        ib = np.random.rand() if ib_init is None else ib_init
        assert ib >= 0 and ib <= 1

        # uniform sampling of orientation between 0 and 2pi
        if theta_init is None:
            theta = np.random.rand() * 2 * np.pi
        else:
            theta = theta_init

        # ------------------------------- local factors ------------------------------ #
        # # convert cluster coordinates into pixels
        cx_p_b = convert_to_pixels(cx_b, c_b_min, c_b_max)
        cy_p_b = convert_to_pixels(cy_b, c_b_min, c_b_max)

        img, cx_1, cy_1, cx_2, cy_2, id_1, id_2 = define_local_vars_3(
            img_size, outer_radius, transition_width, d,
            theta, ib, cx_p_b, cy_p_b, delta_id)
        dataset[i] = img
        cx_bs[i] = cx_b
        cy_bs[i] = cy_b
        thetas[i] = theta

    return dataset, cx_bs, cy_bs, thetas