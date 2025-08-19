import numpy as np
import torch
from typing import Optional, Tuple, Union


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
        theta, delta_id, ib, cx_01, cy_01):
    # state the dependency of id on orientation

    # make sure that the circles are fully within the image
    c_min = d/2 + outer_radius
    c_max = img_size - (d/2 + outer_radius)

    # convert coordinates into pixels
    cx = c_min + (c_max - c_min) * cx_01
    cy = c_min + (c_max - c_min) * cy_01

    # get avg intensity: if cy = c_max, get minimum intensity. if cy = c_min, get maximum intensity.
    min_avg_id = 1.1 * delta_id
    avg_id = min_avg_id + (1 - 2 * min_avg_id) * (c_max - cy) / (c_max - c_min)
    id1 = avg_id + np.cos(theta) * delta_id
    id2 = avg_id - np.cos(theta) * delta_id

    img = make_two_disks_img(
        img_size, outer_radius, transition_width, d,
        id1, id2, ib,
        cx, cy, theta)
    return img, avg_id, id1, id2


def random_two_disk_dataset(
        delta_id:float=.2,
        d:float=20, outer_radius:float=8, transition_width:float=2,
        img_size:int=64, num_imgs:int=1e3,
        cx_01:Optional[float]=None, cy_01:Optional[float]=None, ib:Optional[float]=None, theta:Optional[float]=None,
    ):
    '''The background is a fn of cx and cy.
    '''
    assert d >= 0 and d <= 32
    assert d > outer_radius * 2, 'd should be at least 2*outer_radius'
    assert transition_width >= 0, 'transition_width should be non-negative'
    assert outer_radius > 0, 'outer_radius should be positive'
    assert img_size > d, 'img_size should be greater than d'

    cx_01_init = cx_01
    cy_01_init = cy_01
    ib_init = ib
    theta_init = theta

    num_imgs = int(num_imgs)
    dataset = torch.empty((num_imgs, 1, img_size, img_size), dtype=torch.float32)
    avg_intensities = torch.empty((num_imgs,), dtype=torch.float32)
    cxs = torch.empty((num_imgs,), dtype=torch.float32)
    cys = torch.empty((num_imgs,), dtype=torch.float32)
    thetas = torch.empty((num_imgs,), dtype=torch.float32)

    for i in range(num_imgs):
        # ------------------------------ global factors ------------------------------ #
        # uniform sampling of cluster center position
        cx_01 = np.random.rand() if cx_01_init is None else cx_01_init
        cy_01 = np.random.rand() if cy_01_init is None else cy_01_init

        # uniform sampling of background intensity
        ib = np.random.rand() if ib_init is None else ib_init
        assert ib >= 0 and ib <= 1

        # uniform sampling of orientation between 0 and 2pi
        if theta_init is None:
            theta = np.random.rand() * 2 * np.pi
        else:
            theta = theta_init

        # ------------------------------- local factors ------------------------------ #
        img, avg_id, id1, id2 = create_two_disk_img_using_global_vars(
            img_size, outer_radius, transition_width, d,
            theta, delta_id, ib, cx_01, cy_01)
        dataset[i] = img
        avg_intensities[i] = avg_id
        cxs[i] = cx_01
        cys[i] = cy_01
        thetas[i] = theta

    return dataset, avg_intensities, cxs, cys, thetas