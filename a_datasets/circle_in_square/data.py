import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def make_square(
    img_size,
    foreground,
    background,
    square_x,
    square_y,
    square_width,
) -> Tensor:
    """
    Generate a single square image.
    Args:
        img_size (int or tuple): Size of the image.
        foreground (float): Foreground intensity.
        background (float): Background intensity.
        square_x (float): x-coordinate of the center of the square.
        square_y (float): y-coordinate of the center of the square.
        square_width (float): Width of the square.

    Returns:
        Tensor: The generated square image.
    """
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    assert len(img_size) == 2

    # Create a meshgrid for the x, y coordinates
    y, x = np.ogrid[: img_size[0], : img_size[1]]

    # Calculate the Chebyshev distance (L∞ norm) for each pixel from the center
    # This creates a square shape instead of a circle
    r = np.maximum(np.abs(y - square_y), np.abs(x - square_x))

    # Initialize the mask as zeros
    mask = np.zeros(img_size)

    # # Apply the conditions to the mask
    mask[r < square_width] = foreground  # Inside the square, set to 1
    mask[r > square_width] = background  # Outside the square, set to 0

    return torch.tensor(mask, dtype=torch.float).unsqueeze(0)


def make_square_with_circle(
    img_size,
    square_foreground,
    circle_foreground,
    background,
    square_x,
    square_y,
    square_width,
    circle_radius,
    circle_transition_width=2,
    circle_x=0.5,
    circle_y=0.5,
    square_antialias=False,
) -> Tensor:
    """
    Generate an image with a square containing a circle fully inside it.

    Args:
        img_size (int or tuple): Size of the image.
        square_foreground (float): Foreground intensity for the square.
        circle_foreground (float): Foreground intensity for the circle.
        background (float): Background intensity.
        square_x (float): Normalized x-coordinate (0 to 1) of the center of the square.
        square_y (float): Normalized y-coordinate (0 to 1) of the center of the square.
        square_width (float): Half-width of the square (distance from center to edge).
        circle_radius (float): Outer radius of the circle.
        circle_transition_width (float): Width of the soft edge transition for circle (default 0 = hard edge).
        circle_x (float): Relative x-position of circle within square (0 to 1, 0.5 = centered).
        circle_y (float): Relative y-position of circle within square (0 to 1, 0.5 = centered).
        square_antialias (bool): If True, applies a 1-pixel soft edge transition for the square (default False).

    Returns:
        Tensor: The generated image with shape (1, H, W).
    """
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    assert len(img_size) == 2

    # Validate normalized coordinates
    assert 0 <= square_x <= 1, "square_x must be between 0 and 1"
    assert 0 <= square_y <= 1, "square_y must be between 0 and 1"

    # Convert normalized coordinates to pixel coordinates
    # Map 0-1 to the valid range where the square stays fully in bounds
    # At 0: square center is at square_width (left/top edge at 0)
    # At 1: square center is at img_size - square_width (right/bottom edge at img_size)
    min_x = square_width
    max_x = img_size[1] - square_width
    min_y = square_width
    max_y = img_size[0] - square_width
    square_x_px = min_x + square_x * (max_x - min_x)  # x corresponds to width (dim 1)
    square_y_px = min_y + square_y * (max_y - min_y)  # y corresponds to height (dim 0)

    # Ensure circle fits inside the square
    assert circle_radius <= square_width, "Circle radius must be <= square_width to fit inside"
    assert 0 <= circle_x <= 1, "circle_x must be between 0 and 1"
    assert 0 <= circle_y <= 1, "circle_y must be between 0 and 1"

    # Calculate the maximum offset the circle can have from square center
    # while still staying fully inside the square
    max_offset = square_width - circle_radius

    # Convert relative position (0-1) to actual offset (-max_offset to +max_offset)
    circle_offset_x = (circle_x - 0.5) * 2 * max_offset
    circle_offset_y = (circle_y - 0.5) * 2 * max_offset

    # Calculate circle center
    circle_x = square_x_px + circle_offset_x
    circle_y = square_y_px + circle_offset_y

    # Create a meshgrid for the x, y coordinates
    y, x = np.ogrid[: img_size[0], : img_size[1]]

    # Calculate distances from the center
    # Chebyshev distance (L∞ norm) for square
    chebyshev_dist = np.maximum(np.abs(y - square_y_px), np.abs(x - square_x_px))
    # Euclidean distance for circle (from circle center)
    radial_dist = np.sqrt((y - circle_y) ** 2 + (x - circle_x) ** 2)

    # Initialize the mask with background
    mask = np.full(img_size, background, dtype=np.float64)

    # Apply the square
    if square_antialias:
        # Anti-aliased square with 1-pixel soft edge transition
        inner_width = square_width - 1
        # Inside the inner boundary: full square foreground
        mask[chebyshev_dist <= inner_width] = square_foreground
        # Transition zone: smooth interpolation between background and foreground
        transition_mask = (chebyshev_dist > inner_width) & (chebyshev_dist <= square_width)
        edge_decay = chebyshev_dist - inner_width  # 0 to 1 in the transition zone
        mask[transition_mask] = (
            background + (square_foreground - background) * (1 + np.cos(np.pi * edge_decay[transition_mask])) / 2
        )
    else:
        # Hard edge
        mask[chebyshev_dist <= square_width] = square_foreground

    # Apply the circle (on top of the square)
    inner_radius = circle_radius - circle_transition_width

    # Inside the inner radius: full circle foreground
    mask[radial_dist < inner_radius] = circle_foreground

    if circle_transition_width > 0:
        # Calculate the radial decay for pixels between inner and outer radius
        radial_decay = (radial_dist - inner_radius) / circle_transition_width

        # Apply the radial decay with the cosine function for smooth transition
        transition_mask = (radial_dist >= inner_radius) & (radial_dist <= circle_radius)
        mask[transition_mask] = (
            square_foreground
            + (circle_foreground - square_foreground) * (1 + np.cos(np.pi * radial_decay[transition_mask])) / 2
        )

    return torch.tensor(mask, dtype=torch.float).unsqueeze(0)


class CircleInSquareDataset(Dataset):
    def __init__(
        self,
        num_images: int,
        img_size: int = 64,
        circle_radius: float = 6,
        square_width: float = 12,
        circle_transition_width: float = 2,
        square_antialias: bool = True,
        square_foreground: float = 0.5,
        circle_foreground: float = 1.0,
        background: float = 0.0,
        seed: None | int = None,
    ):
        """
        Create a dataset of square-with-circle images with uniformly distributed positions.

        The square position is uniformly distributed such that it never goes out of bounds.
        The circle position within the square is also uniformly distributed.

        Args:
            num_images (int): Number of images to generate.
            img_size (int): Size of each image (assumes square images).
            circle_radius (float): Radius of the circle.
            square_width (float): Half-width of the square (distance from center to edge).
            circle_transition_width (float): Width of the soft edge transition for circle.
            square_antialias (bool): If True, applies a 1-pixel soft edge transition for the square.
            square_foreground (float): Foreground intensity for the square.
            circle_foreground (float): Foreground intensity for the circle.
            background (float): Background intensity.
            seed (int, optional): Random seed for reproducibility. If None, no seed is set.

        Returns:
            tuple: A tuple containing:
                - dataset (Tensor): The generated images with shape (num_images, 1, img_size, img_size).
                - square_x_positions (Tensor): Normalized x-coordinates of square centers (0 to 1).
                - square_y_positions (Tensor): Normalized y-coordinates of square centers (0 to 1).
                - circle_x_positions (Tensor): Relative x-positions of circles within squares (0 to 1).
                - circle_y_positions (Tensor): Relative y-positions of circles within squares (0 to 1).
        """
        self.num_images = int(num_images)
        self.img_size = img_size
        self.circle_radius = circle_radius
        self.square_width = square_width
        self.circle_transition_width = circle_transition_width
        self.square_antialias = square_antialias
        self.square_foreground = square_foreground
        self.circle_foreground = circle_foreground
        self.background = background
        self.seed = seed
        self.data = torch.empty((num_images, 1, img_size, img_size), dtype=torch.float32)
        # self.targets = []
        self.square_targets = torch.empty((num_images, 2), dtype=torch.float32)
        self.circle_targets = torch.empty((num_images, 2), dtype=torch.float32)
        if seed is not None:
            np.random.seed(seed)

        assert circle_radius <= square_width, "Circle radius must be <= square_width to fit inside"

        # Sample all random positions upfront (vectorized sampling)
        # Square positions: uniformly distributed in normalized coordinates (0 to 1)
        square_x_positions = np.random.rand(num_images).astype(np.float32)
        square_y_positions = np.random.rand(num_images).astype(np.float32)

        # Circle positions within squares: uniformly distributed (0 to 1)
        circle_x_positions = np.random.rand(num_images).astype(np.float32)
        circle_y_positions = np.random.rand(num_images).astype(np.float32)

        # Generate each image
        for i in range(num_images):
            img = make_square_with_circle(
                img_size=img_size,
                square_foreground=square_foreground,
                circle_foreground=circle_foreground,
                background=background,
                square_x=square_x_positions[i],
                square_y=square_y_positions[i],
                square_width=square_width,
                circle_radius=circle_radius,
                circle_transition_width=circle_transition_width,
                circle_x=circle_x_positions[i],
                circle_y=circle_y_positions[i],
                square_antialias=square_antialias,
            )
            self.data[i] = img
            self.square_targets[i] = torch.tensor([square_x_positions[i], square_y_positions[i]])
            self.circle_targets[i] = torch.tensor([circle_x_positions[i], circle_y_positions[i]])

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        return self.data[idx], (self.square_targets[idx], self.circle_targets[idx])

    # return (
    #     dataset,
    #     (torch.from_numpy(square_x_positions), torch.from_numpy(square_y_positions)),
    #     (torch.from_numpy(circle_x_positions), torch.from_numpy(circle_y_positions)),
    # )


def load_cis_data(
    num_images: int = 10000,
    img_size: int = 64,
    circle_radius: float = 6,
    square_width: float = 12,
    circle_transition_width: float = 2,
    square_antialias: bool = False,
    square_foreground: float = 0.5,
    circle_foreground: float = 1.0,
    background: float = 0.0,
    seed: int = 42,
):
    """Load bump data by generating it from GaussianBumpDataset.

    Args:
        num_samples: Number of samples to generate.
        grid_size: Size of the grid (image dimensions).
        sigma: Standard deviation for the Gaussian bump.
        seed: Random seed for reproducibility.
        holdout_center: If True, exclude samples from the center region.
        only_sample_holdout: If True, only sample from the center region.

    Returns:
        Tensor of shape (num_samples, 1, grid_size, grid_size) containing the bump images.
    """
    if img_size == 64:
        # use default params for circle_radius, square_width.
        pass
    elif img_size == 28:
        # scale down circle and square for smaller image size
        circle_radius = 4
        square_width = 6

    dataset = CircleInSquareDataset(
        num_images=num_images,
        img_size=img_size,
        circle_radius=circle_radius,
        square_width=square_width,
        circle_transition_width=circle_transition_width,
        square_antialias=square_antialias,
        square_foreground=square_foreground,
        circle_foreground=circle_foreground,
        background=background,
        seed=seed,
    )

    cpu_tensor = dataset.data
    square_targets, circle_targets = dataset.square_targets, dataset.circle_targets

    return cpu_tensor, (square_targets, circle_targets)


def extract_square_position(
    image: Tensor,
    square_width: float,
    square_foreground: float,
    background: float,
    circle_foreground: float = None,
) -> tuple[Tensor, Tensor]:
    """
    Extract normalized (x, y) position of square center from image(s).

    Uses distance-based classification + centroid method: identifies pixels closer
    to square_foreground than to background/circle, computes the centroid, and
    converts back to normalized coordinates.

    Works with any ordering of intensity values (e.g., dark square on light background,
    bright circle, etc.).

    Args:
        image (Tensor): Image tensor of shape (1, H, W) or (B, 1, H, W).
        square_width (float): Known square half-width (distance from center to edge).
        square_foreground (float): Known foreground intensity of the square.
        background (float): Known background intensity.
        circle_foreground (float, optional): If provided, excludes circle pixels
            by classifying based on distance to all three values.

    Returns:
        tuple[Tensor, Tensor]: (square_x, square_y) normalized coordinates (0 to 1).
            For batch input, returns tensors of shape (B,).
            For single image, returns scalar tensors.
    """
    # Handle both single image (1, H, W) and batch (B, 1, H, W)
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    B, C, H, W = image.shape
    assert C == 1, "Expected single-channel images"

    pixels = image[:, 0]  # Shape: (B, H, W)

    # Compute distance to each intensity value
    dist_to_square = torch.abs(pixels - square_foreground)
    dist_to_background = torch.abs(pixels - background)

    # A pixel belongs to square if it's closer to square_foreground than to background
    mask = dist_to_square < dist_to_background

    # If circle_foreground is provided, also exclude pixels closer to circle
    if circle_foreground is not None:
        dist_to_circle = torch.abs(pixels - circle_foreground)
        mask = mask & (dist_to_square < dist_to_circle)

    # Create coordinate grids
    y_coords = torch.arange(H, dtype=image.dtype, device=image.device)
    x_coords = torch.arange(W, dtype=image.dtype, device=image.device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")  # Shape: (H, W)

    # Compute centroid for each image in batch
    mask_float = mask.float()  # Shape: (B, H, W)
    total_pixels = mask_float.sum(dim=(1, 2))  # Shape: (B,)

    # Weighted sum of coordinates
    centroid_y_px = (mask_float * yy).sum(dim=(1, 2)) / total_pixels  # Shape: (B,)
    centroid_x_px = (mask_float * xx).sum(dim=(1, 2)) / total_pixels  # Shape: (B,)

    # Convert pixel coordinates to normalized (0 to 1)
    # Inverse of: square_x_px = min_x + square_x * (max_x - min_x)
    # where min_x = square_width, max_x = img_size - square_width
    min_coord = square_width
    max_coord_x = W - square_width
    max_coord_y = H - square_width

    square_x = (centroid_x_px - min_coord) / (max_coord_x - min_coord)
    square_y = (centroid_y_px - min_coord) / (max_coord_y - min_coord)

    # Clamp to [0, 1] to handle edge cases
    square_x = torch.clamp(square_x, 0.0, 1.0)
    square_y = torch.clamp(square_y, 0.0, 1.0)

    if squeeze_output:
        square_x = square_x.squeeze(0)
        square_y = square_y.squeeze(0)

    return square_x, square_y
