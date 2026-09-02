from typing import Dict, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from a_datasets.disks.disks import make_disk_vectorized


# Dataset Definition
class HierarchicalTwoDiskIntensityDataset(Dataset):
    """Two-disk hierarchy with a shared *intensity gauge* z and *local* positions.

    Generative model (per image, K = 2 disks):

        b      ~ N(0, sigma_b^2)                 # global intensity gauge  (top latent z)
        y_i    ~ N(0, sigma_y^2),   i = 1, 2     # per-disk intensity residuals (iid)
        I_i    = mu_I + b + y_i                  # disk i intensity = shared gauge + residual
        c^(i)  ~ Uniform(half_i, margin)         # local position, independent of b and y

    Disk 1 lives in the LEFT half of the image, disk 2 in the RIGHT half. The two
    spatial supports Omega_1, Omega_2 are disjoint, so

        x_{Omega_1} _||_ x_{Omega_2} | z   and   y_1 _||_ y_2 | z,

    and each half is an *independent measurement of the same gauge b*.

    Why z is required for local inference of I_i
    --------------------------------------------
    A convolutional encoder with local receptive fields over half i observes only
    I_i = mu_I + b + y_i and cannot split b from y_i (a 1-D gauge orbit
    (b, y_i) -> (b + d, y_i - d) leaves I_i invariant). Only pooling the two
    halves resolves the gauge. With Gaussian b, y_i the MMSE split is

        I_bar          = (I_1 + I_2) / 2
        E[b | I_1,I_2] = kappa * (I_bar - mu_I),   kappa = sigma_b^2 / (sigma_b^2 + sigma_y^2 / K)
        E[y_i | .]     = (I_i - mu_I) - E[b | .]

    so estimating y_i requires I_bar, the cross-disk average carried by z. Ablate
    the z -> y top-down pathway and the residual entropy the local branch cannot
    resolve should equal the gauge information

        I(z; x) = (1/2) * log( 1 + K * sigma_b^2 / sigma_y^2 )      [nats].

    Design constraints (so the gauge does not leak locally)
    -------------------------------------------------------
      * Background is fixed (default 0) and b is added ONLY to disk intensities,
        with no zero-reference disk -> b never appears in a single patch alone.
      * y_i is zero-mean -> the mean over disks recovers b (up to K-shrinkage).
      * Positions are independent of b and of each other -> they are 'free riders'
        that add local (per-disk) rate but leave I(z; x) exactly unchanged.

    Stored ground truth (for identifiability / rate probing)
    --------------------------------------------------------
      data_b        : [N]        gauge b (this is z)
      data_y        : [N, 2]     residuals (y_1, y_2)
      data_I        : [N, 2]     intensities (I_1, I_2)
      data_pos      : [N, 2, 2]  raw pixel centers, (disk, [c_x, c_y])
      data_pos_norm : [N, 2, 2]  per-disk positions mapped to [-1, 1]
      targets       : [N, 1, H, W]  rendered images
    """

    def __init__(
        self,
        num_samples: int = 2000,
        grid_size: int = 32,
        background: float = 0.0,
        mu_I: float = 0.5,
        sigma_b: float = 0.14,
        sigma_y: float = 0.07,
        I_min: float = 0.05,
        I_max: float = 1.0,
        outer_radius: int = 5,
        transition_width: int = 2,
        dataset_seed: int = 42,
        return_latents: bool = False,
    ):
        assert grid_size % 2 == 0, "left-right split needs an even width"
        assert grid_size // 2 > 2 * outer_radius + transition_width, "each half must fit a disk"
        assert 0.0 <= background <= 1.0
        assert 0.0 <= I_min < I_max <= 1.0
        assert sigma_b > 0 and sigma_y > 0
        assert I_min <= mu_I <= I_max

        self.num_samples = num_samples
        self.grid_size = grid_size
        self.background = background
        self.mu_I = mu_I
        self.sigma_b = sigma_b
        self.sigma_y = sigma_y
        self.I_min = I_min
        self.I_max = I_max
        self.outer_radius = outer_radius
        self.transition_width = transition_width
        self.return_latents = return_latents

        self.K = 2

        # --- valid position windows (disjoint, midline padded by transition_width) ---
        half = grid_size // 2
        r = outer_radius
        tw = transition_width
        # left disk: right edge padded so the soft transition never crosses the midline
        self.cx_range_left = (r, half - r - tw)
        # right disk: left edge padded symmetrically
        self.cx_range_right = (half + r + tw, grid_size - r)
        self.cy_range = (r, grid_size - r)

        # --- analytic quantities implied by the gauge parameters ---
        # MMSE shrinkage applied to (I_bar - mu_I) to recover b
        self.kappa = sigma_b ** 2 / (sigma_b ** 2 + sigma_y ** 2 / self.K)
        # gauge information I(z; x) in nats (Gaussian, sufficient statistic I_bar)
        self.gauge_mi_nats = 0.5 * np.log(1.0 + self.K * sigma_b ** 2 / sigma_y ** 2)
        self.gauge_mi_bits = self.gauge_mi_nats / np.log(2.0)

        # --- storage ---
        nan = float("nan")
        self.data_b = torch.full((num_samples,), nan, dtype=torch.float32)
        self.data_y = torch.full((num_samples, self.K), nan, dtype=torch.float32)
        self.data_I = torch.full((num_samples, self.K), nan, dtype=torch.float32)
        self.data_pos = torch.full((num_samples, self.K, 2), nan, dtype=torch.float32)
        self.data_pos_norm = torch.full((num_samples, self.K, 2), nan, dtype=torch.float32)
        self.targets = torch.full((num_samples, 1, grid_size, grid_size), nan, dtype=torch.float32)

        np.random.seed(dataset_seed)
        for i in range(num_samples):
            b, y, intensities = self.sample_gauge()
            positions = self.sample_positions()  # [K, 2] pixel centers, independent of b

            img = self.render_image(positions, intensities)

            self.data_b[i] = b
            self.data_y[i] = torch.from_numpy(y).float()
            self.data_I[i] = torch.from_numpy(intensities).float()
            self.data_pos[i] = torch.from_numpy(positions).float()
            self.data_pos_norm[i] = torch.from_numpy(self._positions_to_norm(positions)).float()
            self.targets[i] = torch.from_numpy(img).float().unsqueeze(0)

    # ------------------------------------------------------------------ latents
    def sample_gauge(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """Rejection-sample (b, y_1, y_2) so that both intensities land in
        [I_min, I_max]. Rejection keeps the clean additive gauge arithmetic
        rather than distorting it with a clamp; keep the sigmas small enough
        that the rejection rate stays low (see class docstring)."""
        while True:
            b = float(np.random.randn() * self.sigma_b)
            y = np.random.randn(self.K) * self.sigma_y
            intensities = self.mu_I + b + y
            if np.all(intensities >= self.I_min) and np.all(intensities <= self.I_max):
                return b, y, intensities

    def sample_positions(self) -> np.ndarray:
        """Sample the two disk centers uniformly within their own half, with a
        margin so blobs never cross the midline or the canvas edge. Positions
        are independent of the gauge and of each other -> free riders."""
        cx1 = np.random.uniform(*self.cx_range_left)
        cy1 = np.random.uniform(*self.cy_range)
        cx2 = np.random.uniform(*self.cx_range_right)
        cy2 = np.random.uniform(*self.cy_range)
        return np.array([[cx1, cy1], [cx2, cy2]], dtype=np.float64)

    def _positions_to_norm(self, positions: np.ndarray) -> np.ndarray:
        """Map each disk's pixel center to [-1, 1] within its own valid window."""
        out = np.empty_like(positions)
        for i, (cx_lo, cx_hi) in enumerate((self.cx_range_left, self.cx_range_right)):
            out[i, 0] = 2.0 * (positions[i, 0] - cx_lo) / (cx_hi - cx_lo) - 1.0
            out[i, 1] = 2.0 * (positions[i, 1] - self.cy_range[0]) / (self.cy_range[1] - self.cy_range[0]) - 1.0
        return out

    # ------------------------------------------------------------------ render
    def render_image(self, positions: np.ndarray, intensities: np.ndarray) -> np.ndarray:
        """Render both disks on a fixed background. Each disk is drawn with
        background 0 (so b never enters the background) and foreground = I_i;
        the disjoint halves let us simply sum the two canvases."""
        img = np.full((self.grid_size, self.grid_size), self.background, dtype=np.float64)
        for (cx, cy), I in zip(positions, intensities):
            disk = make_disk_vectorized(
                (self.grid_size, self.grid_size),
                self.outer_radius,
                self.transition_width,
                foreground=float(I),
                background=0.0,
                c_x=float(cx),
                c_y=float(cy),
            )
            img = img + disk.squeeze(0).numpy()
        return np.clip(img, 0.0, 1.0)

    # ------------------------------------------------------------- gauge helper
    def analytic_gauge_posterior(self, intensities: np.ndarray) -> Dict[str, np.ndarray]:
        """Closed-form MMSE split for a given pair of intensities, for probing /
        sanity-checking a learned encoder against the optimal gauge inference."""
        intensities = np.asarray(intensities, dtype=np.float64)
        I_bar = intensities.mean(axis=-1, keepdims=True)
        b_hat = self.kappa * (I_bar - self.mu_I)
        y_hat = (intensities - self.mu_I) - b_hat
        return {"b_hat": b_hat.squeeze(-1), "y_hat": y_hat, "I_bar": I_bar.squeeze(-1)}

    # ------------------------------------------------------------------ dataset
    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        image = self.targets[idx]
        if not self.return_latents:
            return image
        latents = {
            "b": self.data_b[idx],
            "y": self.data_y[idx],
            "I": self.data_I[idx],
            "pos": self.data_pos[idx],
            "pos_norm": self.data_pos_norm[idx],
        }
        return image, latents


def load_hierarchical_two_disk_intensity_dataset(
    num_samples: int = int(2e3),
    grid_size: int = 32,
    background: float = 0.0,
    mu_I: float = 0.5,
    sigma_b: float = 0.14,
    sigma_y: float = 0.07,
    I_min: float = 0.05,
    I_max: float = 1.0,
    outer_radius: int = 5,
    transition_width: int = 2,
    dataset_seed: int = 42,
):
    """Create the two-disk gauged-intensity dataset.

    Returns
    -------
    targets : [N, 1, H, W] tensor of images
    latents : dict with keys 'b' [N], 'y' [N,2], 'I' [N,2],
              'pos' [N,2,2], 'pos_norm' [N,2,2]
    info    : dict with analytic quantities ('kappa', 'gauge_mi_nats', 'gauge_mi_bits')
    """
    dataset = HierarchicalTwoDiskIntensityDataset(
        num_samples=num_samples,
        grid_size=grid_size,
        background=background,
        mu_I=mu_I,
        sigma_b=sigma_b,
        sigma_y=sigma_y,
        I_min=I_min,
        I_max=I_max,
        outer_radius=outer_radius,
        transition_width=transition_width,
        dataset_seed=dataset_seed,
    )

    latents = {
        "b": dataset.data_b,
        "y": dataset.data_y,
        "I": dataset.data_I,
        "pos": dataset.data_pos,
        "pos_norm": dataset.data_pos_norm,
    }
    info = {
        "kappa": dataset.kappa,
        "gauge_mi_nats": dataset.gauge_mi_nats,
        "gauge_mi_bits": dataset.gauge_mi_bits,
    }
    return dataset.targets, latents, info