from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from a_datasets.disks.disks import make_disk_vectorized


def normalize_to_pm1(image: torch.Tensor) -> torch.Tensor:
    """Map an image from [0, 1] to [-1, 1] (standard DDPM input convention)."""
    return image * 2.0 - 1.0


def denormalize_from_pm1(image: torch.Tensor) -> torch.Tensor:
    """Inverse of `normalize_to_pm1`: map [-1, 1] back to [0, 1]."""
    return (image + 1.0) / 2.0


# Dataset Definition
class HierarchicalTwoDiskDistanceIntensityDataset(Dataset):
    """Two-disk hierarchy where a single top latent z = d is the COMMON CAUSE of
    both the relative vertical distance and the relative intensity of the disks.

    Generative model (per image, K = 2 disks):

        d        ~ N(0, sigma_d^2)                    # signed vertical offset (top latent z)

        # --- vertical position channel (d -> positions) ---
        eps_i    ~ N(0, sigma_eps^2),  i = 1, 2       # per-disk vertical jitter (iid)
        c_y^(1)  = m_y - d/2 + eps_1                  # left  disk height
        c_y^(2)  = m_y + d/2 + eps_2                  # right disk height

        # --- intensity channel (d -> intensities), the SAME d ---
        eta_i    ~ N(0, sigma_eta^2),  i = 1, 2       # per-disk intensity noise (iid)
        I_1      = mu_I + (gamma/2) * d + eta_1       # left  disk intensity
        I_2      = mu_I - (gamma/2) * d + eta_2       # right disk intensity

        c_x^(i)  ~ Uniform(half_i, margin)           # local horizontal position (free rider)

    Sign convention -- "the higher disk is lighter"
    -----------------------------------------------
    With the usual image convention (row 0 at the top, so a SMALLER c_y sits
    HIGHER on screen), d > 0 raises disk 1 (c_y^(1) = m_y - d/2) and brightens it
    (I_1 = mu_I + (gamma/2) d), while disk 2 drops and darkens -- and vice versa
    for d < 0. So whichever disk is higher is lighter, by construction (in
    expectation; realized flips from jitter are rare when gamma*sigma_d dominates
    the intensity noise). If your renderer's vertical axis points the other way
    (darker disk ends up on top), simply negate `intensity_gain`.

    d is now carried by TWO conditionally-independent channels given d
    -----------------------------------------------------------------
    Both readouts of d are *differences* (the sums carry no d):

        delta_c = c_y^(2) - c_y^(1) = d + (eps_2 - eps_1),      noise var 2*sigma_eps^2
        delta_I = I_2   - I_1       = -gamma*d + (eta_2 - eta_1), noise var 2*sigma_eta^2

    Each single disk still gives only a noisy estimate of d (its height, or its
    intensity, is d entangled with that disk's own jitter/noise); pooling the two
    halves sharpens both. Because the two channels are independent given d, they
    are *redundant* about d -- the clean setup for a PID analysis. The combined
    Gaussian posterior precision is additive across channels:

        p_c   = 1 / (2*sigma_eps^2)                  # position-channel precision
        p_I   = gamma^2 / (2*sigma_eta^2)            # intensity-channel precision
        E[d|.] = (p_c*delta_c + p_I*(-delta_I/gamma)) / (1/sigma_d^2 + p_c + p_I)
        I(z;x) = (1/2) * log( 1 + sigma_d^2 * (p_c + p_I) )      [nats].

    (Setting gamma = 0 recovers the position-only distance gauge:
     I(z;x) = (1/2) log(1 + sigma_d^2 / (2 sigma_eps^2)).)

    Design notes
    ------------
      * Intensity here is a genuinely INDEPENDENT channel about d (fork d -> I),
        not a deterministic function of realized height. If you instead want a
        hard guarantee that the realized-higher disk is always lighter, tie
        intensity to c_y^(i) directly (chain d -> c_y -> I); that makes intensity
        redundant *with position* rather than an independent readout of d.
      * Keep sigma_eta small relative to gamma*sigma_d so the "higher = lighter"
        ordering is visually reliable, but nonzero so a single patch cannot read
        d exactly off its own disk.
      * c_x^(i) are independent of d -> free riders (local rate, no top-level rate).

    Stored ground truth
    -------------------
      data_d        : [N]        gauge d (this is z)
      data_eps      : [N, 2]     vertical jitters (eps_1, eps_2)
      data_eta      : [N, 2]     intensity noises (eta_1, eta_2)
      data_I        : [N, 2]     intensities (I_1, I_2)
      data_cx       : [N, 2]     horizontal centers (free riders)
      data_cy       : [N, 2]     vertical centers
      data_pos      : [N, 2, 2]  raw pixel centers, (disk, [c_x, c_y])
      data_pos_norm : [N, 2, 2]  per-disk positions mapped to [-1, 1]
      targets       : [N, 1, H, W]  rendered images
    """

    def __init__(
        self,
        num_samples: int = 2000,
        grid_size: int = 28,
        mu_I: float = 0.6,
        background: float = 0.0,
        sigma_d: float = 4.0,
        sigma_eps: float = 1.5,
        intensity_gain: float = 0.04,
        sigma_eta: float = 0.04,
        I_min: float = 0.1,
        I_max: float = 1.0,
        outer_radius: float = 5.5,
        transition_width: int = 2,
        dataset_seed: int = 42,
        return_latents: bool = False,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        assert grid_size % 2 == 0, "left-right split needs an even width"
        assert grid_size // 2 > 2 * outer_radius + transition_width, "each half must fit a disk"
        assert 0.0 <= background <= 1.0
        assert 0.0 <= I_min < I_max <= 1.0
        assert I_min <= mu_I <= I_max
        assert sigma_d > 0 and sigma_eps > 0 and sigma_eta > 0

        self.num_samples = num_samples
        self.grid_size = grid_size
        self.mu_I = mu_I
        self.background = background
        self.sigma_d = sigma_d
        self.sigma_eps = sigma_eps
        self.gamma = intensity_gain
        self.sigma_eta = sigma_eta
        self.I_min = I_min
        self.I_max = I_max
        self.outer_radius = outer_radius
        self.transition_width = transition_width
        self.return_latents = return_latents
        self.transform = transform

        self.K = 2
        self.m_y = grid_size / 2.0  # fixed, known vertical midline

        # --- valid position windows -----------------------------------------
        half = grid_size // 2
        r = outer_radius
        tw = transition_width
        self.cx_range_left = (r, half - r - tw)
        self.cx_range_right = (half + r + tw, grid_size - r)
        self.cy_range = (r, grid_size - r)

        # --- analytic quantities implied by the gauge parameters ------------
        self.p_c = 1.0 / (2.0 * sigma_eps ** 2)                 # position-channel precision
        self.p_I = self.gamma ** 2 / (2.0 * sigma_eta ** 2)     # intensity-channel precision
        self.post_prec = 1.0 / sigma_d ** 2 + self.p_c + self.p_I
        self.gauge_mi_nats = 0.5 * np.log(1.0 + sigma_d ** 2 * (self.p_c + self.p_I))
        self.gauge_mi_bits = self.gauge_mi_nats / np.log(2.0)
        # per-channel MI contributions (for reference / PID bookkeeping)
        self.mi_position_nats = 0.5 * np.log(1.0 + sigma_d ** 2 * self.p_c)
        self.mi_intensity_nats = 0.5 * np.log(1.0 + sigma_d ** 2 * self.p_I)

        # --- storage --------------------------------------------------------
        nan = float("nan")
        self.data_d = torch.full((num_samples,), nan, dtype=torch.float32)
        self.data_eps = torch.full((num_samples, self.K), nan, dtype=torch.float32)
        self.data_eta = torch.full((num_samples, self.K), nan, dtype=torch.float32)
        self.data_I = torch.full((num_samples, self.K), nan, dtype=torch.float32)
        self.data_cx = torch.full((num_samples, self.K), nan, dtype=torch.float32)
        self.data_cy = torch.full((num_samples, self.K), nan, dtype=torch.float32)
        self.data_pos = torch.full((num_samples, self.K, 2), nan, dtype=torch.float32)
        self.data_pos_norm = torch.full((num_samples, self.K, 2), nan, dtype=torch.float32)
        self.targets = torch.full((num_samples, 1, grid_size, grid_size), nan, dtype=torch.float32)

        np.random.seed(dataset_seed)
        for i in range(num_samples):
            d, eps, eta, cy, intensities = self.sample_gauge()   # coupled height + intensity
            cx = self.sample_free_positions()                    # horizontal free riders

            positions = np.stack([cx, cy], axis=1)               # [K, 2] -> (disk, [c_x, c_y])
            img = self.render_image(positions, intensities)

            self.data_d[i] = d
            self.data_eps[i] = torch.from_numpy(eps).float()
            self.data_eta[i] = torch.from_numpy(eta).float()
            self.data_I[i] = torch.from_numpy(intensities).float()
            self.data_cx[i] = torch.from_numpy(cx).float()
            self.data_cy[i] = torch.from_numpy(cy).float()
            self.data_pos[i] = torch.from_numpy(positions).float()
            self.data_pos_norm[i] = torch.from_numpy(self._positions_to_norm(positions)).float()
            self.targets[i] = torch.from_numpy(img).float().unsqueeze(0)

    # ------------------------------------------------------------------ latents
    def sample_gauge(self) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Rejection-sample (d, eps, eta) so that both vertical centers land in
        the valid window AND both intensities land in [I_min, I_max]. The same d
        drives heights and intensities, so the two constraints are coupled."""
        cy_lo, cy_hi = self.cy_range
        while True:
            d = float(np.random.randn() * self.sigma_d)
            eps = np.random.randn(self.K) * self.sigma_eps
            eta = np.random.randn(self.K) * self.sigma_eta

            cy = np.array(
                [self.m_y - d / 2.0 + eps[0], self.m_y + d / 2.0 + eps[1]],
                dtype=np.float64,
            )
            intensities = np.array(
                [self.mu_I + (self.gamma / 2.0) * d + eta[0],
                 self.mu_I - (self.gamma / 2.0) * d + eta[1]],
                dtype=np.float64,
            )

            cy_ok = np.all(cy >= cy_lo) and np.all(cy <= cy_hi)
            I_ok = np.all(intensities >= self.I_min) and np.all(intensities <= self.I_max)
            if cy_ok and I_ok:
                return d, eps, eta, cy, intensities

    def sample_free_positions(self) -> np.ndarray:
        """Horizontal centers, one per half, independent of the gauge -> free riders."""
        cx1 = np.random.uniform(*self.cx_range_left)
        cx2 = np.random.uniform(*self.cx_range_right)
        return np.array([cx1, cx2], dtype=np.float64)

    def _positions_to_norm(self, positions: np.ndarray) -> np.ndarray:
        """Map each disk's pixel center to [-1, 1]; c_x within its own half window,
        c_y within the shared vertical window."""
        out = np.empty_like(positions)
        cy_lo, cy_hi = self.cy_range
        for i, (cx_lo, cx_hi) in enumerate((self.cx_range_left, self.cx_range_right)):
            out[i, 0] = 2.0 * (positions[i, 0] - cx_lo) / (cx_hi - cx_lo) - 1.0
            out[i, 1] = 2.0 * (positions[i, 1] - cy_lo) / (cy_hi - cy_lo) - 1.0
        return out

    # ------------------------------------------------------------------ render
    def render_image(self, positions: np.ndarray, intensities: np.ndarray) -> np.ndarray:
        """Render both disks at their coupled intensities on a fixed background.
        Each disk drawn with background 0 and summed; disjoint halves => no overlap."""
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
    def analytic_gauge_posterior(
        self,
        cy_pair: np.ndarray,
        I_pair: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Closed-form MMSE estimate of d, optionally pooling both channels.

        cy_pair : [..., 2] = (c_y^(1), c_y^(2))
        I_pair  : [..., 2] = (I_1, I_2), optional. If omitted, uses the position
                  channel alone (the pure distance gauge).
        """
        cy_pair = np.asarray(cy_pair, dtype=np.float64)
        delta_c = cy_pair[..., 1] - cy_pair[..., 0]         # ~ d + (eps_2 - eps_1)

        num = self.p_c * delta_c
        den = 1.0 / self.sigma_d ** 2 + self.p_c
        if I_pair is not None:
            I_pair = np.asarray(I_pair, dtype=np.float64)
            delta_I = I_pair[..., 1] - I_pair[..., 0]       # ~ -gamma d + (eta_2 - eta_1)
            d_from_I = -delta_I / self.gamma                # ~ d
            num = num + self.p_I * d_from_I
            den = den + self.p_I

        d_hat = num / den
        eps1_hat = (cy_pair[..., 0] - self.m_y) + d_hat / 2.0
        eps2_hat = (cy_pair[..., 1] - self.m_y) - d_hat / 2.0
        eps_hat = np.stack([eps1_hat, eps2_hat], axis=-1)

        out = {"d_hat": d_hat, "eps_hat": eps_hat, "delta_c": delta_c}
        if I_pair is not None:
            eta1_hat = (I_pair[..., 0] - self.mu_I) - (self.gamma / 2.0) * d_hat
            eta2_hat = (I_pair[..., 1] - self.mu_I) + (self.gamma / 2.0) * d_hat
            out["eta_hat"] = np.stack([eta1_hat, eta2_hat], axis=-1)
            out["delta_I"] = I_pair[..., 1] - I_pair[..., 0]
        return out

    # ------------------------------------------------------------------ dataset
    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        image = self.targets[idx]
        if self.transform is not None:
            image = self.transform(image)
        if not self.return_latents:
            return image
        latents = {
            "d": self.data_d[idx],
            "eps": self.data_eps[idx],
            "eta": self.data_eta[idx],
            "I": self.data_I[idx],
            "cx": self.data_cx[idx],
            "cy": self.data_cy[idx],
            "pos": self.data_pos[idx],
            "pos_norm": self.data_pos_norm[idx],
        }
        return image, latents


def load_dataset(
    num_samples: int = int(2e3),
    grid_size: int = 28,
    mu_I: float = 0.6,
    background: float = 0.0,
    sigma_d: float = 4.0,
    sigma_eps: float = 1.5,
    intensity_gain: float = 0.04,
    sigma_eta: float = 0.04,
    I_min: float = 0.1,
    I_max: float = 1.0,
    outer_radius: float = 5.5,
    transition_width: int = 2,
    dataset_seed: int = 42,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
):
    """Create the two-disk dataset where d is the common cause of both the
    vertical distance and the intensity contrast (higher disk lighter).

    Returns
    -------
    targets : [N, 1, H, W] tensor of images
    latents : dict with keys 'd' [N], 'eps' [N,2], 'eta' [N,2], 'I' [N,2],
              'cx' [N,2], 'cy' [N,2], 'pos' [N,2,2], 'pos_norm' [N,2,2]
    info    : dict with analytic quantities ('gauge_mi_nats', 'gauge_mi_bits',
              'mi_position_nats', 'mi_intensity_nats', 'p_c', 'p_I')
    """
    dataset = HierarchicalTwoDiskDistanceIntensityDataset(
        num_samples=num_samples,
        grid_size=grid_size,
        mu_I=mu_I,
        background=background,
        sigma_d=sigma_d,
        sigma_eps=sigma_eps,
        intensity_gain=intensity_gain,
        sigma_eta=sigma_eta,
        I_min=I_min,
        I_max=I_max,
        outer_radius=outer_radius,
        transition_width=transition_width,
        dataset_seed=dataset_seed,
        transform=transform,
    )

    latents = {
        "d": dataset.data_d,
        "eps": dataset.data_eps,
        "eta": dataset.data_eta,
        "I": dataset.data_I,
        "cx": dataset.data_cx,
        "cy": dataset.data_cy,
        "pos": dataset.data_pos,
        "pos_norm": dataset.data_pos_norm,
    }
    info = {
        "gauge_mi_nats": dataset.gauge_mi_nats,
        "gauge_mi_bits": dataset.gauge_mi_bits,
        "mi_position_nats": dataset.mi_position_nats,
        "mi_intensity_nats": dataset.mi_intensity_nats,
        "p_c": dataset.p_c,
        "p_I": dataset.p_I,
    }
    # dataset.targets is the raw [0, 1] store; apply the transform to the
    # returned tensor so load_dataset matches what __getitem__/the DataLoader yield.
    targets = dataset.targets
    if transform is not None:
        targets = transform(targets)
    return targets, latents, info