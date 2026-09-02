"""
Flat (single-latent, non-hierarchical) wrapped-phase ring
=========================================================
Generative model:

    y      ~ vM(0, kappa)                                   # single latent (angle)
    x | y  ~ N( r0 [cos(m y), sin(m y)], sigma^2 I_2 )      # 2D circle, wrapped m times

Same y -> x map as the hierarchical version, but y is drawn directly from a von
Mises prior instead of from y|z, so there is no top-down z. With a single latent
the posterior p(y|x) is m-fold aliased and nothing resolves it -- which is the
point of this dataset: it tests whether a one-level model can represent that
multimodal posterior.

Closed forms (up to a y-independent constant; derivation: expand the Gaussian in
polar coords for x, drop y-independent terms):

    log p(x|y) = (r0 ||x|| / sigma^2) cos(m y - angle(x))   # likelihood, m peaks in y
    log p(y)   = kappa cos(y)                               # vM(0, kappa) prior
    log p(y|x) = log p(x|y) + log p(y)

Coverage note: y has angular std ~ kappa**-0.5, and the ring angle m*y wraps, so
small kappa -> a fully populated ring with all m aliases exercised; large kappa
-> a concentrated arc. Use small kappa to stress-test the aliasing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions import VonMises
from torch.utils.data import Dataset

PI = math.pi


def generate(
    n: int,
    m: int = 4,
    kappa: float = 1.0,
    r0: float = 2.0,
    sigma: float = 0.15,
    seed: int | None = 0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, Tensor]:
    """Sample (x, y) from the flat wrapped-phase ring.

    Returns x:(n,2), y:(n,) in radians on (-pi, pi].
    `seed` is applied via torch.manual_seed (VonMises.sample uses the global RNG).
    """
    if seed is not None:
        torch.manual_seed(seed)
    y = VonMises(torch.zeros(n), kappa * torch.ones(n)).sample()
    center = r0 * torch.stack([torch.cos(m * y), torch.sin(m * y)], dim=-1)
    x = center + sigma * torch.randn(n, 2)
    return x.to(device), y.to(device)


@dataclass
class WrappedPhaseRingFlat(Dataset):
    """Map-style dataset; samples precomputed in __init__.

    __getitem__ returns x:(2,) by default, or {"x","y"} when return_latents=True.
    Ground-truth y is always available as `.y`.
    """
    def __init__(self, n: int = 20_000, m: int = 4, kappa: float = 1.0, r0: float = 2.0, sigma: float = 0.15, seed: int | None = 0, return_latents: bool = False):
        super().__init__()
        self.n = n
        self.m = m
        self.kappa = kappa
        self.r0 = r0
        self.sigma = sigma
        self.seed = seed
        self.return_latents = return_latents

        x, y = generate(n, m, kappa, r0, sigma, seed)
        self.x, self.y = x.float(), y.float()

    # n: int = 20_000
    # m: int = 4
    # kappa: float = 1.0
    # r0: float = 2.0
    # sigma: float = 0.15
    # seed: int | None = 0
    # return_latents: bool = False

    # def __post_init__(self) -> None:
    #     super().__init__()
    #     x, y = generate(self.n, self.m, self.kappa, self.r0, self.sigma, self.seed)
    #     self.x, self.y = x.float(), y.float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        if self.return_latents:
            return {"x": self.x[i], "y": self.y[i]}
        return self.x[i]

    # ----- closed-form objects (grids over y in (-pi, pi]) ------------------- #
    def _grid(self, n_grid: int, device) -> Tensor:
        return torch.linspace(-PI, PI, n_grid, device=device)

    def log_likelihood_y(self, y: Tensor, x: Tensor) -> Tensor:
        """log p(x|y) up to a y-independent constant.  y:(G,), x:(B,2) -> (B,G)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        rho = x.norm(dim=-1, keepdim=True)               # (B,1)
        phi = torch.atan2(x[:, 1:2], x[:, 0:1])          # (B,1)
        kappa_x = self.r0 * rho / self.sigma ** 2        # (B,1)
        return kappa_x * torch.cos(self.m * y - phi)     # (B,G)

    def log_prior_y(self, y: Tensor) -> Tensor:
        """log p(y) up to a constant, with prior vM(0, kappa).  y:(G,) -> (G,)."""
        return self.kappa * torch.cos(y)

    def posterior_y_given_x(self, x: Tensor, n_grid: int = 720, include_prior: bool = True):
        """Normalized p(y|x) on a grid.  include_prior=False returns the bare
        likelihood (the 'bottom-up evidence' object)."""
        y = self._grid(n_grid, x.device)
        logp = self.log_likelihood_y(y, x)
        if include_prior:
            logp = logp + self.log_prior_y(y)
        return y, torch.softmax(logp, dim=-1)


def plot_samples():

    # --------------------------------------------------------------------------- #
    #  Sanity check
    # --------------------------------------------------------------------------- #
    ds = WrappedPhaseRingFlat(n=8_000, m=4, kappa=1.0, sigma=0.15, r0=2.0,
                            seed=0, return_latents=True)
    x, y = ds.x, ds.y
    print(f"x{tuple(x.shape)}  y{tuple(y.shape)}  mean||x||={x.norm(dim=-1).mean():.3f}  (r0={ds.r0})")

    yg, p = ds.posterior_y_given_x(x[:1])
    print(f"p(y|x[0]) is {ds.m}-fold aliased; true y[0]={float(y[0]):+.3f}")

    try:
        import matplotlib.pyplot as plt

        fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.6))
        sc = a0.scatter(x[:, 0], x[:, 1], c=y, cmap="hsv", s=4, alpha=0.6)
        a0.set_aspect("equal"); a0.set_title("x, colored by latent y"); fig.colorbar(sc, ax=a0, label="y")
        a0.set_xlabel("x1"); a0.set_ylabel("x2")

        a1.plot(yg, p[0], lw=2)
        a1.fill_between(yg, p[0], alpha=0.15)
        a1.axvline(float(y[0]), color="k", ls="--", lw=1, label="true y")
        a1.set_title(r"$p(y\mid x[0])$ — $m$-fold aliased, no $z$ to resolve")
        a1.set_xlabel("y"); a1.set_xlim(-PI, PI); a1.legend()

        fig.tight_layout(); fig.savefig("wrapped_phase_ring_flat.png", dpi=130)
        print("saved -> wrapped_phase_ring_flat.png")
    except ImportError:
        pass