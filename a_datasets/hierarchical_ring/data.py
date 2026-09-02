"""
Hierarchical wrapped-phase ring
================================
A 2D toy distribution with *continuous* latents and *synergistic* hierarchical
structure, intended as a testbed where guided/hierarchical inference is strictly
necessary (unlike a GSM, whose single radial latent is redundant with ||x_t||).

Generative model (top -> bottom):

    z        ~  vM(0, kappa_z)                                  # top latent  (coarse angle)
    y | z    ~  vM(z, kappa)                                    # middle latent (fine angle)
    x | y    ~  N( r0 * [cos(m*y), sin(m*y)], sigma^2 I_2 )     # 2D observation

Because the observed angle of x reveals (m*y) mod 2*pi, x alone pins down y only
up to m aliases.  The top-down prior p(y|z) is too broad to localize y on its own
but selects the correct alias.  Inferring y therefore requires BOTH x and z:

    p(y|x)    ->  m equally-good sharp peaks   (fine, but aliased)
    p(y|z)    ->  one broad bump               (coarse, unaliased)
    p(y|x,z)  ->  one sharp peak               (resolved)

The closed forms (up to a y-independent constant) are:

    log p(x|y) = (r0*||x|| / sigma^2) * cos(m*y - angle(x))     # bottom-up evidence
    log p(y|z) = kappa * cos(y - z)                            # top-down prior
    log p(y|x,z) = log p(x|y) + log p(y|z)

Synergy regime (each latent alone insufficient, both together sufficient):

        sigma / (m*r0)   <<   kappa**-0.5   <~   2*pi / m
        \-- fine, from x --/   \-- coarse, --/    \-- alias --/
                                  from z            spacing

The left inequality makes x supply precision z lacks; the right makes z resolve
the alias x cannot.  Increase m to deepen the synergy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions import VonMises
from torch.utils.data import Dataset

PI = math.pi


# --------------------------------------------------------------------------- #
#  Sampling
# --------------------------------------------------------------------------- #
def generate(
    n: int,
    m: int = 4,
    kappa: float = 4.0,
    kappa_z: float = 0.0,
    r0: float = 2.0,
    sigma: float = 0.15,
    seed: int | None = 0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample (x, y, z) from the hierarchical wrapped-phase ring.

    Returns
    -------
    x : (n, 2)  observations
    y : (n,)    middle latent (fine angle, radians in (-pi, pi])
    z : (n,)    top latent    (coarse angle, radians in (-pi, pi])

    Note: VonMises.sample() draws from torch's *global* RNG, so `seed` is applied
    via torch.manual_seed for reproducibility.
    """
    if seed is not None:
        torch.manual_seed(seed)

    if kappa_z and kappa_z > 0:
        z = VonMises(torch.zeros(n), kappa_z * torch.ones(n)).sample()
    else:
        z = torch.empty(n).uniform_(-PI, PI)          # kappa_z = 0  ->  uniform top

    y = VonMises(z, kappa * torch.ones(n)).sample()

    center = r0 * torch.stack([torch.cos(m * y), torch.sin(m * y)], dim=-1)
    x = center + sigma * torch.randn(n, 2)

    return x.to(device), y.to(device), z.to(device)


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #
@dataclass
class WrappedPhaseRing(Dataset):
    """Map-style dataset; samples are precomputed in __init__.

    __getitem__ returns just `x` (shape (2,)) by default, or a dict
    {"x", "y", "z"} when `return_latents=True` (useful for encoder analysis).
    The ground-truth latents are always available as `.x`, `.y`, `.z`.
    """

    n: int = 20_000
    m: int = 4
    kappa: float = 4.0
    kappa_z: float = 0.0
    r0: float = 2.0
    sigma: float = 0.15
    seed: int | None = 0
    return_latents: bool = False

    def __post_init__(self) -> None:
        super().__init__()
        x, y, z = generate(
            self.n, self.m, self.kappa, self.kappa_z, self.r0, self.sigma, self.seed
        )
        self.x, self.y, self.z = x.float(), y.float(), z.float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        if self.return_latents:
            return {"x": self.x[i], "y": self.y[i], "z": self.z[i]}
        return self.x[i]

    # ----- closed-form posteriors over y (on a grid; up to normalization) ---- #
    def _grid(self, n_grid: int, device) -> Tensor:
        return torch.linspace(-PI, PI, n_grid, device=device)

    def log_likelihood_y(self, y: Tensor, x: Tensor) -> Tensor:
        """log p(x|y) up to a y-independent constant.  y:(G,), x:(B,2) -> (B,G)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        rho = x.norm(dim=-1, keepdim=True)                  # (B,1)
        phi = torch.atan2(x[:, 1:2], x[:, 0:1])             # (B,1)
        kappa_x = self.r0 * rho / self.sigma ** 2           # (B,1)  effective conc on (m*y)
        return kappa_x * torch.cos(self.m * y - phi)        # (B,G)

    def log_prior_y(self, y: Tensor, z: Tensor) -> Tensor:
        """log p(y|z) up to a constant.  y:(G,), z:(B,) -> (B,G)."""
        if z.dim() == 0:
            z = z.unsqueeze(0)
        return self.kappa * torch.cos(y - z[:, None])

    def posterior_y_given_x(self, x: Tensor, n_grid: int = 720):
        y = self._grid(n_grid, x.device)
        return y, torch.softmax(self.log_likelihood_y(y, x), dim=-1)

    def posterior_y_given_z(self, z: Tensor, n_grid: int = 720):
        y = self._grid(n_grid, z.device)
        return y, torch.softmax(self.log_prior_y(y, z), dim=-1)

    def posterior_y_given_xz(self, x: Tensor, z: Tensor, n_grid: int = 720):
        y = self._grid(n_grid, x.device)
        logp = self.log_likelihood_y(y, x) + self.log_prior_y(y, z)
        return y, torch.softmax(logp, dim=-1)


def grid_entropy(p: Tensor, eps: float = 1e-12) -> Tensor:
    """Discrete entropy (nats) of a normalized grid distribution.  The constant
    log(dy) is common to all posteriors here, so differences are meaningful."""
    return -(p * (p + eps).log()).sum(dim=-1)


# --------------------------------------------------------------------------- #
#  Sanity check / synergy demonstration
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ds = WrappedPhaseRing(n=8_000, m=4, kappa=4.0, sigma=0.15, r0=2.0,
                          seed=0, return_latents=True)
    x, y, z = ds.x, ds.y, ds.z
    print(f"data: x{tuple(x.shape)}  m={ds.m}  prior std~{ds.kappa**-0.5:.2f} rad  "
          f"alias spacing 2pi/m={2*PI/ds.m:.2f} rad")

    _, p_x  = ds.posterior_y_given_x(x)
    _, p_z  = ds.posterior_y_given_z(z)
    _, p_xz = ds.posterior_y_given_xz(x, z)
    H = lambda p: grid_entropy(p).mean().item()
    print(f"H(y|x)   = {H(p_x):.3f} nats   (fine, but {ds.m}-fold aliased)")
    print(f"H(y|z)   = {H(p_z):.3f} nats   (coarse, unaliased)")
    print(f"H(y|x,z) = {H(p_xz):.3f} nats  (resolved)")
    print(f"  unique from z (resolves alias): {H(p_x)-H(p_xz):.3f}  ~ log m = {math.log(ds.m):.3f}")
    print(f"  unique from x (localizes y):    {H(p_z)-H(p_xz):.3f}")

    # ----- optional figure -------------------------------------------------- #
    try:
        import matplotlib.pyplot as plt

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.6))

        sc = ax0.scatter(x[:, 0], x[:, 1], c=y, cmap="hsv", s=4, alpha=0.6)
        ax0.set_aspect("equal"); ax0.set_title("x, colored by latent y\n(y wraps m times around the ring)")
        ax0.set_xlabel("x1"); ax0.set_ylabel("x2")
        fig.colorbar(sc, ax=ax0, label="y")

        i = 0
        yg, _ = ds.posterior_y_given_x(x[i:i+1])
        _, px  = ds.posterior_y_given_x(x[i:i+1])
        _, pz  = ds.posterior_y_given_z(z[i:i+1])
        _, pxz = ds.posterior_y_given_xz(x[i:i+1], z[i:i+1])
        ax1.plot(yg, px[0],  label="p(y|x)  (aliased)")
        ax1.plot(yg, pz[0],  label="p(y|z)  (coarse)")
        ax1.plot(yg, pxz[0], label="p(y|x,z) (resolved)", lw=2.2)
        ax1.axvline(float(y[i]), color="k", ls="--", lw=1, label="true y")
        ax1.set_title("posteriors over y for one sample"); ax1.set_xlabel("y"); ax1.legend()

        fig.tight_layout()
        fig.savefig("wrapped_phase_ring.png", dpi=130)
        print("saved figure -> wrapped_phase_ring.png")
    except ImportError:
        pass



# ---------------------------------------------------------------------------- #
#                    plotting all the relevant distributions                   #
# ---------------------------------------------------------------------------- #

def plot_distributions(seed=3, m=4, kappa=4., kappa_z=1.0, r0=2.0, sigma=0.15):
    """
    Plot the eight distributions of the hierarchical wrapped-phase ring model.

    Standalone (numpy + matplotlib only) so it runs without torch; it mirrors the
    generative model of WrappedPhaseRing exactly:

        z      ~ vM(0, kappa_z)
        y | z  ~ vM(z, kappa)
        x | y  ~ N( r0*[cos(m*y), sin(m*y)], sigma^2 I_2 )

    Panels (a representative sample x*, y*, z* is drawn and held fixed for all the
    conditionals so you can watch the aliasing in p(y|x) / p(z|x) get resolved):

        p(z)        p(y|z*)      p(x|y*)      p(x|z*)
        p(x)        p(y|x*)      p(z|x*)      p(y|x*,z*)

    Angular densities are line plots over (-pi, pi]; densities over x are heatmaps.
    All fields are normalized for display (relative scale), which is all that matters
    for reading off shape and multimodality.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import gridspec

    # --------------------------------------------------------------------------- #
    #  Model parameters  (kappa_z > 0 here so p(z) and p(x) are visibly structured;
    #  set kappa_z = 0 to recover a uniform top latent and a uniform ring p(x))
    # --------------------------------------------------------------------------- #
    # m, kappa, kappa_z = m, kappa, kappa_z
    # r0, sigma = r0, sigma
    # seed = seed

    # --------------------------------------------------------------------------- #
    #  Grids
    # --------------------------------------------------------------------------- #
    G = 720                                   # fine angular grid (y, z; 1-D panels)
    ang = np.linspace(-np.pi, np.pi, G)
    dang = ang[1] - ang[0]

    D = 200                                   # x-grid resolution (2-D panels)
    R = r0 + 5 * sigma
    gx = np.linspace(-R, R, D)
    XX, YY = np.meshgrid(gx, gx)              # 'xy' indexing: XX[i,j]=gx[j], YY[i,j]=gx[i]
    xy = np.stack([XX.ravel(), YY.ravel()], -1).astype(np.float32)   # (P, 2)

    G2 = 512                                  # angular grid for the 2-D marginal integrals
    ang2 = np.linspace(-np.pi, np.pi, G2)


    # --------------------------------------------------------------------------- #
    #  Helpers
    # --------------------------------------------------------------------------- #
    def vm_unnorm(theta, mu, kap):
        """von Mises shape exp(kap cos(theta - mu)); kap=0 -> uniform."""
        return np.exp(kap * np.cos(theta - mu))


    def norm1d(p):
        return p / (p.sum() * dang)


    def ring_gaussian(xy, ring_ang):
        """N(x; r0*[cos,sin](ring_ang), sigma^2 I) up to a constant.
        ring_ang: (A,)  ->  returns (P, A) float32."""
        cx = (r0 * np.cos(ring_ang)).astype(np.float32)
        cy = (r0 * np.sin(ring_ang)).astype(np.float32)
        d2 = (xy[:, 0:1] - cx[None, :]) ** 2 + (xy[:, 1:2] - cy[None, :]) ** 2
        return np.exp(-0.5 * d2 / np.float32(sigma ** 2))


    # --------------------------------------------------------------------------- #
    #  Draw one representative sample and its sufficient statistics
    # --------------------------------------------------------------------------- #
    rng = np.random.default_rng(seed)
    z_star = rng.vonmises(0.0, kappa_z)
    y_star = rng.vonmises(z_star, kappa)
    x_star = r0 * np.array([np.cos(m * y_star), np.sin(m * y_star)]) + sigma * rng.standard_normal(2)
    phi_star = np.arctan2(x_star[1], x_star[0])          # observed angle
    rho_star = np.linalg.norm(x_star)
    kx_star = r0 * rho_star / sigma ** 2                 # effective conc. of likelihood on (m*y)

    # the m aliases of y consistent with x* (wrapped to (-pi, pi])
    aliases = np.mod((phi_star + 2 * np.pi * np.arange(m)) / m + np.pi, 2 * np.pi) - np.pi

    # --------------------------------------------------------------------------- #
    #  Reusable ring-Gaussian matrix (shared by the two 2-D marginals)
    # --------------------------------------------------------------------------- #
    G_ring = ring_gaussian(xy, m * ang2)                 # (P, G2)

    # marginal prior over the top latent on both grids
    wz_fine = norm1d(vm_unnorm(ang, 0.0, kappa_z))       # p(z) on fine grid
    wz2 = vm_unnorm(ang2, 0.0, kappa_z); wz2 /= wz2.sum()

    # marginal p(y) = int vM(y;z,kappa) p(z) dz
    PY_fine = (vm_unnorm(ang[:, None], ang[None, :], kappa) * wz_fine[None, :] * dang).sum(1)
    PY_fine = norm1d(PY_fine)
    PY2 = (np.exp(kappa * np.cos(ang2[:, None] - ang[None, :])) * wz_fine[None, :] * dang).sum(1)
    PY2 /= PY2.sum()

    # --------------------------------------------------------------------------- #
    #  The eight distributions
    # --------------------------------------------------------------------------- #
    # 1) p(z)
    p_z = wz_fine

    # 2) p(y | z*)
    p_y_z = norm1d(vm_unnorm(ang, z_star, kappa))

    # 3) p(x | y*)               single Gaussian blob on the ring
    p_x_y = ring_gaussian(xy, np.array([m * y_star]))[:, 0].reshape(D, D)

    # 4) p(x | z*) = int p(x|y) vM(y;z*,kappa) dy   smeared arc
    wy = vm_unnorm(ang2, z_star, kappa); wy /= wy.sum()
    p_x_z = (G_ring * wy[None, :]).sum(1).reshape(D, D)

    # 5) p(x) = int p(x|y) p(y) dy
    p_x = (G_ring * PY2[None, :]).sum(1).reshape(D, D)

    # 6) p(y | x*) ~ p(x*|y) p(y)                    m-fold aliased
    lik_y = np.exp(kx_star * np.cos(m * ang - phi_star))
    p_y_x = norm1d(lik_y * PY_fine)

    # 7) p(z | x*) ~ p(z) int p(x*|y) vM(y;z,kappa) dy   also aliased
    hz = (vm_unnorm(ang[:, None], ang[None, :], kappa) * lik_y[None, :] * dang).sum(1)  # over y
    p_z_x = norm1d(vm_unnorm(ang, 0.0, kappa_z) * hz)

    # 8) p(y | x*, z*) ~ p(x*|y) vM(y;z*,kappa)      single resolved peak
    p_y_xz = norm1d(lik_y * vm_unnorm(ang, z_star, kappa))

    # a few samples to overlay on p(x)
    N = 4000
    zz = rng.vonmises(0.0, kappa_z, N)
    yy = rng.vonmises(zz, kappa, N)
    xs = r0 * np.stack([np.cos(m * yy), np.sin(m * yy)], -1) + sigma * rng.standard_normal((N, 2))


    # --------------------------------------------------------------------------- #
    #  Plotting
    # --------------------------------------------------------------------------- #
    ACC = "#3b6ea5"        # density line
    ACC2 = "#c25b3a"       # secondary
    TRUE = "#111111"       # true-value marker
    ALIAS = "#888888"      # alias markers
    PI_TICKS = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    PI_LABELS = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]


    def angular(ax, p, title, color=ACC, true=None, marks=None, mark_label=None, sample_label=None):
        ax.plot(ang, p, color=color, lw=2)
        ax.fill_between(ang, p, color=color, alpha=0.15)
        if marks is not None:
            for k, mk in enumerate(np.atleast_1d(marks)):
                ax.axvline(mk, color=ALIAS, ls=":", lw=1.2,
                        label=(mark_label if k == 0 else None))
        if true is not None:
            sample_label = "sample" if sample_label is None else sample_label
            ax.axvline(true, color=TRUE, ls="--", lw=1.4, label=sample_label)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks(PI_TICKS); ax.set_xticklabels(PI_LABELS)
        ax.set_yticks([])
        ax.set_title(title, fontsize=12)
        ax.margins(y=0.05)


    def field(ax, F, title, pts=None, pt_styles=None):
        F = F / F.max()
        ax.imshow(F, origin="lower", extent=[-R, R, -R, R], cmap="magma",
                aspect="equal", vmin=0, vmax=1)
        th = np.linspace(-np.pi, np.pi, 256)
        ax.plot(r0 * np.cos(th), r0 * np.sin(th), color="w", lw=0.6, alpha=0.25)
        if pts is not None:
            for (px, py), st in zip(pts, pt_styles):
                ax.scatter([px], [py], **st)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=12)


    fig = plt.figure(figsize=(15, 7.6))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.32, wspace=0.18)

    # ---- row 0: generative / forward ----
    ax = fig.add_subplot(gs[0, 0])
    angular(ax, p_z, r"$p(z)$  — prior on top latent", color=ACC2, true=z_star, sample_label="z sample")
    ax.legend(fontsize=8, loc="upper right", frameon=False)

    ax = fig.add_subplot(gs[0, 1])
    angular(ax, p_y_z, r"$p(y\mid z^\star)$  — top-down prior", color=ACC2,
            true=y_star, marks=z_star, mark_label=r"$z^\star$", sample_label="y sample")
    ax.legend(fontsize=8, loc="upper right", frameon=False)

    ax = fig.add_subplot(gs[0, 2])
    center = (r0 * np.cos(m * y_star), r0 * np.sin(m * y_star))
    field(ax, p_x_y, r"$p(x\mid y^\star)$  — likelihood",
        pts=[center, (x_star[0], x_star[1])],
        pt_styles=[dict(marker="o", s=28, facecolor="none", edgecolor="w", lw=1.2),
                    dict(marker="x", s=42, color="cyan", lw=1.6)])

    ax = fig.add_subplot(gs[0, 3])
    field(ax, p_x_z, r"$p(x\mid z^\star)=\int p(x\mid y)\,p(y\mid z^\star)\,dy$")

    # ---- row 1: marginal + inference ----
    ax = fig.add_subplot(gs[1, 0])
    field(ax, p_x, r"$p(x)=\int p(x\mid y)\,p(y)\,dy$",
        pts=[(x_star[0], x_star[1])],
        pt_styles=[dict(marker="x", s=42, color="cyan", lw=1.6)])
    ax.scatter(xs[:, 0], xs[:, 1], s=1.5, color="white", alpha=0.12)

    ax = fig.add_subplot(gs[1, 1])
    angular(ax, p_y_x, r"$p(y\mid x^\star)$  — $m$-fold aliased",
            true=y_star, marks=aliases, mark_label="aliases", sample_label="true y")
    ax.legend(fontsize=8, loc="upper right", frameon=False)

    ax = fig.add_subplot(gs[1, 2])
    angular(ax, p_z_x, r"$p(z\mid x^\star)$  — also aliased",
            color=ACC2, true=z_star, marks=aliases, mark_label="aliases", sample_label="true z")
    ax.legend(fontsize=8, loc="upper right", frameon=False)

    ax = fig.add_subplot(gs[1, 3])
    angular(ax, p_y_xz, r"$p(y\mid x^\star,z^\star)$  — resolved",
            true=y_star, marks=z_star, mark_label=r"$z^\star$", sample_label="true y")
    ax.legend(fontsize=8, loc="upper right", frameon=False)

    fig.suptitle(
        rf"Wrapped-phase ring   $m={m}$, $\kappa={kappa}$, $\kappa_z={kappa_z}$, "
        rf"$r_0={r0}$, $\sigma={sigma}$   "
        rf"(x* drawn from y*={y_star:+.2f}, z*={z_star:+.2f})",
        fontsize=13, y=0.99,
    )
    # fig.savefig("wrapped_phase_ring_distributions.png", dpi=130, bbox_inches="tight")
    # print("saved -> wrapped_phase_ring_distributions.png")
    # print(f"x* = ({x_star[0]:+.2f}, {x_star[1]:+.2f})   phi* = {phi_star:+.2f}   "
    #       f"y* = {y_star:+.2f}   z* = {z_star:+.2f}")
    # print(f"aliases of y consistent with x*: {np.sort(aliases).round(2)}")