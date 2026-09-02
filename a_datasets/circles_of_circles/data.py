r"""
Nested Manifolds Toy Dataset
=============================
A 2D toy distribution with continuous latents and synergistic hierarchical
structure, intended as a testbed where guided/hierarchical inference is strictly
necessary to avoid latent collapse.

Generative model (top -> bottom):

$$
z \sim \mathcal{U}(-\pi, \pi)
$$

$$
y \mid z \sim \mathcal{N}(R [\cos(z), \sin(z)]^\top, \sigma_y^2 I_2)
$$

$$
x \mid y \sim \text{Ring}(y, r, \sigma_x)
$$

Because the observed $x$ only provides a local origin point, $x$ alone pins down $y$
to a ring of radius $r$. The top-down prior $p(y|z)$ restricts $y$ to a small
Gaussian blob on a macro-circle of radius $R$. Inferring $y$ optimally requires
BOTH $x$ and $z$:

    $p(y|x)$    ->  one thin ring (localized, but structurally ambiguous)
    $p(y|z)$    ->  one broad blob (unambiguous macro-position, poor local precision)
    $p(y|x,z)$  ->  one sharp peak (perfectly resolved at their intersection)

The closed forms (up to a y-independent constant) heavily rely on the fact that
integrating over an angle yields the modified Bessel function of the first kind.
By employing exponential scaling ($I_0^e$), these are highly stable:

    $\log p(x|y) \approx -\frac{(||x-y|| - r)^2}{2\sigma_x^2} + \log I_0^e\left(\frac{r ||x-y||}{\sigma_x^2}\right)$
    $\log p(y|z) = -\frac{||y - \mu_z||^2}{2\sigma_y^2}$
    $\log p(y|x,z) = \log p(x|y) + \log p(y|z)$

Synergy regime:
Increase $R$ to deepen the macro-ambiguity without $z$. Decrease $\sigma_x$ to
sharpen the micro-constraint from $x$.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

PI = math.pi


# --------------------------------------------------------------------------- #
#  Sampling
# --------------------------------------------------------------------------- #
def generate(
    n: int,
    R: float = 10.0,
    r: float = 2.0,
    sigma_y: float = 0.5,
    sigma_x: float = 0.15,
    seed: int | None = 0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample (x, y, z) from the Nested Manifolds structure.

    Returns
    -------
    x : (n, 2)  observations (micro-structure)
    y : (n, 2)  middle latent (meso-structure)
    z : (n,)    top latent    (macro-angle, radians in (-pi, pi])
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Macro latent
    z = torch.empty(n).uniform_(-PI, PI)

    # Meso latent
    mu_z = R * torch.stack([torch.cos(z), torch.sin(z)], dim=-1)
    y = mu_z + sigma_y * torch.randn(n, 2)

    # Micro latent (integrated out for closed forms, sampled here)
    theta = torch.empty(n).uniform_(-PI, PI)
    mu_theta = r * torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    
    # Observation
    x = y + mu_theta + sigma_x * torch.randn(n, 2)

    return x.to(device), y.to(device), z.to(device)


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #
@dataclass
class NestedManifolds(Dataset):
    """Map-style dataset; samples are precomputed in __init__.

    __getitem__ returns just `x` (shape (2,)) by default, or a dict
    {"x", "y", "z"} when `return_latents=True`.
    """

    n: int = 20_000
    R: float = 10.0
    r: float = 2.0
    sigma_y: float = 0.5
    sigma_x: float = 0.15
    seed: int | None = 0
    return_latents: bool = False

    def __post_init__(self) -> None:
        super().__init__()
        x, y, z = generate(
            self.n, self.R, self.r, self.sigma_y, self.sigma_x, self.seed
        )
        self.x, self.y, self.z = x.float(), y.float(), z.float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        if self.return_latents:
            return {"x": self.x[i], "y": self.y[i], "z": self.z[i]}
        return self.x[i]

    # ----- closed-form posteriors over y (on a 2D grid) ---- #
    def _grid(self, n_grid: int, device) -> Tensor:
        bound = self.R + self.r + 4 * max(self.sigma_x, self.sigma_y)
        g = torch.linspace(-bound, bound, n_grid, device=device)
        Y, X = torch.meshgrid(g, g, indexing='ij')
        return torch.stack([X.flatten(), Y.flatten()], dim=-1)

    def log_likelihood_y(self, y: Tensor, x: Tensor) -> Tensor:
        """log p(x|y) up to a y-independent constant. y:(G,2), x:(B,2) -> (B,G)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        d = torch.cdist(x, y)  # Distance ||x - y||, shape (B,G)
        
        # Exponentially scaled modified Bessel for numerical stability
        z_bessel = self.r * d / (self.sigma_x ** 2)
        log_i0 = torch.special.i0e(z_bessel).log()
        
        return -(d - self.r)**2 / (2 * self.sigma_x**2) + log_i0

    def log_prior_y(self, y: Tensor, z: Tensor) -> Tensor:
        """log p(y|z) up to a constant. y:(G,2), z:(B,) -> (B,G)."""
        if z.dim() == 0:
            z = z.unsqueeze(0)
        mu_z = self.R * torch.stack([torch.cos(z), torch.sin(z)], dim=-1) # (B,2)
        d2 = torch.cdist(mu_z.unsqueeze(1), y.unsqueeze(0)).squeeze(1).pow(2) # (B,G)
        return -d2 / (2 * self.sigma_y**2)

    def posterior_y_given_x(self, x: Tensor, n_grid: int = 150):
        y = self._grid(n_grid, x.device)
        return y, torch.softmax(self.log_likelihood_y(y, x), dim=-1)

    def posterior_y_given_z(self, z: Tensor, n_grid: int = 150):
        y = self._grid(n_grid, z.device)
        return y, torch.softmax(self.log_prior_y(y, z), dim=-1)

    def posterior_y_given_xz(self, x: Tensor, z: Tensor, n_grid: int = 150):
        y = self._grid(n_grid, x.device)
        logp = self.log_likelihood_y(y, x) + self.log_prior_y(y, z)
        return y, torch.softmax(logp, dim=-1)


def grid_entropy(p: Tensor, eps: float = 1e-12) -> Tensor:
    """Discrete entropy (nats) of a normalized grid distribution. The 2D area
    element dy^2 is common to all posteriors here, so differences are meaningful."""
    return -(p * (p + eps).log()).sum(dim=-1)


# --------------------------------------------------------------------------- #
#  Sanity check / synergy demonstration
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ds = NestedManifolds(n=8_000, R=10.0, r=2.0, sigma_y=0.5, sigma_x=0.15,
                         seed=0, return_latents=True)
    x, y, z = ds.x, ds.y, ds.z
    print(f"data: x{tuple(x.shape)}  R={ds.R}  r={ds.r}  "
          f"macro blur={ds.sigma_y:.2f}  micro blur={ds.sigma_x:.2f}")

    # Evaluate posteriors over a dense 2D grid (150x150 = 22,500 points)
    _, p_x  = ds.posterior_y_given_x(x[:5])
    _, p_z  = ds.posterior_y_given_z(z[:5])
    _, p_xz = ds.posterior_y_given_xz(x[:5], z[:5])
    H = lambda p: grid_entropy(p).mean().item()
    
    print(f"H(y|x)   = {H(p_x):.3f} nats   (ring shape; structurally ambiguous)")
    print(f"H(y|z)   = {H(p_z):.3f} nats   (Gaussian shape; lacks local precision)")
    print(f"H(y|x,z) = {H(p_xz):.3f} nats   (resolved to a sharp local peak)")
    print(f"  info from z (resolves structure): {H(p_x)-H(p_xz):.3f} nats")
    print(f"  info from x (adds precision):     {H(p_z)-H(p_xz):.3f} nats")


# ---------------------------------------------------------------------------- #
#                    plotting all the relevant distributions                   #
# ---------------------------------------------------------------------------- #

def plot_distributions(seed=42):
    """
    Plot the eight distributions of the Nested Manifolds model.

    Standalone (numpy + scipy.special + matplotlib only) so it runs without torch.
    Exploits exact convolution properties (sum of independent Gaussians and rings)
    to compute fully closed-form density fields without 2D integration.

    Panels (a representative sample x*, y*, z* is drawn and held fixed):
        p(z)        p(y|z*)      p(x|y*)      p(x|z*)
        p(x)        p(y|x*)      p(z|x*)      p(y|x*,z*)
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.special
    from matplotlib import gridspec

    # --------------------------------------------------------------------------- #
    #  Model parameters
    # --------------------------------------------------------------------------- #
    R, r = 10.0, 2.0
    sigma_y, sigma_x = 0.5, 0.15

    # --------------------------------------------------------------------------- #
    #  Grids
    # --------------------------------------------------------------------------- #
    G = 720                                   # fine angular grid for z
    ang = np.linspace(-np.pi, np.pi, G)

    D = 200                                   # spatial grid resolution
    bound = R + r + 2.0
    gx = np.linspace(-bound, bound, D)
    XX, YY = np.meshgrid(gx, gx)
    xy = np.stack([XX.ravel(), YY.ravel()], -1).astype(np.float32)   # (D*D, 2)

    # --------------------------------------------------------------------------- #
    #  Helpers (Exact Closed Forms)
    # --------------------------------------------------------------------------- #
    def norm1d(p):
        return p / (p.sum() * (ang[1] - ang[0]))

    def norm2d(p):
        return p / p.sum() # relative scale for visualization

    def ring_pdf(v, center, radius, var):
        """p = Ring(v; center, radius, var). v:(P,2), center:(M,2) -> (M,P)"""
        d = np.linalg.norm(v[None, :, :] - center[:, None, :], axis=-1)
        log_p = -(d - radius)**2 / (2 * var) + np.log(scipy.special.i0e(radius * d / var) + 1e-12)
        return np.exp(log_p)

    def gaussian_2d(v, center, var):
        """p = N(v; center, var*I). v:(P,2), center:(M,2) -> (M,P)"""
        d2 = np.sum((v[None, :, :] - center[:, None, :])**2, axis=-1)
        return np.exp(-d2 / (2 * var))

    # --------------------------------------------------------------------------- #
    #  Draw one representative sample
    # --------------------------------------------------------------------------- #
    rng = np.random.default_rng(seed)
    z_star = rng.uniform(-np.pi, np.pi)
    mu_z_star = R * np.array([np.cos(z_star), np.sin(z_star)])
    y_star = mu_z_star + sigma_y * rng.standard_normal(2)
    theta_star = rng.uniform(-np.pi, np.pi)
    x_star = y_star + r * np.array([np.cos(theta_star), np.sin(theta_star)]) + sigma_x * rng.standard_normal(2)

    origin = np.zeros((1, 2))
    mu_z_all = R * np.stack([np.cos(ang), np.sin(ang)], -1)  # (G, 2)

    # --------------------------------------------------------------------------- #
    #  The eight exact distributions
    # --------------------------------------------------------------------------- #
    # 1) p(z)
    p_z = norm1d(np.ones_like(ang))

    # 2) p(y | z*) = N(y; mu_z*, sigma_y^2)
    p_y_z = gaussian_2d(xy, mu_z_star[None, :], sigma_y**2)[0].reshape(D, D)

    # 3) p(x | y*) = Ring(x; y*, r, sigma_x^2)
    p_x_y = ring_pdf(xy, y_star[None, :], r, sigma_x**2)[0].reshape(D, D)

    # 4) p(x | z*) = Convolution of N(sigma_y^2) and Ring(sigma_x^2)
    # Exact closure: Ring with variance = sigma_y^2 + sigma_x^2
    p_x_z = ring_pdf(xy, mu_z_star[None, :], r, sigma_x**2 + sigma_y**2)[0].reshape(D, D)

    # 5) p(x) = int p(x|z) p(z) dz
    px_all = ring_pdf(xy, mu_z_all, r, sigma_x**2 + sigma_y**2)
    p_x = px_all.mean(axis=0).reshape(D, D)

    # 6) p(y | x*) ~ p(x*|y) p(y)  (Product of two rings)
    py_x_star_lik = ring_pdf(xy, x_star[None, :], r, sigma_x**2)[0]
    py_marginal = ring_pdf(xy, origin, R, sigma_y**2)[0]
    p_y_x = norm2d(py_x_star_lik * py_marginal).reshape(D, D)

    # 7) p(z | x*) ~ p(x*|z) p(z)
    px_z_all = ring_pdf(x_star[None, :], mu_z_all, r, sigma_x**2 + sigma_y**2)[:, 0]
    p_z_x = norm1d(px_z_all)

    # 8) p(y | x*, z*) ~ p(x*|y) p(y|z*)
    py_z_prior = gaussian_2d(xy, mu_z_star[None, :], sigma_y**2)[0]
    p_y_xz = norm2d(py_x_star_lik * py_z_prior).reshape(D, D)

    # Overlay samples
    N_s = 4000
    zz = rng.uniform(-np.pi, np.pi, N_s)
    yy = R * np.stack([np.cos(zz), np.sin(zz)], -1) + sigma_y * rng.standard_normal((N_s, 2))
    tt = rng.uniform(-np.pi, np.pi, N_s)
    xs = yy + r * np.stack([np.cos(tt), np.sin(tt)], -1) + sigma_x * rng.standard_normal((N_s, 2))

    # --------------------------------------------------------------------------- #
    #  Plotting
    # --------------------------------------------------------------------------- #
    ACC = "#3b6ea5"
    ACC2 = "#c25b3a"
    TRUE = "#111111"

    def angular(ax, p, title, color=ACC, true=None):
        ax.plot(ang, p, color=color, lw=2)
        ax.fill_between(ang, p, color=color, alpha=0.15)
        if true is not None:
            ax.axvline(true, color=TRUE, ls="--", lw=1.4, label="true sample")
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
        ax.set_yticks([])
        ax.set_title(title, fontsize=12)
        ax.margins(y=0.05)

    def field(ax, F, title, pts=None, pt_styles=None):
        ax.imshow(F, origin="lower", extent=[-bound, bound, -bound, bound],
                  cmap="magma", aspect="equal", vmin=0, vmax=F.max())
        th = np.linspace(-np.pi, np.pi, 256)
        ax.plot(R * np.cos(th), R * np.sin(th), color="w", lw=0.6, alpha=0.25, ls="--")
        if pts is not None:
            for (px, py), st in zip(pts, pt_styles):
                ax.scatter([px], [py], **st)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=12)

    fig = plt.figure(figsize=(15, 7.6))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.32, wspace=0.18)

    # ---- row 0: generative / forward ----
    ax = fig.add_subplot(gs[0, 0])
    angular(ax, p_z, r"$p(z)$  — prior on top latent", color=ACC2, true=z_star)
    ax.legend(fontsize=8, loc="upper right", frameon=False)

    ax = fig.add_subplot(gs[0, 1])
    field(ax, p_y_z, r"$p(y\mid z^\star)$  — top-down prior",
          pts=[(y_star[0], y_star[1]), (mu_z_star[0], mu_z_star[1])],
          pt_styles=[dict(marker="x", s=42, color="cyan", lw=1.6, label="true y"),
                     dict(marker="+", s=60, color="w", lw=1.2)])

    ax = fig.add_subplot(gs[0, 2])
    field(ax, p_x_y, r"$p(x\mid y^\star)$  — likelihood",
          pts=[(y_star[0], y_star[1]), (x_star[0], x_star[1])],
          pt_styles=[dict(marker="o", s=28, facecolor="none", edgecolor="w", lw=1.2),
                     dict(marker="x", s=42, color="cyan", lw=1.6)])

    ax = fig.add_subplot(gs[0, 3])
    field(ax, p_x_z, r"$p(x\mid z^\star)=\int p(x\mid y)p(y\mid z^\star)dy$")

    # ---- row 1: marginal + inference ----
    ax = fig.add_subplot(gs[1, 0])
    field(ax, p_x, r"$p(x)=\int p(x\mid z)p(z)dz$",
          pts=[(x_star[0], x_star[1])],
          pt_styles=[dict(marker="x", s=42, color="cyan", lw=1.6)])
    ax.scatter(xs[:, 0], xs[:, 1], s=1.5, color="white", alpha=0.12)

    ax = fig.add_subplot(gs[1, 1])
    field(ax, p_y_x, r"$p(y\mid x^\star)$  — ambiguity ring",
          pts=[(y_star[0], y_star[1])],
          pt_styles=[dict(marker="x", s=42, color="cyan", lw=1.6)])

    ax = fig.add_subplot(gs[1, 2])
    angular(ax, p_z_x, r"$p(z\mid x^\star)$  — multi-modal posterior", color=ACC2, true=z_star)
    ax.legend(fontsize=8, loc="upper right", frameon=False)

    ax = fig.add_subplot(gs[1, 3])
    field(ax, p_y_xz, r"$p(y\mid x^\star,z^\star)$  — resolved",
          pts=[(y_star[0], y_star[1])],
          pt_styles=[dict(marker="x", s=42, color="cyan", lw=1.6)])

    fig.suptitle(
        rf"Nested Manifolds Hierarchy: $R={R}$, $r={r}$, $\sigma_y={sigma_y}$, $\sigma_x={sigma_x}$",
        fontsize=13, y=0.99,
    )
    plt.show()

if __name__ == "__main__":
    plot_distributions()