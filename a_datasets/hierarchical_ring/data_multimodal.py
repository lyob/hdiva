r"""
Prior-winding wrapped-phase ring
=================================
Sibling of `WrappedPhaseRing` with the m-fold winding *relocated from the
emission to the prior*.  Where the aliased ring hides an m:1 map in the
likelihood (so inference is hard and z is needed to resolve it), this model
puts the m:1 map in the top-down prior p(y|z): generation is multimodal but
every inference posterior is unimodal.

Generative model (top -> bottom):

    z        ~  vM(0, kappa_z)                                  # top latent (coarse angle)
    y | z    ~  (1/m) * sum_{k=0}^{m-1} vM( (z + 2*pi*k)/m , kappa )   # MULTIMODAL: m preimages of z
    x | y    ~  N( r0 * [cos(y), sin(y)] , sigma^2 I_2 )        # emission winds ONCE (injective)

The m mixture centers (z + 2*pi*k)/m all satisfy  m*y == z (mod 2*pi),
so they are exactly the m angles that the coarsening y -> (m*y mod 2*pi) folds
onto a single z.  Because the emission winds once, the observed angle of x pins
y to a single value; the winding lives entirely upstream of y now.

Modality structure (all verified by plot_distributions):

    p(y|z)    ->  m sharp von Mises modes      (MULTIMODAL  -- the m preimages)
    p(x|z)    ->  m blobs around the ring       (multimodal arc)
    p(y|x)    ->  one sharp peak at angle(x)     (unimodal -- emission injective)
    p(z|x)    ->  one broad peak at m*y mod 2pi  (unimodal; ~ m^2 wider than p(y|x))
    p(y|x,z)  ->  one peak; likelihood selects the nearest of the m prior modes (unimodal)

Closed forms over y (up to a y-independent constant):

    log p(x|y)   = (r0*||x|| / sigma^2) * cos(y - angle(x))          # single mode, NO m
    log p(y|z)   = logsumexp_k [ kappa * cos(y - (z+2*pi*k)/m) ] - log m
    log p(y|x,z) = log p(x|y) + log p(y|z)

Resolution regime (likelihood must resolve which of the m prior bumps fired):

        sigma / r0   <<   2*pi / m
        \-- angular  --/   \-- bump --/
            noise           spacing

Increase m to add more prior modes; the inference posteriors stay unimodal as
long as the inequality holds.
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
    """Sample (x, y, z) from the prior-winding wrapped-phase ring.

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

    # y | z is a uniform mixture of m von Mises bumps; pick a component then sample
    k = torch.randint(0, m, (n,))
    mu = (z + 2 * PI * k) / m                          # bump center for each sample
    y = VonMises(mu, kappa * torch.ones(n)).sample()
    y = (y + PI) % (2 * PI) - PI                        # wrap to (-pi, pi]

    center = r0 * torch.stack([torch.cos(y), torch.sin(y)], dim=-1)   # winds ONCE
    x = center + sigma * torch.randn(n, 2)

    return x.to(device), y.to(device), z.to(device)


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #
@dataclass
class PriorWindingRing(Dataset):
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
        """log p(x|y) up to a y-independent constant.  y:(G,), x:(B,2) -> (B,G).

        Single-mode (emission winds once): a von Mises in y with mean angle(x)
        and concentration r0*||x||/sigma^2 -- note there is NO factor of m here,
        which is exactly why p(y|x) is unaliased.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        rho = x.norm(dim=-1, keepdim=True)                  # (B,1)
        phi = torch.atan2(x[:, 1:2], x[:, 0:1])             # (B,1)
        kappa_x = self.r0 * rho / self.sigma ** 2           # (B,1)  effective conc on y
        return kappa_x * torch.cos(y - phi)                 # (B,G)

    def log_prior_y(self, y: Tensor, z: Tensor) -> Tensor:
        """log p(y|z) up to a constant.  y:(G,), z:(B,) -> (B,G).

        Mixture of m von Mises bumps centered at (z + 2*pi*k)/m, k = 0..m-1.
        """
        if z.dim() == 0:
            z = z.unsqueeze(0)
        k = torch.arange(self.m, device=y.device)
        mu = (z[:, None, None] + 2 * PI * k[None, :, None]) / self.m   # (B, m, 1)
        comp = self.kappa * torch.cos(y[None, None, :] - mu)           # (B, m, G)
        return torch.logsumexp(comp, dim=1) - math.log(self.m)        # (B, G)

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
#  Sanity check / "inference is easy" demonstration
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ds = PriorWindingRing(n=8_000, m=4, kappa=4.0, sigma=0.15, r0=2.0,
                          kappa_z=1.0, seed=0, return_latents=True)
    x, y, z = ds.x, ds.y, ds.z
    print(f"data: x{tuple(x.shape)}  m={ds.m}  prior modes per z = {ds.m}  "
          f"bump spacing 2pi/m={2*PI/ds.m:.2f} rad  noise sigma/r0={ds.sigma/ds.r0:.3f} rad")

    _, p_x  = ds.posterior_y_given_x(x)
    _, p_z  = ds.posterior_y_given_z(z)
    _, p_xz = ds.posterior_y_given_xz(x, z)
    H = lambda p: grid_entropy(p).mean().item()
    print(f"H(y|x)   = {H(p_x):.3f} nats   (unimodal -- emission winds once)")
    print(f"H(y|z)   = {H(p_z):.3f} nats   ({ds.m}-modal prior, ~log m + spread)")
    print(f"H(y|x,z) = {H(p_xz):.3f} nats  (unimodal, ~ H(y|x))")
    print(f"  unique from z (resolves alias): {H(p_x)-H(p_xz):.3f}  ~ 0  (no alias to resolve)")
    print(f"  unique from x (localizes y):    {H(p_z)-H(p_xz):.3f}  (x does the localizing)")

    # ----- optional figure -------------------------------------------------- #
    try:
        import matplotlib.pyplot as plt

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.6))

        sc = ax0.scatter(x[:, 0], x[:, 1], c=y, cmap="hsv", s=4, alpha=0.6)
        ax0.set_aspect("equal")
        ax0.set_title("x, colored by latent y\n(hue wraps once: emission is injective)")
        ax0.set_xlabel("x1"); ax0.set_ylabel("x2")
        fig.colorbar(sc, ax=ax0, label="y")

        i = 0
        yg, px  = ds.posterior_y_given_x(x[i:i+1])
        _,  pz  = ds.posterior_y_given_z(z[i:i+1])
        _,  pxz = ds.posterior_y_given_xz(x[i:i+1], z[i:i+1])
        ax1.plot(yg, pz[0],  label="p(y|z)  (multimodal)")
        ax1.plot(yg, px[0],  label="p(y|x)  (unimodal)")
        ax1.plot(yg, pxz[0], label="p(y|x,z) (resolved)", lw=2.2)
        ax1.axvline(float(y[i]), color="k", ls="--", lw=1, label="true y")
        ax1.set_title("posteriors over y for one sample"); ax1.set_xlabel("y"); ax1.legend()

        fig.tight_layout()
        fig.savefig("prior_winding_ring.png", dpi=130)
        print("saved figure -> prior_winding_ring.png")
    except ImportError:
        pass


# ---------------------------------------------------------------------------- #
#                    plotting all the relevant distributions                   #
# ---------------------------------------------------------------------------- #

def plot_distributions(x_obs=None, y_star=None, z_star=None, *,
                       seed=3, m=4, kappa=4.0, kappa_z=1.0, r0=2.0, sigma=0.15):
    """
    Plot the eight distributions of the prior-winding wrapped-phase ring.

    Standalone (numpy + matplotlib only) so it runs without torch; it mirrors the
    generative model of PriorWindingRing exactly:

        z      ~ vM(0, kappa_z)
        y | z  ~ (1/m) sum_k vM( (z + 2*pi*k)/m , kappa )
        x | y  ~ N( r0*[cos(y), sin(y)], sigma^2 I_2 )

    Parameters
    ----------
    x_obs : array-like (2,), optional
        Observation to condition on. If None, a representative x* is drawn from
        the model using `seed`. When supplied it overrides the drawn x*.
    y_star, z_star : float, optional
        Ground-truth latents for the prior-conditioned panels (p(y|z*), p(x|z*),
        p(y|x*,z*)). Each defaults to the drawn representative sample; pass your
        own to inspect a specific datapoint. (If you pass x_obs from real data,
        pass its y_star/z_star too so the conditioning panels stay coherent.)

    Panels (note the modality is *inverted* relative to the aliased sibling:
    the multimodality now lives in generation, p(y|z) / p(x|z), while every
    inference posterior is unimodal):

        p(z)        p(y|z*)      p(x|y*)      p(x|z*)
        p(x)        p(y|x*)      p(z|x*)      p(y|x*,z*)
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import gridspec

    TWO_PI = 2 * np.pi
    wrap = lambda a: (a + np.pi) % TWO_PI - np.pi

    # --------------------------------------------------------------------------- #
    #  Grids
    # --------------------------------------------------------------------------- #
    G = 720                                   # fine angular grid (y, z; 1-D panels)
    ang = np.linspace(-np.pi, np.pi, G)
    dang = ang[1] - ang[0]

    D = 200                                   # x-grid resolution (2-D panels)
    R = r0 + 5 * sigma
    gx = np.linspace(-R, R, D)
    XX, YY = np.meshgrid(gx, gx)
    xy = np.stack([XX.ravel(), YY.ravel()], -1).astype(np.float32)   # (P, 2)

    G2 = 512                                  # angular grid for the 2-D marginal integrals
    ang2 = np.linspace(-np.pi, np.pi, G2)
    dang2 = ang2[1] - ang2[0]

    # --------------------------------------------------------------------------- #
    #  Helpers
    # --------------------------------------------------------------------------- #
    def vm_unnorm(theta, mu, kap):
        """von Mises shape exp(kap cos(theta - mu)); kap=0 -> uniform."""
        return np.exp(kap * np.cos(theta - mu))

    def norm1d(p):
        return p / (p.sum() * dang)

    def p_y_given_z(y_grid, z_scalar):
        """Mixture prior p(y|z): m von Mises bumps at (z + 2*pi*k)/m (unnormalized shape)."""
        out = np.zeros_like(y_grid, dtype=float)
        for k in range(m):
            out += vm_unnorm(y_grid, (z_scalar + TWO_PI * k) / m, kappa)
        return out / m

    def ring_gaussian(xy, ring_ang):
        """N(x; r0*[cos,sin](ring_ang), sigma^2 I) up to a constant.  Emission winds
        ONCE: the ring position is the angle itself, not m*angle.
        ring_ang: (A,)  ->  returns (P, A) float32."""
        cx = (r0 * np.cos(ring_ang)).astype(np.float32)
        cy = (r0 * np.sin(ring_ang)).astype(np.float32)
        d2 = (xy[:, 0:1] - cx[None, :]) ** 2 + (xy[:, 1:2] - cy[None, :]) ** 2
        return np.exp(-0.5 * d2 / np.float32(sigma ** 2))

    # --------------------------------------------------------------------------- #
    #  Representative sample (any latent / observation not supplied is drawn)
    # --------------------------------------------------------------------------- #
    rng = np.random.default_rng(seed)
    if z_star is None:
        z_star = rng.vonmises(0.0, kappa_z) if kappa_z > 0 else rng.uniform(-np.pi, np.pi)
    if y_star is None:
        k0 = int(rng.integers(0, m))
        y_star = wrap(rng.vonmises((z_star + TWO_PI * k0) / m, kappa))
    if x_obs is None:
        x_star = r0 * np.array([np.cos(y_star), np.sin(y_star)]) + sigma * rng.standard_normal(2)
    else:
        x_star = np.asarray(x_obs, dtype=float).reshape(2)

    phi_star = np.arctan2(x_star[1], x_star[0])          # observed angle (the single y x points to)
    rho_star = np.linalg.norm(x_star)
    kx_star = r0 * rho_star / sigma ** 2                 # likelihood concentration on y
    prior_modes = wrap((z_star + TWO_PI * np.arange(m)) / m)   # m centers of p(y|z*)

    # --------------------------------------------------------------------------- #
    #  Shared prior matrices  P[z_i, y_j] = p(y_j | z_i)
    # --------------------------------------------------------------------------- #
    P = np.zeros((G, G))                                 # cols on fine y-grid
    P2 = np.zeros((G, G2))                               # cols on coarse y-grid (for 2-D integrals)
    for k in range(m):
        mu_k = (ang[:, None] + TWO_PI * k) / m           # (G,1) center as function of z=ang[i]
        P  += np.exp(kappa * np.cos(ang[None, :]  - mu_k))
        P2 += np.exp(kappa * np.cos(ang2[None, :] - mu_k))
    P  /= m;  P2 /= m
    P  = P  / (P.sum(1, keepdims=True)  * dang)          # rows -> p(y|z_i)
    P2 = P2 / (P2.sum(1, keepdims=True) * dang2)

    wz_fine = norm1d(vm_unnorm(ang, 0.0, kappa_z))       # p(z)
    PY_fine = norm1d((P  * wz_fine[:, None]).sum(0) * dang)   # marginal p(y), fine grid
    PY2     = (P2 * wz_fine[:, None]).sum(0); PY2 /= PY2.sum()

    G_ring = ring_gaussian(xy, ang2)                     # (P, G2), winds once

    # --------------------------------------------------------------------------- #
    #  The eight distributions
    # --------------------------------------------------------------------------- #
    p_z   = wz_fine                                              # 1) p(z)
    p_y_z = norm1d(p_y_given_z(ang, z_star))                     # 2) p(y|z*)  MULTIMODAL
    p_x_y = ring_gaussian(xy, np.array([y_star]))[:, 0].reshape(D, D)  # 3) p(x|y*)

    wy = p_y_given_z(ang2, z_star); wy /= wy.sum()
    p_x_z = (G_ring * wy[None, :]).sum(1).reshape(D, D)          # 4) p(x|z*)  m blobs

    p_x = (G_ring * PY2[None, :]).sum(1).reshape(D, D)           # 5) p(x)

    lik_y  = np.exp(kx_star * np.cos(ang - phi_star))            # single-mode likelihood
    p_y_x  = norm1d(lik_y * PY_fine)                            # 6) p(y|x*)  UNIMODAL

    hz = (P * lik_y[None, :]).sum(1) * dang                      # int p(x|y) p(y|z) dy, per z
    p_z_x = norm1d(vm_unnorm(ang, 0.0, kappa_z) * hz)            # 7) p(z|x*)  UNIMODAL

    p_y_xz = norm1d(lik_y * p_y_given_z(ang, z_star))            # 8) p(y|x*,z*)  resolved

    # a few samples to overlay on p(x)
    N = 4000
    zz = rng.vonmises(0.0, kappa_z, N) if kappa_z > 0 else rng.uniform(-np.pi, np.pi, N)
    kk = rng.integers(0, m, N)
    yy = wrap(rng.vonmises((zz + TWO_PI * kk) / m, kappa))
    xs = r0 * np.stack([np.cos(yy), np.sin(yy)], -1) + sigma * rng.standard_normal((N, 2))

    # --------------------------------------------------------------------------- #
    #  Plotting
    # --------------------------------------------------------------------------- #
    ACC = "#3b6ea5"; ACC2 = "#c25b3a"; TRUE = "#111111"; ALIAS = "#888888"
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
            ax.axvline(true, color=TRUE, ls="--", lw=1.4,
                       label=("sample" if sample_label is None else sample_label))
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks(PI_TICKS); ax.set_xticklabels(PI_LABELS)
        ax.set_yticks([]); ax.set_title(title, fontsize=12); ax.margins(y=0.05)

    def field(ax, F, title, pts=None, pt_styles=None):
        F = F / F.max()
        ax.imshow(F, origin="lower", extent=[-R, R, -R, R], cmap="magma",
                  aspect="equal", vmin=0, vmax=1)
        th = np.linspace(-np.pi, np.pi, 256)
        ax.plot(r0 * np.cos(th), r0 * np.sin(th), color="w", lw=0.6, alpha=0.25)
        if pts is not None:
            for (px, py), st in zip(pts, pt_styles):
                ax.scatter([px], [py], **st)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=12)

    fig = plt.figure(figsize=(15, 7.6))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.32, wspace=0.18)

    # ---- row 0: generative / forward ----
    ax = fig.add_subplot(gs[0, 0])
    angular(ax, p_z, r"$p(z)$  — prior on top latent", color=ACC2, true=z_star, sample_label="z sample")
    ax.legend(fontsize=8, loc="upper right", frameon=False)

    ax = fig.add_subplot(gs[0, 1])
    angular(ax, p_y_z, r"$p(y\mid z^\star)$  — $m$-modal prior", color=ACC2,
            true=y_star, marks=prior_modes, mark_label="prior modes", sample_label="y sample")
    ax.legend(fontsize=8, loc="upper right", frameon=False)

    ax = fig.add_subplot(gs[0, 2])
    center = (r0 * np.cos(y_star), r0 * np.sin(y_star))
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
    angular(ax, p_y_x, r"$p(y\mid x^\star)$  — unimodal",
            true=y_star, marks=prior_modes, mark_label="prior modes", sample_label="true y")
    ax.legend(fontsize=8, loc="upper right", frameon=False)

    ax = fig.add_subplot(gs[1, 2])
    angular(ax, p_z_x, r"$p(z\mid x^\star)$  — unimodal",
            color=ACC2, true=z_star, sample_label="true z")
    ax.legend(fontsize=8, loc="upper right", frameon=False)

    ax = fig.add_subplot(gs[1, 3])
    angular(ax, p_y_xz, r"$p(y\mid x^\star,z^\star)$  — resolved",
            true=y_star, marks=prior_modes, mark_label="prior modes", sample_label="true y")
    ax.legend(fontsize=8, loc="upper right", frameon=False)

    fig.suptitle(
        rf"Prior-winding ring   $m={m}$, $\kappa={kappa}$, $\kappa_z={kappa_z}$, "
        rf"$r_0={r0}$, $\sigma={sigma}$   "
        rf"(x* drawn from y*={y_star:+.2f}, z*={z_star:+.2f})",
        fontsize=13, y=0.99,
    )
    return fig