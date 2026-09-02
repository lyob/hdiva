r"""
Orthogonal-subspace wrapped-phase ring  (CommonCauseRing embedded in one vector)
================================================================================
The common-cause plate written as a *single* high-dimensional observation.
Instead of C separate children x_1..x_C, the latent y = (y_1..y_C) is one
higher-dimensional latent whose components each live in a separate orthogonal
2-plane of R^{2C}, and there is one observation x in R^{2C}.

Generative model:

    z          ~  vM(0, kappa_z)                                  # shared top latent
    y_c | z    ~  (1/m) sum_k vM( (z + 2*pi*k)/m , kappa )        # c = 1..C, conditionally iid
    e(y)       =  [ r0 cos y_1, r0 sin y_1, ... , r0 cos y_C, r0 sin y_C ]   # block embedding
    x          =  Q e(y) + sigma * eps,   eps ~ N(0, I_{2C}),  Q in O(2C)

Q is a fixed orthonormal frame (identity -> axis-aligned blocks; random -> the
factors are linearly entangled across all 2C observed coordinates).

Why orthogonality reproduces the plate
--------------------------------------
Because Q is orthonormal and the noise isotropic, un-rotating with x~ = Q^T x
keeps the noise isotropic and the squared error block-decomposes:

    || x - Q e(y) ||^2 = || x~ - e(y) ||^2 = sum_c || x~^(c) - r0[cos y_c, sin y_c] ||^2

so   p(x|y) = prod_c p( x~^(c) | y_c ),   x~^(c) = (Q^T x)^(c)  (the c-th 2-block of x~).

This is exactly CommonCauseRing's factorized likelihood with child x_i replaced by
the un-rotated block x~^(c).  Every closed form transfers under that substitution,
and p(z|x) concentrates with the number of orthogonal subspaces C just as it did
with the number of children.  (A non-orthogonal mixing A would give noise
covariance sigma^2 (A^T A)^{-1} in the un-mixed frame, coupling the blocks and
breaking the factorization -- orthogonality is precisely the "separate subspaces"
condition.)

Closed forms (per block, identical to CommonCauseRing on x~^(c)):
    log p(x~^(c)|y) = kappa_x cos(y - phi_c),   kappa_x = r0||x~^(c)||/sigma^2, phi_c = angle(x~^(c))
    p(x~^(c)|z)     = (1/m) sum_k I0(R_k(z)) * const,
        R_k = sqrt(kappa_x^2 + kappa^2 + 2 kappa_x kappa cos(phi_c - (z+2*pi*k)/m))
    log p(z|x)      = kappa_z cos(z) + sum_c log p(x~^(c)|z) + const

Note: with a non-trivial Q you cannot observe "a subset of coordinates" -- every
coordinate mixes all factors.  You observe the full x, un-rotate to recover all C
subspaces, and may then analyze any subset of the recovered blocks.

Identifiability: Q is recoverable only up to a per-block SO(2) (each subspace's
angular origin) and block permutation (the y_c are exchangeable).
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
#  Orthonormal frame
# --------------------------------------------------------------------------- #
def random_orthogonal(d: int) -> Tensor:
    """A (near-)Haar orthonormal d x d matrix (QR of a Gaussian, sign-fixed)."""
    a = torch.randn(d, d)
    q, r = torch.linalg.qr(a)
    return q * torch.sign(torch.diagonal(r))            # fix column signs


# --------------------------------------------------------------------------- #
#  Sampling
# --------------------------------------------------------------------------- #
def generate(
    n: int,
    n_blocks: int = 8,
    m: int = 4,
    kappa: float = 4.0,
    kappa_z: float = 0.0,
    r0: float = 2.0,
    sigma: float = 0.15,
    rotate: bool = True,
    seed: int | None = 0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sample (x, y, z, Q) from the orthogonal-subspace wrapped-phase ring.

    Returns
    -------
    x : (n, 2C)  observations in the ambient space (C = n_blocks)
    y : (n, C)   per-subspace latents (fine angles, radians in (-pi, pi])
    z : (n,)     top latent (coarse angle)
    Q : (2C, 2C) the fixed orthonormal frame used for embedding

    Note: VonMises.sample() draws from torch's *global* RNG, so `seed` is applied
    via torch.manual_seed for reproducibility (Q is drawn first, then z, y, noise).
    """
    if seed is not None:
        torch.manual_seed(seed)

    C = n_blocks
    d = 2 * C
    Q = random_orthogonal(d) if rotate else torch.eye(d)

    if kappa_z and kappa_z > 0:
        z = VonMises(torch.zeros(n), kappa_z * torch.ones(n)).sample()
    else:
        z = torch.empty(n).uniform_(-PI, PI)

    k = torch.randint(0, m, (n, C))
    mu = (z[:, None] + 2 * PI * k) / m                       # (n, C)
    y = VonMises(mu, kappa * torch.ones(n, C)).sample()      # (n, C)
    y = (y + PI) % (2 * PI) - PI

    # block embedding: interleave [cos, sin] per subspace -> (n, 2C)
    e = torch.empty(n, d)
    e[:, 0::2] = r0 * torch.cos(y)
    e[:, 1::2] = r0 * torch.sin(y)

    x = e @ Q.T + sigma * torch.randn(n, d)                  # rotate, then add isotropic noise

    return x.to(device), y.to(device), z.to(device), Q.to(device)


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #
@dataclass
class OrthogonalSubspaceRing(Dataset):
    """Map-style dataset; samples are precomputed in __init__.

    __getitem__ returns `x` of shape (2C,) by default, or a dict
    {"x":(2C,), "y":(C,), "z":()} when `return_latents=True`.
    Ground-truth latents are `.x`, `.y`, `.z`; the frame is `.Q` (2C, 2C).
    """

    n: int = 20_000
    n_blocks: int = 8
    m: int = 4
    kappa: float = 4.0
    kappa_z: float = 0.0
    r0: float = 2.0
    sigma: float = 0.15
    rotate: bool = True
    seed: int | None = 0
    return_latents: bool = False

    def __post_init__(self) -> None:
        super().__init__()
        x, y, z, Q = generate(
            self.n, self.n_blocks, self.m, self.kappa, self.kappa_z,
            self.r0, self.sigma, self.rotate, self.seed,
        )
        self.x, self.y, self.z, self.Q = x.float(), y.float(), z.float(), Q.float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        if self.return_latents:
            return {"x": self.x[i], "y": self.y[i], "z": self.z[i]}
        return self.x[i]

    # ----- frame inversion: recover the per-subspace 2-blocks ----------------- #
    def unmix(self, x: Tensor) -> Tensor:
        """x~ = Q^T x, reshaped to blocks.  x:(...,2C) -> (...,C,2)."""
        xt = x @ self.Q                                     # rows: x~ = x Q  (= Q^T x in column form)
        return xt.reshape(*xt.shape[:-1], self.n_blocks, 2)

    # ----- grids -------------------------------------------------------------- #
    def _grid(self, n_grid: int, device) -> Tensor:
        return torch.linspace(-PI, PI, n_grid, device=device)

    # ----- per-subspace posteriors over y (identical to PriorWindingRing) ----- #
    def log_likelihood_y(self, y: Tensor, xblock: Tensor) -> Tensor:
        """log p(x~^(c)|y) up to const.  y:(G,), xblock:(B,2) -> (B,G)."""
        if xblock.dim() == 1:
            xblock = xblock.unsqueeze(0)
        rho = xblock.norm(dim=-1, keepdim=True)
        phi = torch.atan2(xblock[:, 1:2], xblock[:, 0:1])
        kappa_x = self.r0 * rho / self.sigma ** 2
        return kappa_x * torch.cos(y - phi)

    def log_prior_y(self, y: Tensor, z: Tensor) -> Tensor:
        """log p(y|z) up to const.  y:(G,), z:(B,) -> (B,G).  Mixture of m bumps."""
        if z.dim() == 0:
            z = z.unsqueeze(0)
        k = torch.arange(self.m, device=y.device)
        mu = (z[:, None, None] + 2 * PI * k[None, :, None]) / self.m
        comp = self.kappa * torch.cos(y[None, None, :] - mu)
        return torch.logsumexp(comp, dim=1) - math.log(self.m)

    def posterior_y_given_x(self, x: Tensor, n_grid: int = 720):
        """Per-subspace y-posteriors from one observation.  x:(2C,) -> (C, n_grid)."""
        y = self._grid(n_grid, x.device)
        blocks = self.unmix(x)                              # (C,2)
        return y, torch.softmax(self.log_likelihood_y(y, blocks), dim=-1)

    def posterior_y_given_z(self, z: Tensor, n_grid: int = 720):
        y = self._grid(n_grid, z.device)
        return y, torch.softmax(self.log_prior_y(y, z), dim=-1)

    def posterior_y_given_xz(self, x: Tensor, z: Tensor, n_grid: int = 720):
        y = self._grid(n_grid, x.device)
        blocks = self.unmix(x)
        logp = self.log_likelihood_y(y, blocks) + self.log_prior_y(y, z)
        return y, torch.softmax(logp, dim=-1)

    # ----- top posterior over z ---------------------------------------------- #
    def log_prior_z(self, z: Tensor) -> Tensor:
        return self.kappa_z * torch.cos(z)

    def log_evidence_z(self, z: Tensor, blocks: Tensor) -> Tensor:
        """Per-subspace evidence log p(x~^(c)|z) up to const.  z:(G,), blocks:(C,2) -> (C,G).
        Exact, via the product-of-von-Mises integral (Bessel I0)."""
        if blocks.dim() == 1:
            blocks = blocks.unsqueeze(0)
        rho = blocks.norm(dim=-1, keepdim=True)
        phi = torch.atan2(blocks[:, 1:2], blocks[:, 0:1])
        kx = self.r0 * rho / self.sigma ** 2
        k = torch.arange(self.m, device=z.device)
        mu = (z[None, None, :] + 2 * PI * k[None, :, None]) / self.m   # (1, m, G)
        cosd = torch.cos(phi[:, :, None] - mu)                          # (C, m, G)
        R = torch.sqrt(kx[:, :, None] ** 2 + self.kappa ** 2
                       + 2 * kx[:, :, None] * self.kappa * cosd + 1e-12)
        log_I0 = torch.special.i0e(R).log() + R
        return torch.logsumexp(log_I0, dim=1) - math.log(self.m)        # (C, G)

    def posterior_z_given_x(self, x: Tensor, n_grid: int = 720, use_blocks: int | None = None):
        """p(z | x) on a grid.  x:(2C,) one observation.
        With a non-trivial Q you must observe the full x to un-rotate; `use_blocks`
        then restricts the analysis to the first k recovered subspaces (handy for
        watching identifiability grow with the number of factors)."""
        z = self._grid(n_grid, x.device)
        blocks = self.unmix(x)                              # (C,2)
        if use_blocks is not None:
            blocks = blocks[:use_blocks]
        logp = self.log_prior_z(z) + self.log_evidence_z(z, blocks).sum(0)
        return z, torch.softmax(logp, dim=-1)


def grid_entropy(p: Tensor, eps: float = 1e-12) -> Tensor:
    """Discrete entropy (nats) of a normalized grid distribution."""
    return -(p * (p + eps).log()).sum(dim=-1)


# --------------------------------------------------------------------------- #
#  Sanity check
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    C = 8
    ds = OrthogonalSubspaceRing(n=2_000, n_blocks=C, m=4, kappa=4.0, kappa_z=0.0,
                                sigma=0.15, r0=2.0, rotate=True, seed=0,
                                return_latents=True)
    print(f"data: x{tuple(ds.x.shape)}  ambient dim={2*C}  C={C} orthogonal subspaces  m={ds.m}")
    print(f"orthonormality ||Q^T Q - I|| = "
          f"{(ds.Q.T @ ds.Q - torch.eye(2*C)).norm().item():.2e}")

    x0, z0, y0 = ds.x[0], float(ds.z[0]), ds.y[0]
    H = lambda p: grid_entropy(p).mean().item()

    # un-rotation recovers each subspace: angle(x~^(c)) ~ y_c, and m*angle ~ z (mod 2pi)
    blocks = ds.unmix(x0)                                   # (C,2)
    phi = torch.atan2(blocks[:, 1], blocks[:, 0])
    ang_err = ((phi - y0 + PI) % (2 * PI) - PI).abs().mean().item()
    z_vote_err = (((ds.m * phi - z0 + PI) % (2 * PI) - PI).abs().mean().item())
    print(f"un-mix recovery:  mean|angle(x~^c) - y_c| = {ang_err:.3f} rad   "
          f"mean|m*angle - z*| = {z_vote_err:.3f} rad")

    print(f"\ntrue z* = {z0:+.3f}")
    print("  #subspaces   H(z | x)   (lower = more concentrated)")
    for c in [1, 2, 4, 8]:
        _, p_zc = ds.posterior_z_given_x(x0, use_blocks=c)
        print(f"     {c:>2}        {H(p_zc):+.3f}")

    _, p_yx = ds.posterior_y_given_x(x0)
    _, p_yz = ds.posterior_y_given_z(ds.z[:1])
    print(f"\nper-subspace:  H(y_c|x) = {grid_entropy(p_yx).mean().item():+.3f} (unimodal)   "
          f"H(y|z) = {grid_entropy(p_yz).mean().item():+.3f} (multimodal)")


# ---------------------------------------------------------------------------- #
#                    plotting all the relevant distributions                   #
# ---------------------------------------------------------------------------- #

def plot_distributions(x_obs=None, z_star=None, *,
                       seed=3, n_blocks=8, m=4, kappa=4.0, kappa_z=0.0,
                       r0=2.0, sigma=0.15, rotate=True, subspace_steps=(1, 2, 4, 8)):
    """
    Visualize the orthogonal-subspace embedding and the concentration of p(z|x).

    Standalone (numpy + matplotlib only).  Builds its own orthonormal frame Q and
    a small dataset for the entanglement scatters; per-subspace z-evidence is
    grid-marginalized (mirrors the exact Bessel closed form used by the class).

    Parameters
    ----------
    x_obs : array-like (2C,), optional
        One observation to condition on.  If None, a sample is drawn using `seed`.
    z_star : float, optional
        Ground-truth top latent (defaults to the drawn sample).

    Panels (2 x 3):
        observed coords (entangled)   un-rotated subspace 1 (a ring)   p(z|x; first c subspaces)
        H(z|x) vs #subspaces          prior p(z) vs posterior          one subspace: p(y|x),p(y|z),p(y|x,z)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    TWO_PI = 2 * np.pi
    wrap = lambda a: (a + np.pi) % TWO_PI - np.pi
    C = n_blocks
    d = 2 * C

    # ---- grids / helpers ----
    G = 720
    ang = np.linspace(-np.pi, np.pi, G); dang = ang[1] - ang[0]

    def vm_unnorm(theta, mu, kap):
        return np.exp(kap * np.cos(theta - mu))

    def norm1d(p):
        return p / (p.sum() * dang)

    def softmax1d(logp):
        logp = logp - logp.max()
        p = np.exp(logp)
        return p / (p.sum() * dang)

    def entropy1d(p):
        return -(p * np.log(p + 1e-300)).sum() * dang

    def p_y_given_z(y_grid, z_scalar):
        out = np.zeros_like(y_grid, dtype=float)
        for k in range(m):
            out += vm_unnorm(y_grid, (z_scalar + TWO_PI * k) / m, kappa)
        return out / m

    P = np.zeros((G, G))
    for k in range(m):
        mu_k = (ang[:, None] + TWO_PI * k) / m
        P += np.exp(kappa * np.cos(ang[None, :] - mu_k))
    P /= m; P = P / (P.sum(1, keepdims=True) * dang)
    log_prior_z = kappa_z * np.cos(ang)

    def block_log_evidence(phi_c, kx_c):
        lik = np.exp(kx_c * (np.cos(ang - phi_c) - 1.0))
        ev = (P * lik[None, :]).sum(1) * dang
        return np.log(ev + 1e-300)

    # ---- orthonormal frame + a small dataset for the scatters ----
    rng = np.random.default_rng(seed)
    if rotate:
        A = rng.standard_normal((d, d))
        Q, Rqr = np.linalg.qr(A)
        Q = Q * np.sign(np.diag(Rqr))
    else:
        Q = np.eye(d)

    def embed(yv):
        e = np.empty((*yv.shape[:-1], d))
        e[..., 0::2] = r0 * np.cos(yv)
        e[..., 1::2] = r0 * np.sin(yv)
        return e

    Nset = 3000
    z_set = rng.vonmises(0.0, kappa_z, Nset) if kappa_z > 0 else rng.uniform(-np.pi, np.pi, Nset)
    k_set = rng.integers(0, m, (Nset, C))
    y_set = wrap(rng.vonmises((z_set[:, None] + TWO_PI * k_set) / m, kappa))
    x_set = embed(y_set) @ Q.T + sigma * rng.standard_normal((Nset, d))
    xt_set = x_set @ Q                                       # un-rotated -> recovers embedding

    # ---- the conditioning sample ----
    if z_star is None:
        z_star = rng.vonmises(0.0, kappa_z) if kappa_z > 0 else rng.uniform(-np.pi, np.pi)
    if x_obs is None:
        kk = rng.integers(0, m, C)
        yv = wrap(rng.vonmises((z_star + TWO_PI * kk) / m, kappa))
        x_star = embed(yv) @ Q.T + sigma * rng.standard_normal(d)
    else:
        x_star = np.asarray(x_obs, dtype=float).reshape(d)

    blocks = (x_star @ Q).reshape(C, 2)                     # (C,2) recovered subspaces
    phi = np.arctan2(blocks[:, 1], blocks[:, 0])
    rho = np.linalg.norm(blocks, axis=1)
    kx = r0 * rho / sigma ** 2

    # ---- posteriors over z from the first c subspaces ----
    log_evs = [block_log_evidence(phi[c], kx[c]) for c in range(C)]
    H_curve = []; cum = log_prior_z.copy(); cum_posts = {}
    steps = [c for c in subspace_steps if c <= C]
    for c in range(1, C + 1):
        cum = cum + log_evs[c - 1]
        H_curve.append(entropy1d(softmax1d(cum)))
        if c in steps:
            cum_posts[c] = softmax1d(cum)
    p_z = norm1d(vm_unnorm(ang, 0.0, kappa_z))
    p_z_final = softmax1d(log_prior_z + np.sum(log_evs, axis=0))

    lik_y0 = np.exp(kx[0] * np.cos(ang - phi[0]))
    p_y_x0 = norm1d(lik_y0)
    p_y_z0 = norm1d(p_y_given_z(ang, z_star))
    p_y_xz0 = norm1d(lik_y0 * p_y_given_z(ang, z_star))

    # ---- plotting ----
    ACC = "#3b6ea5"; ACC2 = "#c25b3a"; TRUE = "#111111"; ALIAS = "#888888"
    PI_TICKS = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    PI_LABELS = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]
    cmap = plt.get_cmap("viridis")

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    # (A) two observed coordinates: factors are linearly entangled
    ax = axs[0, 0]
    ax.scatter(x_set[:, 0], x_set[:, 1], s=3, alpha=0.25, color=ACC)
    ax.scatter([x_star[0]], [x_star[1]], color="crimson", s=40, zorder=5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ttl = "observed coords $(x_1,x_2)$\n" + ("entangled (random $Q$)" if rotate else "axis-aligned ($Q=I$)")
    ax.set_title(ttl, fontsize=12)

    # (B) un-rotated subspace 1: a clean ring
    ax = axs[0, 1]
    ax.scatter(xt_set[:, 0], xt_set[:, 1], s=3, alpha=0.25, color="#2a7")
    ax.scatter([blocks[0, 0]], [blocks[0, 1]], color="crimson", s=40, zorder=5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(r"un-rotated subspace 1, $(Q^\top x)^{(1)}$" + "\nrecovers a ring", fontsize=12)

    # (C) p(z | first c subspaces) sharpening
    ax = axs[0, 2]
    for j, c in enumerate(steps):
        ax.plot(ang, cum_posts[c], lw=2.0,
                color=cmap(j / max(1, len(steps) - 1)), label=f"$c={c}$")
    ax.axvline(z_star, color=TRUE, ls="--", lw=1.4)
    ax.set_xlim(-np.pi, np.pi); ax.set_xticks(PI_TICKS); ax.set_xticklabels(PI_LABELS)
    ax.set_yticks([]); ax.legend(fontsize=8, loc="upper right", frameon=False)
    ax.set_title(r"$p(z\mid x)$ from first $c$ subspaces", fontsize=12)

    # (D) entropy vs number of subspaces
    ax = axs[1, 0]
    cc = np.arange(1, C + 1)
    ax.plot(cc, H_curve, "-o", color=ACC, ms=4)
    ax.set_xlabel("number of subspaces $c$"); ax.set_ylabel(r"$H(z\mid x)$  (nats)")
    ax.set_title("z identifiability grows with #subspaces", fontsize=12)
    ax.grid(alpha=0.25)

    # (E) prior vs full posterior
    ax = axs[1, 1]
    ax.plot(ang, p_z, color=ALIAS, lw=2, label=r"prior $p(z)$")
    ax.fill_between(ang, p_z, color=ALIAS, alpha=0.12)
    ax.plot(ang, p_z_final, color=ACC2, lw=2.2, label=r"$p(z\mid x)$ (all $C$)")
    ax.fill_between(ang, p_z_final, color=ACC2, alpha=0.15)
    ax.axvline(z_star, color=TRUE, ls="--", lw=1.4, label="true $z$")
    ax.set_xlim(-np.pi, np.pi); ax.set_xticks(PI_TICKS); ax.set_xticklabels(PI_LABELS)
    ax.set_yticks([]); ax.legend(fontsize=8, loc="upper right", frameon=False)
    ax.set_title(r"prior vs full posterior over $z$", fontsize=12)

    # (F) one subspace: per-subspace structure preserved
    ax = axs[1, 2]
    ax.plot(ang, p_y_z0, color=ACC2, lw=1.8, label=r"$p(y_1\mid z^\star)$ (multimodal)")
    ax.plot(ang, p_y_x0, color=ACC, lw=1.8, label=r"$p(y_1\mid x)$ (unimodal)")
    ax.plot(ang, p_y_xz0, color="#2a7", lw=2.2, label=r"$p(y_1\mid x,z^\star)$")
    ax.set_xlim(-np.pi, np.pi); ax.set_xticks(PI_TICKS); ax.set_xticklabels(PI_LABELS)
    ax.set_yticks([]); ax.legend(fontsize=7.5, loc="upper right", frameon=False)
    ax.set_title("one subspace — structure unchanged", fontsize=12)

    fig.suptitle(
        rf"Orthogonal-subspace ring   $C={C}$ (dim ${d}$), $m={m}$, $\kappa={kappa}$, "
        rf"$\kappa_z={kappa_z}$, $r_0={r0}$, $\sigma={sigma}$, rotate={rotate}   "
        rf"(true $z^\star={z_star:+.2f}$)",
        fontsize=12.5, y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig