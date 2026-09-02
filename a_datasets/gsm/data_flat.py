from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse
from scipy.integrate import trapezoid
from scipy.stats import lognorm
from torch.utils.data import Dataset


class GSMDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 1000,
        sigma_z: float = 0.5,
        lambdas: Sequence[float] = (1.0, 0.25),
        A: Optional[np.ndarray] = None,
        sigma_x: float = 0.1,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        gsm = HierarchicalGSM(sigma_z=sigma_z, lambdas=lambdas, A=A, sigma_x=sigma_x, seed=seed)

        z, y, x = gsm.sample_joint(num_samples)

        self.data_z = torch.tensor(z, dtype=torch.float32)       # (n,)
        self.data_y = torch.tensor(y, dtype=torch.float32)       # (n, k)
        self.targets = torch.tensor(x, dtype=torch.float32)      # (n, d)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            (
                self.data_z[idx],
                self.data_y[idx],
            ),
            self.targets[idx],
        )


def load_gsm_dataset(
    num_samples: int = int(2e3),
    sigma_z: float = 0.5,
    lambdas: Sequence[float] = (1.0, 0.25),
    A: Optional[np.ndarray] = None,
    sigma_x: float = 0.1,
    seed: int = 42,
) -> Tuple[torch.Tensor, Tuple[np.ndarray, np.ndarray]]:
    """Create a GSM dataset and return (x_tensor, (z, y)) arrays."""
    dataset = GSMDataset(
        num_samples=num_samples,
        sigma_z=sigma_z,
        lambdas=lambdas,
        A=A,
        sigma_x=sigma_x,
        seed=seed,
    )
    z = np.array(dataset.data_z)
    y = np.array(dataset.data_y)
    return dataset.targets, (z, y)


class HierarchicalGSM:
    """
    Hierarchical Gaussian Scale Mixture: z -> y -> x

        z ~ LogNormal(0, sigma_z^2),                    z in R_+
        y | z ~ N(0, z * Sigma_y),                       y in R^k
        x | y ~ N(A y, sigma_x^2 I),                     x in R^d

    Provides:
    - Joint sampler
    - Closed-form p(y | x, z)         (linear-Gaussian Bayes)
    - Closed-form p(x | z)            (Gaussian, marginalizing y)
    - Posterior p(z | x) on a grid    (1D quadrature)
    - First two moments of p(y | x)   (law of total covariance)

    Defaults match the toy 2D example: sigma_z = 0.5, lambdas = (1.0, 0.25),
    A = I, sigma_x = 0.1.
    """
    def __init__(self, sigma_z=0.5, lambdas=(1.0, 0.25), A=None, sigma_x=0.1, seed=None):
        self.sigma_z = float(sigma_z)
        self.Sigma_y = np.diag(np.asarray(lambdas, dtype=float))
        self.k = self.Sigma_y.shape[0]
        self.A = np.eye(self.k) if A is None else np.asarray(A, dtype=float)
        self.d = self.A.shape[0]
        self.sigma_x = float(sigma_x)
        self.rng = np.random.default_rng(seed)

        # Cached quantities
        self._Sy_inv = np.linalg.inv(self.Sigma_y)
        self._AtA = self.A.T @ self.A
        self._M = self.A @ self.Sigma_y @ self.A.T  # for p(x | z)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample_z(self, n):
        # LogNormal(0, sigma_z^2): log(z) ~ N(0, sigma_z^2)
        return self.rng.lognormal(mean=0.0, sigma=self.sigma_z, size=n)

    def sample_y_given_z(self, z):
        z = np.atleast_1d(z)
        L = np.linalg.cholesky(self.Sigma_y)
        eps = self.rng.standard_normal((z.shape[0], self.k))
        return np.sqrt(z)[:, None] * (eps @ L.T)

    def sample_x_given_y(self, y):
        n = y.shape[0]
        noise = self.sigma_x * self.rng.standard_normal((n, self.d))
        return y @ self.A.T + noise

    def sample_joint(self, n):
        """Returns z (n,), y (n, k), x (n, d)."""
        z = self.sample_z(n)
        y = self.sample_y_given_z(z)
        x = self.sample_x_given_y(y)
        return z, y, x

    # ------------------------------------------------------------------
    # Closed-form conditionals
    # ------------------------------------------------------------------
    def posterior_y_given_xz(self, x, z):
        """
        p(y | x, z) = N(mu, Sigma).
        x: (n, d), z: (n,) or scalar broadcastable to n.
        Returns (mu (n, k), Sigma (n, k, k)).
        """
        x = np.atleast_2d(x)
        z = np.broadcast_to(np.atleast_1d(z), (x.shape[0],))
        n = x.shape[0]

        mus = np.empty((n, self.k))
        Sigmas = np.empty((n, self.k, self.k))
        for i in range(n):
            prec = self._Sy_inv / z[i] + self._AtA / self.sigma_x ** 2
            Sigma = np.linalg.inv(prec)
            mu = Sigma @ self.A.T @ x[i] / self.sigma_x ** 2
            mus[i] = mu
            Sigmas[i] = Sigma
        return mus, Sigmas

    def log_p_x_given_z(self, x, z):
        """
        Log evidence log p(x | z), where x | z ~ N(0, z*M + sigma_x^2 I).
        x: (n, d), z: (n_z,). Returns (n, n_z).
        """
        x = np.atleast_2d(x)
        z = np.atleast_1d(z)
        n, n_z = x.shape[0], z.shape[0]
        out = np.empty((n, n_z))
        eye = np.eye(self.d)
        for j in range(n_z):
            cov = z[j] * self._M + self.sigma_x ** 2 * eye
            sign, logdet = np.linalg.slogdet(cov)
            inv = np.linalg.inv(cov)
            quad = np.einsum("ni,ij,nj->n", x, inv, x)
            out[:, j] = -0.5 * (self.d * np.log(2 * np.pi) + logdet + quad)
        return out

    def log_p_z(self, z):
        """Log-density of LogNormal(0, sigma_z^2)."""
        z = np.asarray(z)
        log_z = np.log(z)
        return (-log_z - np.log(self.sigma_z) - 0.5 * np.log(2 * np.pi)
                - 0.5 * (log_z / self.sigma_z) ** 2)

    def posterior_z_given_x(self, x, z_grid=None, n_grid=200, n_sigma=5):
        """
        p(z | x) discretized on a log-spaced grid.
        Returns (z_grid (n_z,), weights (n, n_z)) where weights sum to 1 over z.
        Weights already include the trapezoidal dz factor, so sum_j w_ij = 1
        approximates the integral of the posterior density.
        """
        if z_grid is None:
            # cover ~ [exp(-n_sigma * sigma_z), exp(+n_sigma * sigma_z)]
            t = np.linspace(-n_sigma * self.sigma_z, n_sigma * self.sigma_z, n_grid)
            z_grid = np.exp(t)

        log_lik = self.log_p_x_given_z(x, z_grid)          # (n, n_z)
        log_prior = self.log_p_z(z_grid)                   # (n_z,)
        log_post = log_lik + log_prior[None, :]
        log_post -= log_post.max(axis=1, keepdims=True)
        w = np.exp(log_post)

        # Trapezoidal weights for the log-spaced grid
        dz = np.gradient(z_grid)
        w = w * dz[None, :]
        w /= w.sum(axis=1, keepdims=True)
        return z_grid, w

    def posterior_y_given_x_moments(self, x, z_grid=None, n_grid=200):
        """
        First two moments of p(y | x) by 1D quadrature over z.
        Returns (mean (n, k), cov (n, k, k)).
        """
        x = np.atleast_2d(x)
        z_grid, w = self.posterior_z_given_x(x, z_grid=z_grid, n_grid=n_grid)
        n, n_z = x.shape[0], z_grid.shape[0]

        # mu(z_j) and Sigma(z_j) depend only on z_j, modulo a linear x term
        # Precompute for each z_j
        Sigma_z = np.empty((n_z, self.k, self.k))
        AtSigma_over_sx2 = np.empty((n_z, self.k, self.d))
        for j in range(n_z):
            prec = self._Sy_inv / z_grid[j] + self._AtA / self.sigma_x ** 2
            Sigma_j = np.linalg.inv(prec)
            Sigma_z[j] = Sigma_j
            AtSigma_over_sx2[j] = Sigma_j @ self.A.T / self.sigma_x ** 2

        # mu_all[i, j] = AtSigma_over_sx2[j] @ x[i]
        mu_all = np.einsum("jkd,id->ijk", AtSigma_over_sx2, x)        # (n, n_z, k)

        # Marginal mean
        mean = np.einsum("ij,ijk->ik", w, mu_all)                      # (n, k)

        # Cov(y|x) = E[Sigma_z] + E[mu mu^T] - E[mu] E[mu]^T
        E_Sigma = np.einsum("ij,jkl->ikl", w, Sigma_z)                 # (n, k, k)
        E_mumuT = np.einsum("ij,ijk,ijl->ikl", w, mu_all, mu_all)      # (n, k, k)
        outer = np.einsum("ik,il->ikl", mean, mean)                    # (n, k, k)
        cov = E_Sigma + E_mumuT - outer
        return mean, cov
    

def plot_distributions(sigma_z=0.5, lambdas=(1.0, 0.25),
                            A=None, sigma_x=0.3, seed=0):
    """
    Visualize all distributions in the hierarchical GSM:
    p(z), p(y|z), p(x|y), p(x|z),
    p(x), p(z|x), p(y|x,z), p(y|x).
    """    
    
    def gauss_ellipse(ax, mu, Sigma, color, n_sigmas=(1.0, 2.0), label=None, lw=1.5, alpha=1.0):
        """Draw confidence ellipses for a 2D Gaussian."""
        vals, vecs = np.linalg.eigh(Sigma)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        for i, ns in enumerate(n_sigmas):
            ell = Ellipse(xy=mu, width=2*ns*np.sqrt(vals[0]),
                        height=2*ns*np.sqrt(vals[1]), angle=angle,
                        facecolor='none', edgecolor=color, lw=lw, alpha=alpha,
                        label=label if i == 0 else None)
            ax.add_patch(ell)
    
    
    # def main():
    # sigma_x bumped to 0.3 (from the toy default 0.1) so that the posterior
    # over y at a single x varies visibly with z — at sigma_x = 0.1, x is so
    # informative that p(y|x,z) is essentially flat in z.
    model = HierarchicalGSM(sigma_z=sigma_z, lambdas=lambdas, A=A, sigma_x=sigma_x, seed=seed)
    rng = np.random.default_rng(1)

    z_vals = [0.3, 1.0, 3.0]
    z_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(z_vals)))

    x_vals = np.array([[0.3, 0.1], [1.0, 0.3], [2.5, 0.8]])
    x_colors = plt.cm.plasma(np.linspace(0.15, 0.75, len(x_vals)))

    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5))

    # --- p(z) ----------------------------------------------------------
    ax = axes[0, 0]
    zg = np.linspace(0.01, 5, 600)
    ax.plot(zg, lognorm.pdf(zg, s=model.sigma_z, scale=1.0), 'C0', lw=2)
    ax.fill_between(zg, 0, lognorm.pdf(zg, s=model.sigma_z, scale=1.0), alpha=0.25)
    for zv, c in zip(z_vals, z_colors):
        ax.axvline(zv, color=c, lw=1, ls='--', alpha=0.8)
    ax.set_xlim(0, 5)
    ax.set_xlabel('z'); ax.set_ylabel('density')
    ax.set_title(r'$p(z) = \mathrm{LogNormal}(0, \sigma_z^2)$')

    # --- p(y|z) --------------------------------------------------------
    ax = axes[0, 1]
    for zv, c in zip(z_vals, z_colors):
        gauss_ellipse(ax, np.zeros(2), zv * model.Sigma_y, c, label=f'z = {zv}')
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-2.5, 2.5); ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.5, alpha=0.4); ax.axvline(0, color='gray', lw=0.5, alpha=0.4)
    ax.set_xlabel(r'$y_1$'); ax.set_ylabel(r'$y_2$')
    ax.set_title(r'$p(y \mid z) = \mathcal{N}(0, z\, \Sigma_y)$')
    ax.legend(fontsize=8, loc='upper right')

    # --- p(x|y) --------------------------------------------------------
    ax = axes[0, 2]
    y_vals = np.array([[-1.5, 0.4], [1.2, -0.3], [0.0, 0.0]])
    y_colors = plt.cm.cool(np.linspace(0.1, 0.8, len(y_vals)))
    for yv, c in zip(y_vals, y_colors):
        mu = model.A @ yv
        gauss_ellipse(ax, mu, model.sigma_x**2 * np.eye(2), c,
                        n_sigmas=(1, 2, 3), label=f'y = ({yv[0]:+.1f}, {yv[1]:+.1f})')
        ax.plot(*mu, 'x', color=c, markersize=8, mew=2)
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-1.2, 1.2); ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.5, alpha=0.4); ax.axvline(0, color='gray', lw=0.5, alpha=0.4)
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
    ax.set_title(r'$p(x \mid y) = \mathcal{N}(Ay, \sigma_x^2 I)$')
    ax.legend(fontsize=7, loc='upper right')

    # --- p(x|z) --------------------------------------------------------
    ax = axes[0, 3]
    M = model.A @ model.Sigma_y @ model.A.T
    for zv, c in zip(z_vals, z_colors):
        gauss_ellipse(ax, np.zeros(2), zv*M + model.sigma_x**2*np.eye(2), c, label=f'z = {zv}')
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-2.5, 2.5); ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.5, alpha=0.4); ax.axvline(0, color='gray', lw=0.5, alpha=0.4)
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
    ax.set_title(r'$p(x \mid z) = \mathcal{N}(0, zM + \sigma_x^2 I)$')
    ax.legend(fontsize=8, loc='upper right')

    # --- p(x) ----------------------------------------------------------
    ax = axes[1, 0]
    n_samp = 20000
    z_s, y_s, x_s = model.sample_joint(n_samp)
    sc = ax.scatter(x_s[:, 0], x_s[:, 1], s=2, alpha=0.25,
                    c=np.log(z_s), cmap='viridis', rasterized=True)
    # Overlay markers for the x_vals we use later
    for xv, c in zip(x_vals, x_colors):
        ax.plot(*xv, '*', color=c, markersize=12, mec='black', mew=0.6)
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-2.5, 2.5); ax.set_aspect('equal')
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
    ax.set_title(r'$p(x)$ — samples, colored by $\log z$')
    cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(r'$\log z$', fontsize=9)

    # --- p(z|x) --------------------------------------------------------
    ax = axes[1, 1]
    zg = np.linspace(0.01, 5, 600)
    ax.plot(zg, lognorm.pdf(zg, s=model.sigma_z, scale=1.0), 'k--', lw=1, alpha=0.5,
            label='prior p(z)')
    for xv, c in zip(x_vals, x_colors):
        log_lik = model.log_p_x_given_z(xv[None, :], zg)[0]
        log_post = log_lik + model.log_p_z(zg)
        log_post -= log_post.max()
        post = np.exp(log_post)
        post /= trapezoid(post, zg)
        ax.plot(zg, post, color=c, lw=2,
                label=f'x=({xv[0]:.1f}, {xv[1]:.1f})')
    ax.set_xlim(0, 5)
    ax.set_xlabel('z'); ax.set_ylabel('density')
    ax.set_title(r'$p(z \mid x)$')
    ax.legend(fontsize=7, loc='upper right')

    # --- p(y|x,z) ------------------------------------------------------
    ax = axes[1, 2]
    x_focus = np.array([1.0, 0.3])
    for zv, c in zip(z_vals, z_colors):
        mu, Sig = model.posterior_y_given_xz(x_focus[None, :], np.array([zv]))
        gauss_ellipse(ax, mu[0], Sig[0], c, label=f'z = {zv}')
        ax.plot(*mu[0], 'o', color=c, markersize=7, mec='black', mew=0.6)
    ax.plot(*x_focus, 'kx', markersize=11, mew=2, label='x = (1.0, 0.3)')
    ax.set_xlim(-0.5, 2.0); ax.set_ylim(-0.8, 1.2); ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.5, alpha=0.4); ax.axvline(0, color='gray', lw=0.5, alpha=0.4)
    ax.set_xlabel(r'$y_1$'); ax.set_ylabel(r'$y_2$')
    ax.set_title(r'$p(y \mid x, z)$ at $x = (1.0, 0.3)$')
    ax.legend(fontsize=7, loc='upper right')

    # --- p(y|x) (mixture) ---------------------------------------------
    ax = axes[1, 3]
    # Sample y | x via the model
    z_grid, w = model.posterior_z_given_x(x_focus[None, :])
    w = w[0]
    n_post = 5000
    z_idx = rng.choice(len(z_grid), size=n_post, p=w)
    z_draws = z_grid[z_idx]
    mu_all, Sig_all = model.posterior_y_given_xz(np.tile(x_focus, (n_post, 1)), z_draws)
    # Sample y from each Gaussian (use Cholesky for speed)
    L_all = np.linalg.cholesky(Sig_all)              # (n, 2, 2)
    eps = rng.standard_normal((n_post, 2))
    y_draws = mu_all + np.einsum('nij,nj->ni', L_all, eps)
    ax.scatter(y_draws[:, 0], y_draws[:, 1], s=3, alpha=0.25, c='C2', rasterized=True)
    # Moment-matched Gaussian for comparison
    mean, cov = model.posterior_y_given_x_moments(x_focus[None, :])
    gauss_ellipse(ax, mean[0], cov[0], 'k', n_sigmas=(1, 2),
                    label='Gaussian moment match', lw=1.5)
    ax.plot(*mean[0], 'k+', markersize=11, mew=2, label=r'$\mathbb{E}[y \mid x]$')
    ax.plot(*x_focus, 'rx', markersize=11, mew=2, label='x = (1.0, 0.3)')
    ax.set_xlim(-0.5, 2.0); ax.set_ylim(-0.8, 1.2); ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.5, alpha=0.4); ax.axvline(0, color='gray', lw=0.5, alpha=0.4)
    ax.set_xlabel(r'$y_1$'); ax.set_ylabel(r'$y_2$')
    ax.set_title(r'$p(y \mid x)$ at $x = (1.0, 0.3)$ — mixture samples')
    ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    # out = 'hgsm_distributions.png'
    # plt.savefig(out, dpi=130, bbox_inches='tight')
    # print(f"Saved {out}")