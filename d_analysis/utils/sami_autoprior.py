'''Analyses of autoprior / SAMI-autoprior models (AutoPrior_Inference objects) on 64x64 grayscale celeba.

All functions take images in [0, 1] (as returned by load_celeba_old_data) and rescale them to [-1, 1]
internally, matching how the models were trained.
'''
import math

import matplotlib.pyplot as plt
import numpy as np
import torch


# --------------------------------- helpers ---------------------------------- #
def noise_level_grid(ap, num: int = 50):
    '''noise levels evenly spaced in log SNR over the range of the training schedule, from clean to noisy.

    The encoder takes no timestep, so noise levels between the discrete training timesteps are fine;
    even spacing in log SNR resolves the clean end, which an even grid in timesteps samples very coarsely.
    Returns (log_snr, alpha_bar), each of shape (num,).
    '''
    schedule_log_snr = torch.log(ap.alpha_bars) - torch.log1p(-ap.alpha_bars)
    log_snr = torch.linspace(schedule_log_snr.max().item(), schedule_log_snr.min().item(), num, device=ap.device)
    return log_snr, torch.sigmoid(log_snr)  # SNR = alpha_bar / (1 - alpha_bar)


def make_noisy(x0: torch.Tensor, alpha_bar: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    '''x0 in [-1, 1]; same forward process as training'''
    eps = torch.randn(x0.shape, generator=generator, device=x0.device)
    return alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * eps


def batches(x: torch.Tensor, batch_size: int):
    for i in range(0, len(x), batch_size):
        yield x[i:i + batch_size]


# ------------------------ 1. posterior variance vs SNR ----------------------- #
@torch.no_grad()
def posterior_variance_vs_snr(ap, images: torch.Tensor, num_levels: int = 50, batch_size: int = 256, seed: int = 0):
    '''encoder posterior q(z|x_t) = N(mu_t, diag(var_t)) evaluated on noisy test images across noise levels.

    Returns (numpy arrays, T = num_levels, D = latent dims):
        log_snr:  (T,) evenly spaced, clean to noisy
        var:      (T, D) posterior variance, averaged over images
        mu_var:   (T, D) variance of the posterior mean across images (how much the axis varies with the image)
    '''
    ap.eval()
    g = torch.Generator(device=ap.device).manual_seed(seed)
    log_snr, alpha_bars = noise_level_grid(ap, num_levels)
    D = ap.encoder.latent_dims
    var = torch.zeros(num_levels, D, device=ap.device)
    mu_sum = torch.zeros(num_levels, D, device=ap.device)
    mu_sq_sum = torch.zeros(num_levels, D, device=ap.device)

    for x in batches(images, batch_size):
        x0 = ap.scale_to_minus_one_to_one(x.to(ap.device))
        for i, alpha_bar in enumerate(alpha_bars):
            _, mu_t, var_t = ap.encoder(make_noisy(x0, alpha_bar, g))
            var[i] += var_t.sum(0)
            mu_sum[i] += mu_t.sum(0)
            mu_sq_sum[i] += mu_t.pow(2).sum(0)

    n = len(images)
    mu_mean = mu_sum / n
    return {
        'log_snr': log_snr.cpu().numpy(),
        'var': (var / n).cpu().numpy(),
        'mu_var': (mu_sq_sum / n - mu_mean.pow(2)).cpu().numpy(),
    }


# ------------------------------ 2. interpolation ----------------------------- #
@torch.no_grad()
def encode_mean(ap, images: torch.Tensor) -> torch.Tensor:
    '''posterior mean of clean images in [0, 1]'''
    ap.eval()
    _, mu, _ = ap.encoder(ap.scale_to_minus_one_to_one(images.to(ap.device)))
    return mu


def interpolate_latents(z_a: torch.Tensor, z_b: torch.Tensor, num_steps: int = 9, method: str = 'linear') -> torch.Tensor:
    '''z_a, z_b: (D,) -> (num_steps, D)'''
    alphas = torch.linspace(0, 1, num_steps, device=z_a.device)[:, None]
    if method == 'linear':
        return (1 - alphas) * z_a + alphas * z_b
    elif method == 'slerp':
        omega = torch.arccos(torch.clamp(torch.dot(z_a / z_a.norm(), z_b / z_b.norm()), -1, 1))
        return (torch.sin((1 - alphas) * omega) * z_a + torch.sin(alphas * omega) * z_b) / torch.sin(omega)
    raise ValueError(f"unknown method: {method}, expected 'linear' or 'slerp'")


def decode_latents(ap, z: torch.Tensor, seed: int = 0, share_noise: bool = True) -> torch.Tensor:
    '''conditional sampling from latents z (N, D), same update as AutoPrior_Inference.reverse_one_timestep.

    share_noise=True uses the same initial noise and per-step noise for every row, so differences between
    the outputs come only from the latents (needed to judge the smoothness of an interpolation).
    Returns images in [0, 1], shape (N, C, H, W).
    '''
    ap.eval()
    N = z.shape[0]
    shape = (1 if share_noise else N, ap.img_C, ap.img_H, ap.img_W)
    g = torch.Generator(device=ap.device).manual_seed(seed)

    def noise():
        return torch.randn(shape, generator=g, device=ap.device).expand(N, *shape[1:])

    z = z.to(ap.device)
    x_t = noise().clone()
    for t in range(ap.n_times - 1, -1, -1):
        timestep = torch.full((N,), t, device=ap.device, dtype=torch.long)

        # guidance: score of the encoder posterior at the fixed latent
        x_in = x_t.detach().requires_grad_(True)
        _, mu_t, var_t = ap.encoder(x_in)
        log_p_z = ap.compute_log_posterior_vectorized(mu_t, var_t, z)
        z_score = torch.autograd.grad(log_p_z.sum(), x_in)[0]

        with torch.no_grad():
            pred_epsilon = ap.denoiser(x_in, timestep)
            alpha = ap.extract(ap.alphas, timestep, x_in.shape)
            sqrt_alpha = ap.extract(ap.sqrt_alphas, timestep, x_in.shape)
            mean = ap.predict_mu_t(x_in, pred_epsilon, timestep) + (1 - alpha) / sqrt_alpha * z_score
            if t > 1:
                x_t = mean + ap.extract(ap.sqrt_betas, timestep, x_in.shape) * noise()
            else:
                x_t = mean

    return ap.reverse_scale_to_zero_to_one(x_t.detach()).clamp(0, 1).cpu()


def interpolate_images(ap, img_a: torch.Tensor, img_b: torch.Tensor, num_steps: int = 9,
                       method: str = 'linear', seed: int = 0) -> torch.Tensor:
    '''img_a, img_b: (1, 1, H, W) in [0, 1]. Returns decoded frames (num_steps, 1, H, W) in [0, 1].'''
    mu = encode_mean(ap, torch.cat([img_a, img_b]))
    z = interpolate_latents(mu[0], mu[1], num_steps, method)
    return decode_latents(ap, z, seed=seed, share_noise=True)


# ----------------------------- latent traversals ----------------------------- #
def encode_means(ap, images: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    '''posterior means of many clean images in [0, 1], (N, D)'''
    return torch.cat([encode_mean(ap, x) for x in batches(images, batch_size)])


def traverse_axes(ap, image: torch.Tensor, axes, values: torch.Tensor, seed: int = 0):
    '''sweep individual latent axes of one image and decode.

    image:  (1, 1, H, W) in [0, 1]
    axes:   K latent axes to sweep, one at a time (the other axes stay at the image's posterior mean)
    values: (Q, K) values to set for each axis
    All K*Q latents plus the unmodified reconstruction are decoded in one batch with shared noise,
    so differences within and across rows come only from the swept axis.
    Returns traversals (K, Q, 1, H, W) and the reconstruction (1, 1, H, W), in [0, 1].
    '''
    mu = encode_mean(ap, image)[0]
    K, Q = len(axes), values.shape[0]
    z = mu.repeat(K * Q + 1, 1)
    for k, d in enumerate(axes):
        z[k * Q:(k + 1) * Q, d] = values[:, k].to(z.device)
    frames = decode_latents(ap, z, seed=seed, share_noise=True)
    return frames[:-1].view(K, Q, *frames.shape[1:]), frames[-1:]


# ---------------------- 3. per-axis guidance norm vs SNR --------------------- #
def guidance_norm_per_axis(ap, images: torch.Tensor, num_levels: int = 50, batch_size: int = 32,
                           dim_chunk: int = 32, seed: int = 0, axes=None):
    '''E ||grad_{x_t} log q(z_d | x_t)||^2 for every latent axis d (or only `axes`), across noise levels.

    z ~ q(z|x_0) is sampled from the clean-image posterior (as in training and sampling). Because the posterior
    is diagonal, log q(z|x_t) = sum_d log q(z_d|x_t), so the term for axis d is that axis's guidance field.

    axes: optional subset of latent axes to compute per-axis norms for (cost scales with the number of axes).

    Returns (numpy arrays, T = num_levels, D = number of axes computed):
        log_snr:          (T,) evenly spaced, clean to noisy
        sigma_sq:         (T,) 1 - alpha_bar
        sq_norm:          (T, D) per-axis squared guidance norm, averaged over images
        total_sq_norm:    (T,) squared norm of the full guidance field (all axes together; includes cross terms)
        info_density:     (T, D) 0.5 * sigma_sq * sq_norm = -dI(z_d; x_t)/dlogSNR under the model's posterior
        axes:             (D,) the latent axis each column corresponds to
    '''
    ap.eval()
    g = torch.Generator(device=ap.device).manual_seed(seed)
    log_snr, alpha_bars = noise_level_grid(ap, num_levels)
    axes = torch.arange(ap.encoder.latent_dims) if axes is None else torch.as_tensor(axes).long().flatten()
    K = len(axes)
    sq_norm = torch.zeros(num_levels, K, device=ap.device)
    total_sq_norm = torch.zeros(num_levels, device=ap.device)

    for x in batches(images, batch_size):
        x0 = ap.scale_to_minus_one_to_one(x.to(ap.device))
        with torch.no_grad():
            z_sample, _, _ = ap.encoder(x0)
        for i, alpha_bar in enumerate(alpha_bars):
            x_t = make_noisy(x0, alpha_bar, g).requires_grad_(True)
            _, mu_t, var_t = ap.encoder(x_t)
            var_t = var_t.clamp(min=1e-6)  # same floor as compute_log_posterior_vectorized
            log_p_per_dim = -0.5 * ((z_sample - mu_t).pow(2) / var_t + torch.log(2 * math.pi * var_t))  # (B, D)

            total = torch.autograd.grad(log_p_per_dim.sum(), x_t, retain_graph=True)[0]
            total_sq_norm[i] += total.flatten(1).pow(2).sum(1).sum()

            # one backward per axis, batched over a chunk of axes with is_grads_batched
            for c0 in range(0, K, dim_chunk):
                chunk = axes[c0:c0 + dim_chunk]
                k = len(chunk)
                grad_outputs = torch.zeros(k, *log_p_per_dim.shape, device=ap.device)
                grad_outputs[torch.arange(k), :, chunk] = 1.0
                grads = torch.autograd.grad(log_p_per_dim, x_t, grad_outputs, retain_graph=c0 + k < K, is_grads_batched=True)[0]
                sq_norm[i, c0:c0 + k] += grads.flatten(2).pow(2).sum(2).sum(1)  # (k, B, ...) -> (k,)

    n = len(images)
    sigma_sq = 1 - alpha_bars
    sq_norm = sq_norm / n
    return {
        'log_snr': log_snr.cpu().numpy(),
        'sigma_sq': sigma_sq.cpu().numpy(),
        'sq_norm': sq_norm.cpu().numpy(),
        'total_sq_norm': (total_sq_norm / n).cpu().numpy(),
        'info_density': (0.5 * sigma_sq[:, None] * sq_norm).cpu().numpy(),
        'axes': axes.numpy(),
    }


def per_axis_information(result) -> np.ndarray:
    '''integrate info_density over log SNR -> approximate I(z_d; x_0) in nats per axis, (D,)'''
    order = np.argsort(result['log_snr'])
    return np.trapz(result['info_density'][order], result['log_snr'][order], axis=0)


def centroid_log_snr(result) -> np.ndarray:
    '''information-weighted mean log SNR of each axis, (D,): where along the noise axis the axis's information sits.
    More robust than the peak for broad or two-humped curves.'''
    order = np.argsort(result['log_snr'])
    ls, density = result['log_snr'][order], result['info_density'][order]
    return np.trapz(density * ls[:, None], ls, axis=0) / np.trapz(density, ls, axis=0)


def noise_tracking_axes(result, frac: float = 0.5) -> np.ndarray:
    '''axes whose guidance is still near its maximum at pure noise: they respond to the noise level itself,
    not image content, so their integrated "information" is not meaningful. Boolean mask, (D,)'''
    density = result['info_density']
    return density[-1] > frac * density.max(0)


def most_informative_axes(result, n: int = 5, noise_frac: float | None = 0.1) -> np.ndarray:
    '''indices of the n axes with the largest per-axis information, most informative first.

    noise_frac: exclude axes whose guidance at pure noise is above this fraction of their maximum. Partly
    noise-tracking axes get an inflated "information" from the noisy end, so 0.1 is stricter than the 0.5
    used to flag the extreme ones. None keeps every axis.
    '''
    info = per_axis_information(result)
    if noise_frac is not None:
        info = np.where(noise_tracking_axes(result, frac=noise_frac), -np.inf, info)
    # results computed for a subset of axes store which axis each column is; return axis numbers, not columns
    axis_ids = np.asarray(result.get('axes', np.arange(len(info))))
    return axis_ids[np.argsort(-info)[:n]]


def peak_log_snr(values: np.ndarray, log_snr_values: np.ndarray) -> np.ndarray:
    '''log SNR at which each axis's curve peaks, (T, D) -> (D,).

    Refined below the grid spacing by fitting a parabola through the grid maximum and its two neighbours;
    peaks on the edge of the grid (or without a concave neighbourhood) stay at the grid point.
    '''
    T = len(log_snr_values)
    i = np.argmax(values, axis=0)
    peak = log_snr_values[i].astype(float)
    interior = (i > 0) & (i < T - 1)
    ii, dd = i[interior], np.where(interior)[0]
    x1, x2, x3 = log_snr_values[ii - 1], log_snr_values[ii], log_snr_values[ii + 1]
    y1, y2, y3 = values[ii - 1, dd], values[ii, dd], values[ii + 1, dd]
    # vertex of the parabola through (x1, y1), (x2, y2), (x3, y3); works for uneven spacing too
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    b = (x3**2 * (y1 - y2) + x2**2 * (y3 - y1) + x1**2 * (y2 - y3)) / denom
    concave = a < 0
    vertex = -b[concave] / (2 * a[concave])
    lo, hi = np.minimum(x1, x3)[concave], np.maximum(x1, x3)[concave]
    peak[dd[concave]] = np.clip(vertex, lo, hi)
    return peak


# ---------------------------------- plotting --------------------------------- #
def plot_axes_vs_snr(values: np.ndarray, log_snr_values: np.ndarray, order: np.ndarray | None = None,
                     normalize: bool = True, title: str = '', ax=None, cmap: str = 'viridis', log_color: bool = False):
    '''heatmap of a (T, D) per-axis quantity: rows are latent axes (sorted by `order`), columns noise levels.
    Noise increases to the right.'''
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    v = values.T  # (D, T)
    if order is not None:
        v = v[order]
    if normalize:
        v = v / (v.max(axis=1, keepdims=True) + 1e-12)
    if log_color:
        v = np.log10(v + 1e-12)
    im = ax.imshow(v, aspect='auto', cmap=cmap, interpolation='nearest',
                   extent=[log_snr_values[0], log_snr_values[-1], v.shape[0], 0])
    ax.set(xlabel='log SNR  (noise increases →)', ylabel='latent axis (sorted)', title=title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    return ax


# ------------------------ top axes: extremes of one image -------------------- #
@torch.no_grad()
def encode_posteriors(ap, images: torch.Tensor, batch_size: int = 256):
    '''posterior means and variances of clean images in [0, 1], each (N, D), on cpu'''
    ap.eval()
    mus, variances = [], []
    for x in batches(images, batch_size):
        _, mu, var = ap.encoder(ap.scale_to_minus_one_to_one(x.to(ap.device)))
        mus.append(mu.cpu()); variances.append(var.cpu())
    return torch.cat(mus), torch.cat(variances)


def top_axis_extremes(ap, images: torch.Tensor, idx: int, method: str = 'mse', K: int = 6,
                      lo: float = 0.02, hi: float = 0.98, num_ref: int = 2000, candidates=None,
                      guidance_result=None, noise_frac: float | None = 0.5, seed: int = 0,
                      show_recon: bool = True, shared_diff_scale: bool = True, save_path: str | None = None):
    '''for one image, rank latent axes by `method`, then show the image decoded with each of the top K axes
    set to its `lo` and `hi` percentile (across reference images) and the difference between the two.

    images:     test images in [0, 1], (N, 1, H, W); images[idx] is the one shown, images[:num_ref] set the
                percentiles and the "variance" score
    method:     "mse"      - MSE between the lo and hi images of this image (decodes every candidate axis)
                "variance" - latent SNR on clean images: across-image variance of the posterior mean / mean
                             posterior variance
                "guidance" - integrated ½σ²E‖g_d‖² (per_axis_information); pass guidance_result (e.g. from
                             guidance_norm_per_axis on the same checkpoint) or it is computed on images[:256]
    candidates: axes to consider (default all). Mainly useful to limit the cost of "mse".
    noise_frac: for "guidance", exclude axes with more than this fraction of their peak guidance at pure noise
                (the noise-level axes, whose "information" is meaningless); None keeps all
    Returns a dict with axes, scores, mse (between lo and hi images), frames (K, 2, 1, H, W), recon, values, fig.
    '''
    mus, variances = encode_posteriors(ap, images[:num_ref])
    D = mus.shape[1]
    candidates = np.arange(D) if candidates is None else np.asarray(candidates)
    q = torch.tensor([lo, hi])
    image = images[idx:idx + 1].clone()

    def decode_pairs(axes):
        values = torch.quantile(mus[:, axes], q, dim=0)  # (2, len(axes))
        frames, recon = traverse_axes(ap, image, axes, values, seed=seed)
        mse = (frames[:, 0] - frames[:, 1]).pow(2).flatten(1).mean(1).numpy()
        return frames, recon, values, mse

    if method == 'mse':
        frames, recon, values, mse = decode_pairs(candidates)
        top = np.argsort(-mse)[:K]
        axes, scores = candidates[top], mse[top]
        frames, values, mse = frames[top], values[:, top], mse[top]
    else:
        if method == 'variance':
            all_scores = (mus.var(0) / variances.mean(0)).numpy()
        elif method == 'guidance':
            if guidance_result is None:
                guidance_result = guidance_norm_per_axis(ap, images[:256], num_levels=60, axes=candidates)
            result_axes = np.asarray(guidance_result.get('axes', np.arange(guidance_result['info_density'].shape[1])))
            info = per_axis_information(guidance_result)
            if noise_frac is not None:
                info = np.where(noise_tracking_axes(guidance_result, frac=noise_frac), -np.inf, info)
            all_scores = np.full(D, -np.inf)
            all_scores[result_axes] = info
        else:
            raise ValueError(f"unknown method: {method}, expected 'mse', 'variance' or 'guidance'")
        cand_scores = all_scores[candidates]
        top = np.argsort(-cand_scores)[:K]
        axes, scores = candidates[top], cand_scores[top]
        frames, recon, values, mse = decode_pairs(axes)

    # plot: one row per axis; columns [recon], lo, hi, hi - lo
    diffs = frames[:, 1, 0] - frames[:, 0, 0]  # (K, H, W)
    c0 = 1 if show_recon else 0
    fig, ax = plt.subplots(len(axes), c0 + 3, figsize=(1.5 * (c0 + 3), 1.5 * len(axes)), squeeze=False)
    shared_lim = diffs.abs().max().item() + 1e-8
    score_name = {'mse': 'MSE', 'variance': 'latent SNR', 'guidance': 'nats'}[method]
    for k, d in enumerate(axes):
        if show_recon:
            ax[k, 0].imshow(recon[0, 0], cmap='gray', vmin=0, vmax=1)
        ax[k, c0].imshow(frames[k, 0, 0], cmap='gray', vmin=0, vmax=1)
        ax[k, c0 + 1].imshow(frames[k, 1, 0], cmap='gray', vmin=0, vmax=1)
        lim = shared_lim if shared_diff_scale else diffs[k].abs().max().item() + 1e-8
        ax[k, c0 + 2].imshow(diffs[k], cmap='RdBu_r', vmin=-lim, vmax=lim)
        label = f'axis {d}\n{score_name} {scores[k]:.3g}'
        if method != 'mse':
            label += f'\nMSE {mse[k]:.3g}'
        ax[k, 0].set_ylabel(label, fontsize=8)
    for a in ax.flat:
        a.set(xticks=[], yticks=[])
    if show_recon:
        ax[0, 0].set_title('recon', fontsize=8)
    ax[0, c0].set_title(f'{lo * 100:.0f}%', fontsize=8)
    ax[0, c0 + 1].set_title(f'{hi * 100:.0f}%', fontsize=8)
    ax[0, c0 + 2].set_title(f'{hi * 100:.0f}% − {lo * 100:.0f}%', fontsize=8)
    fig.suptitle(f'image {idx}: top {len(axes)} axes by {method}', fontsize=10)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return {'axes': axes, 'scores': scores, 'mse': mse, 'frames': frames, 'recon': recon,
            'values': values, 'fig': fig}


# --------------------------- two-axis joint traversal ------------------------- #
def two_axis_targets(mu: torch.Tensor, ref_mus: torch.Tensor, axis: int, steps, units: str = 'std') -> torch.Tensor:
    '''latent values for one axis.
    units="std":        steps are offsets from the image's own value, in units of the axis's std across ref images
    units="percentile": steps are percentiles in [0, 1] of the axis across ref images; None keeps the image's value
    '''
    if units == 'std':
        return mu[axis] + torch.as_tensor(steps, dtype=torch.float32) * ref_mus[:, axis].std()
    if units == 'percentile':
        return torch.stack([mu[axis] if s is None else torch.quantile(ref_mus[:, axis], torch.tensor(float(s)))
                            for s in steps])
    raise ValueError(f"unknown units: {units}, expected 'std' or 'percentile'")


def traverse_two_axes(ap, image: torch.Tensor, axis_a: int, axis_b: int, steps_a, steps_b, ref_mus: torch.Tensor,
                      units: str = 'std', cells: str = 'full', seed: int = 0):
    '''decode an image with two latent axes set jointly: grid[i, j] has axis_a at steps_a[i] and axis_b at steps_b[j].

    image:   (1, 1, H, W) in [0, 1]; all other axes stay at its posterior mean
    ref_mus: (N, D) posterior means of reference images, which set the std / percentiles
    cells:   "full" decodes every (i, j); "cross" only the first row, first column and diagonal (the others are nan)
    All cells are decoded in one batch with shared noise.
    Returns grid (len(steps_a), len(steps_b), 1, H, W) in [0, 1] and the latent values used for each axis.
    '''
    mu = encode_mean(ap, image)[0].cpu()
    values_a = two_axis_targets(mu, ref_mus, axis_a, steps_a, units)
    values_b = two_axis_targets(mu, ref_mus, axis_b, steps_b, units)
    n_a, n_b = len(values_a), len(values_b)
    if cells == 'full':
        cell_list = [(i, j) for i in range(n_a) for j in range(n_b)]
    elif cells == 'cross':
        cell_list = [(i, j) for i in range(n_a) for j in range(n_b) if i == 0 or j == 0 or i == j]
    else:
        raise ValueError(f"unknown cells: {cells}, expected 'full' or 'cross'")

    z = mu.repeat(len(cell_list), 1)
    for k, (i, j) in enumerate(cell_list):
        z[k, axis_a] = values_a[i]
        z[k, axis_b] = values_b[j]
    frames = decode_latents(ap, z, seed=seed, share_noise=True)

    grid = torch.full((n_a, n_b, *frames.shape[1:]), float('nan'))
    for k, (i, j) in enumerate(cell_list):
        grid[i, j] = frames[k]
    return grid, values_a, values_b


def two_axis_interaction(grid: torch.Tensor) -> np.ndarray:
    '''non-additivity of the two axes on the diagonal, assuming grid[0, 0] is the unchanged image (steps 0 / None).

    For each diagonal cell i >= 1: ||x_ii - x_i0 - x_0i + x_00||^2 / ||x_ii - x_00||^2.
    0 means the joint change is exactly the sum of the two single-axis changes; ~1 or more means they interact.
    '''
    x00 = grid[0, 0]
    out = []
    for i in range(1, min(grid.shape[0], grid.shape[1])):
        interaction = grid[i, i] - grid[i, 0] - grid[0, i] + x00
        out.append((interaction.pow(2).sum() / (grid[i, i] - x00).pow(2).sum().clamp(min=1e-12)).item())
    return np.array(out)


def plot_two_axis_grid(grid: torch.Tensor, axis_a: int, axis_b: int, steps_a, steps_b, units: str = 'std',
                       title: str = '', save_path: str | None = None, cell_size: float = 1.4):
    '''rows: axis_a steps (down), columns: axis_b steps (right); the diagonal is outlined.'''
    n_a, n_b = grid.shape[:2]
    fig, ax = plt.subplots(n_a, n_b, figsize=(cell_size * n_b, cell_size * n_a), squeeze=False)

    def step_label(s):
        if units == 'std':
            return f'{s:+g}σ'
        return 'own' if s is None else f'{float(s) * 100:g}%'

    for i in range(n_a):
        for j in range(n_b):
            a = ax[i, j]
            a.set(xticks=[], yticks=[])
            if torch.isnan(grid[i, j]).any():
                a.axis('off')
                continue
            a.imshow(grid[i, j, 0], cmap='gray', vmin=0, vmax=1)
            if i == j:
                for spine in a.spines.values():
                    spine.set_edgecolor('tab:red'); spine.set_linewidth(2)
        ax[i, 0].set_ylabel(step_label(steps_a[i]), fontsize=8)
    for j in range(n_b):
        ax[0, j].set_title(step_label(steps_b[j]), fontsize=8)
    fig.supylabel(f'← axis {axis_a}', fontsize=9)  # rotated 90°, so the arrow points down the rows
    fig.supxlabel(f'axis {axis_b} →', fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig
