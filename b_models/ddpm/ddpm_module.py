import numpy as np
import torch
import torch.nn as nn


# ------------------------------ unconditional diffusion model ------------------------------ #
class DDPM(nn.Module):
    def __init__(
        self,
        denoiser,
        noise_schedule: str = "cosine_in_alpha_bar",
        parameterization: str = "noise",
        sigma_minmax: tuple[float, float] = (0.0001, 0.9999),
        timestep_dist: str = "uniform",
        num_timesteps: int = 1000,
        reduction: str = "mean",
        weighted_mse: bool = False,
    ):
        super(DDPM, self).__init__()
        self.denoiser = denoiser
        self.num_timesteps = num_timesteps
        self.register_buffer("timesteps", torch.linspace(start=0, end=1, steps=self.num_timesteps))
        self.timestep_dist = timestep_dist
        self.sigma_minmax = sigma_minmax
        self.define_noise_schedule(noise_schedule)
        self.reduction = reduction
        self.parameterization = parameterization
        self.weighted_mse = weighted_mse

    def broadcast(self, x, like):
        return x.view(-1, *((1,) * (len(like.shape) - 1)))

    def scale_to_minus_one_to_one(self, x):
        # according to the DDPMs paper, normalization seems to be crucial to train reverse process network
        return x * 2 - 1

    def reverse_scale_to_zero_to_one(self, x):
        return (x + 1) * 0.5

    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def set_schedules_from_sigma(self, sigmas):
        sqrt_one_minus_alpha_bars = sigmas
        one_minus_alpha_bars = sqrt_one_minus_alpha_bars**2
        alpha_bars = 1 - one_minus_alpha_bars
        sqrt_alpha_bars = torch.sqrt(alpha_bars)
        alphas = torch.ones_like(alpha_bars)
        for i in range(1, len(alphas)):
            alphas[i] = alpha_bars[i] / alpha_bars[i - 1]
        alphas[0] = alphas[1] - (alphas[2] - alphas[1])  # linearly extrapolate
        alphas = alphas
        sqrt_alphas = torch.sqrt(alphas)
        betas = 1 - alphas
        sqrt_betas = torch.sqrt(betas)

        log_snr = torch.log(alpha_bars) - torch.log1p(-alpha_bars)

        # register buffers
        self.register_buffer("alphas", alphas)
        self.register_buffer("sqrt_alphas", sqrt_alphas)
        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_betas", sqrt_betas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", sqrt_alpha_bars)
        self.register_buffer("one_minus_alpha_bars", one_minus_alpha_bars)
        self.register_buffer("sqrt_one_minus_alpha_bars", sqrt_one_minus_alpha_bars)
        self.register_buffer("log_snr", log_snr)

        # posterior variance: beta_tilde_t = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t
        alpha_bars_prev = torch.cat([torch.tensor([1.0]), alpha_bars[:-1]])
        posterior_variance = (1.0 - alpha_bars_prev) / (1.0 - alpha_bars) * betas
        self.register_buffer("posterior_variance", posterior_variance)
        # clamp the first entry since at t=0, alpha_bar_prev=1 gives variance=0
        posterior_variance_clipped = posterior_variance.clone()
        posterior_variance_clipped[0] = posterior_variance[1]  # avoid log(0)
        self.register_buffer("sqrt_posterior_variance", torch.sqrt(posterior_variance_clipped))

    def get_cosine_in_alpha_bar(self, timesteps, s=0.008):
        """cosine schedule for alpha bars"""
        sigma_min, sigma_max = self.sigma_minmax
        f_t = torch.cos((timesteps + s) / (1 + s) * torch.pi / 2) ** 2
        f_0 = np.cos(s / (1 + s) * np.pi / 2) ** 2
        alpha_bars = sigma_min + (sigma_max - sigma_min) * (f_t / f_0)
        return alpha_bars

    def get_inverted_alpha_bars(self, timesteps, s=0.008, warp_a=1):
        """inverted alpha bars, with optional warping. To emphasize the
        middle noise levels more, decrease the value of warp_a from 1 to 0.
        """
        sigma_min, sigma_max = self.sigma_minmax
        f_0 = np.cos(s / (1 + s) * np.pi / 2) ** 2
        # f_t = torch.cos((timesteps + s) / (1 + s) * torch.pi / 2) ** 2
        ab = (1 + s) * (2 / np.pi) * torch.arccos(torch.sqrt(f_0 * timesteps)) - s
        ab = sigma_min + (sigma_max - sigma_min) * ab
        # u = torch.clamp(u, 0, 1)

        from scipy.special import betainc

        ab_warped = torch.from_numpy(betainc(warp_a, warp_a, ab.numpy())).to(ab.dtype)
        return ab_warped

    def define_noise_schedule(self, noise_schedule):
        """define beta schedule depending on the distribution of the noise (as defined by the std))"""
        # from the SODA paper: diffusion models commonly set the variance of the additive noise term ε to follow either
        # cosine [69], sigmoid [70] or linear [5] decay schedules.

        if noise_schedule == "cosine_in_alpha_bar":
            # using the schedule outlined in the Nichol and Dhariwal (2021) paper
            s = 0.008
            alpha_bars = self.get_cosine_in_alpha_bar(self.timesteps, s)
            sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)
            self.set_schedules_from_sigma(sqrt_one_minus_alpha_bars)

        elif noise_schedule == "inverted_cosine_in_alpha_bar":
            s = 0.008
            alpha_bars = self.get_inverted_alpha_bars(self.timesteps, s)
            sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)
            self.set_schedules_from_sigma(sqrt_one_minus_alpha_bars)

        elif noise_schedule == "linear_in_alpha_bar":
            sigma_min, sigma_max = self.sigma_minmax
            alpha_bars = torch.linspace(sigma_max, sigma_min, self.num_timesteps)
            sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)
            self.set_schedules_from_sigma(sqrt_one_minus_alpha_bars)

        elif noise_schedule == "reverse_linear_in_alpha_bar":
            sigma_min, sigma_max = self.sigma_minmax
            alpha_bars = torch.linspace(sigma_min, sigma_max, self.num_timesteps)
            sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)
            self.set_schedules_from_sigma(sqrt_one_minus_alpha_bars)

        elif noise_schedule == "linear_in_beta":
            sigma_min, sigma_max = self.sigma_minmax
            beta_min = 1 - sigma_max ** (1 / self.num_timesteps)
            beta_max = 1 - sigma_min ** (1 / self.num_timesteps)
            betas = torch.linspace(beta_min, beta_max, self.num_timesteps)
            alpha_bars = torch.cumprod(1 - betas, dim=0)
            sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)
            self.set_schedules_from_sigma(sqrt_one_minus_alpha_bars)

        else:
            raise ValueError(f"Unknown noise_schedule: {noise_schedule}, expected 'cosine_in_alpha_bar'")

    def make_noisy(self, x_zeros, t):
        """perturb x_0 into x_t (i.e., take x_0 samples into forward diffusion kernels)"""
        epsilon = torch.randn_like(x_zeros)

        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, t, x_zeros.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, t, x_zeros.shape)

        noisy_sample = x_zeros * sqrt_alpha_bar + epsilon * sqrt_one_minus_alpha_bar

        return noisy_sample.detach(), epsilon

    def sample_timesteps(self, B, num_timesteps, min_timestep=0, dist="uniform", device=None):
        """
        Sample timesteps from different probability distributions
        """
        # Generate random samples from a uniform distribution
        uniform_samples = torch.rand(size=(B,), device=device)

        if dist == "linearly_increasing":
            # Apply a transformation to bias towards larger values
            biased_samples = (uniform_samples**0.5) * (num_timesteps - min_timestep) + min_timestep
        elif dist == "exponentially_increasing":
            # Apply an exponential transformation to bias towards larger values
            biased_samples = (uniform_samples**0.3) * (num_timesteps - min_timestep) + min_timestep
        elif dist == "hump":
            # Apply a transformation to create a hump shape
            biased_samples = (uniform_samples**0.7) * (num_timesteps - min_timestep) + min_timestep
        elif dist == "linear_then_uniform":
            # Split point at 1/4 of the range
            # This corresponds to a PDF that increases linearly from 0 to C over the first 1/4,
            # then stays constant at C for the remaining 3/4.
            # The fraction of mass in the linear part is F1 = 1/7.
            f1 = 1.0 / 7.0
            mask = uniform_samples < f1

            total_range = num_timesteps - min_timestep

            # Linear part: [0, f1] -> [min, min + 0.25 * range]
            samples_linear = torch.sqrt(uniform_samples / f1) * (total_range * 0.25) + min_timestep

            # Uniform part: [f1, 1] -> [min + 0.25 * range, max]
            samples_uniform = (uniform_samples - f1) / (1.0 - f1) * (total_range * 0.75) + (
                min_timestep + total_range * 0.25
            )

            biased_samples = torch.where(mask, samples_linear, samples_uniform)
        else:  # default to uniform distribution
            biased_samples = uniform_samples * (num_timesteps - min_timestep) + min_timestep

        # Convert to integer timesteps
        biased_timesteps = torch.clip(biased_samples.to(torch.long), 0, num_timesteps - 1)
        return biased_timesteps


    def forward(self, x_zeros: torch.Tensor):
        """forward pass of the model"""

        B, *_ = x_zeros.shape  # batch size

        # (1) randomly choose diffusion time-step
        t_tensor = self.sample_timesteps(B, self.num_timesteps, dist=self.timestep_dist, device=x_zeros.device)

        # (2) forward diffusion process: perturb x_zeros with fixed variance schedule
        x_zeros = x_zeros.detach().requires_grad_(True)
        noisy_x, epsilon = self.make_noisy(x_zeros, t_tensor)
        noisy_x = noisy_x.detach().requires_grad_(True)

        gamma = self.extract(self.one_minus_alpha_bars, t_tensor, x_zeros.shape)

        if self.parameterization == "noise":
            # estimate the noise epsilon: predict epsilon (noise) given perturbed data at diffusion-timestep t.
            uncond_noise_estimate = self.denoiser(noisy_x, t_tensor)  # = score of p(x_t|x_t+1)

            target = epsilon
            prediction = uncond_noise_estimate

            # mse weight
            weight = 1 / (2 * gamma * (1-gamma))

        elif self.parameterization == "velocity1":
            pred_velocity_unguided = self.denoiser(noisy_x, t_tensor)

            target = (1-gamma).sqrt() * epsilon - gamma.sqrt() * x_zeros
            prediction = pred_velocity_unguided

            # mse weight
            weight = 1 / (2 * gamma)

        elif self.parameterization == "velocity2":
            pred_velocity_unguided = self.denoiser(noisy_x, t_tensor)

            target = -x_zeros / (1-gamma).sqrt() + epsilon / gamma.sqrt()
            prediction = pred_velocity_unguided

            weight = 1 / (2 * (1-gamma))

        return target, prediction, weight

    def compute_loss(self, x_zeros):
        target, prediction, weight = self.forward(x_zeros)

        if self.weighted_mse:
            mse_loss = (prediction - target).pow(2) * weight
        else:
            mse_loss = (prediction - target).pow(2)

        mse_loss = (
            mse_loss.mean() if self.reduction == "mean" 
            else mse_loss.mean(dim=0).sum()
        )
        return mse_loss
        