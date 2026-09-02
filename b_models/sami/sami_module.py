import torch
import torch.nn.functional as F
from typing import Literal
from torch import Tensor

from b_models.ddpm.ddpm_module import DDPM


class SAMI(DDPM):
    def __init__(
        self,
        denoiser,
        infnet,
        noise_schedule,
        parameterization: Literal[
            "noise", "velocity", "image"
        ] = "velocity", 
        sigma_minmax: tuple[float, float] = (0.0001, 0.9999),
        timestep_dist: Literal[
            "uniform", "linearly_increasing", "exponentially_increasing", "hump"
        ] = "uniform",
        num_timesteps=1000,
        reduction: Literal["mean", "sum"] = "sum",
        rate_type: Literal[
            "grad", "norm", "cumulative", "kl"
        ] = "grad",  # how to compute the rate term in the SAMI loss
        weighted_mse: bool = False,
        weighted_rate: bool = False,
    ):
        """

        Args:
            denoiser: denoiser model
            infnet: inference network
            noise_schedule: noise schedule
            sigma_minmax: min and max values of sigma
            timestep_dist: distribution of timesteps
            num_timesteps: number of timesteps
            reduction: reduction method
            use_kl: whether to use KL divergence
        """
        super(SAMI, self).__init__(
            denoiser,
            noise_schedule,
            parameterization,
            sigma_minmax,
            timestep_dist,
            num_timesteps,
            reduction,
        )
        self.infnet = infnet
        self.parameterization = parameterization
        self.rate_type = rate_type
        self.weighted_mse = weighted_mse
        self.weighted_rate = weighted_rate


    def compute_log_posterior(self, mu, logvar, z_sample):
        z_dim = mu.shape[-1]
        var = logvar.exp().clamp(min=1e-8)
        quadratic_term = ((z_sample - mu) ** 2 / var).sum(dim=-1)
        constant = z_dim * torch.log(torch.tensor(2.0 * torch.pi, device=mu.device, dtype=mu.dtype))
        log_p_z = -0.5 * (constant + logvar.sum(dim=-1) + quadratic_term)
        return log_p_z

    def compute_kl_against_zero(self, mu, logvar):
        """
        Computes KL divergence between q = N(mu1, diag(var1)) and p = N(0, I).
        Args:
            posterior_params: mean and logvar of q. Tensor of shape (B, 2*z_dim).
        Returns:
            kl: Scalar, KL divergence after reduction (sum over z_dim, mean over batch).
        """
        kl_per_dim = 0.5 * (logvar.exp().clamp(min=1e-8) + mu.pow(2) - 1 - logvar)
        # if reduction is sum, sum over dimensions, mean over batch
        return kl_per_dim.mean() if self.reduction == "mean" else kl_per_dim.sum(dim=1).mean()

    def compute_kl(self, mu0, logvar0, mu1, logvar1):
        """
        Computes KL divergence KL(p0||p1) between p0 and p1.
        p0 = N(mu0, diag(var0)) and p1 = N(mu1, diag(var1)).
        Args:
            mean of p0. Tensor of shape (B, z_dim).
            logvar of p0. Tensor of shape (B, z_dim).
            mean of p1. Tensor of shape (B, z_dim).
            logvar of p1. Tensor of shape (B, z_dim).
        Returns:
            kl: Scalar, KL divergence after reduction (sum over z_dim, reduction over batch).
        """
        kl = 0.5 * (
            logvar1 - logvar0 + (logvar0.exp() + (mu0 - mu1) ** 2) / logvar1.exp() - 1
        )
        kl = kl.sum(dim=-1)  # sum over z_dim
        return kl.mean() if self.reduction == "mean" else kl.sum()  # reduce over batch

    def compute_score_from_logp(self, log_p, noisy_x):
        # get the log posterior score
        score = torch.autograd.grad(
            outputs = log_p,
            inputs = noisy_x,
            grad_outputs = torch.ones_like(log_p),
            retain_graph = True,
            create_graph = True,
        )[0]
        return score

    def compute_score(self, mu, logvar, noisy_x):
        z_sample = self.infnet.sample(mu, logvar)
        mu_t, logvar_t = self.infnet(noisy_x)
        log_p_z = self.compute_log_posterior(mu_t, logvar_t, z_sample)
        score = self.compute_score_from_logp(log_p_z, noisy_x)
        return score

    
    def compute_score_and_rate(self, encoder, noisy_x, clean_x, gamma):
        """
        Arguments:
            encoder: the infnet encoder that maps from images to latent space parameters (mu, logvar)
            noisy_x: the perturbed image at diffusion time t (shape: B, C, H, W)
            clean_x: the original clean image (shape: B, C, H, W)
            gamma: the diffusion time-dependent scalar (shape: B,)
        Returns:
            score: ∇_{x_g} log p(z|x_g) (shape: B, C, H, W)
            rate_grad: the gradient of the SAMI rate with respect to the diffusion time gamma (shape: B, C, H, W)
        """
        # --- Standard Forward Pass ---
        # with torch.no_grad():
        clean_x = clean_x.detach()
        mu, logvar = encoder(clean_x)
        z_sample = self.infnet.sample(mu, logvar)

        # --- Score via standard backward pass ---
        mu_t, logvar_t = encoder(noisy_x)
        log_posterior = self.compute_log_posterior(mu_t, logvar_t, z_sample)
        score = self.compute_score_from_logp(log_posterior, noisy_x)

        # --- Assemble rate derivative ---
        if self.rate_type == "grad":
            # The tangent vector
            gamma = self.broadcast(gamma, like=noisy_x)
            dx_dgamma = (noisy_x - clean_x / (1 - gamma).sqrt())
            rate = -(score * dx_dgamma)
            if self.weighted_rate:
                rate_weight = 1 / (2 * gamma) 
                rate = rate_weight * rate

        elif self.rate_type == "norm":
            rate = score.norm(dim=[1, 2, 3])
        elif self.rate_type == "cumulative":
            rate = self.compute_kl(mu_t, logvar_t, mu, logvar)
        elif self.rate_type == "kl":
            rate = self.compute_kl_against_zero(mu_t, logvar_t)
        else:
            raise ValueError(f"Invalid rate_type: {self.rate_type}")

        return score, rate

    def forward(self, clean_x, unconditional=False):
        """forward pass of the model. Assume the clean images have already been normalized to [-1, 1]."""
        B, *_ = clean_x.shape  # batch size

        # (1) randomly choose diffusion time-step
        t_tensor = self.sample_timesteps(B, self.num_timesteps, dist=self.timestep_dist, device=clean_x.device)
        
        # Add noise to the clean image
        clean_x = clean_x.detach()
        noisy_x, noise = self.make_noisy(clean_x, t_tensor)
        noisy_x = noisy_x.detach().requires_grad_(True)

        gamma = self.extract(self.one_minus_alpha_bars, t_tensor, clean_x.shape)
        z_score, rate = self.compute_score_and_rate(self.infnet, noisy_x, clean_x, gamma)

        if self.parameterization == "noise":
            pred_noise = self.denoiser(noisy_x, t_tensor)  # = instantaneous entropy
            z_score_weight = gamma.sqrt()
            pred_noise_guided = pred_noise - z_score_weight * z_score

            target = noise
            prediction = pred_noise_guided

            # distortion weight
            weight = 1 / (2 * gamma * (1 - gamma))

        elif self.parameterization == "velocity1":
            pred_velocity = self.denoiser(noisy_x, t_tensor)

            z_score_weight = (gamma / (1 - gamma)).sqrt()
            weighted_z_score = z_score_weight * z_score

            pred_velocity_guided = pred_velocity - weighted_z_score

            target = (1 - gamma).sqrt() * noise - gamma.sqrt() * clean_x
            prediction = pred_velocity_guided

            # distortion weight
            weight = 1 / (2 * gamma)

        elif self.parameterization == "velocity2":
            pred_velocity = self.denoiser(noisy_x, t_tensor)
            pred_velocity_guided = pred_velocity - z_score

            target = -clean_x / (1-gamma).sqrt() + noise / gamma.sqrt()
            prediction = pred_velocity_guided

            weight = 1 / (2 * (1-gamma))

        elif self.parameterization == "image":
            z_score_weight = gamma / (1 - gamma).sqrt()
            weighted_z_score = z_score_weight * z_score

            pred_x0 = self.denoiser(noisy_x, t_tensor)
            pred_x0_guided = pred_x0 + weighted_z_score

            target = clean_x
            prediction = pred_x0_guided

            weight = 1 / (2 * gamma**2)

        else:
            raise ValueError(f"Invalid parameterization: {self.parameterization}")

        return target, prediction, weight, rate


    def compute_loss(self, clean_x, beta=1.0):
        target, prediction, weight, rate = self.forward(clean_x)
        # target and prediction shape: (B, C, H, W)
        # weight shape: (B, 1, 1, 1)
        # rate shape: (B, C, H, W)

        # Compute the MSE loss weighted by weight
        mse_loss = (prediction - target) ** 2
        if self.weighted_mse:
            mse_loss = weight * mse_loss

        # always average over batch, but sum/avg over other dimensions
        mse_loss = (
            mse_loss.mean() if self.reduction == "mean" 
            else mse_loss.mean(dim=0).sum()
        )

        if self.rate_type == "grad":
            rate_loss = (
                rate.mean() if self.reduction == "mean" 
                else rate.mean(dim=0).sum()
            )
            # reduce over spatial dimensions, then avg over batch
        elif self.rate_type == "cumulative":
            rate_loss = rate
        elif self.rate_type == "norm":
            rate_loss = (
                rate.mean() if self.reduction == "mean" 
                else rate.mean(dim=0).sum()
            )
        elif self.rate_type == "kl":
            rate_loss = rate

        rate_loss = rate_loss.mean()  # average over batch

        total_loss = mse_loss + beta * rate_loss  # average over batch

        return total_loss, mse_loss, rate_loss

    
    def compute_rate(self, clean_x):
        B, *_ = clean_x.shape  # batch size

        # (1) randomly choose diffusion time-step
        t_tensor = self.sample_timesteps(
            B, self.num_timesteps, dist=self.timestep_dist, device=clean_x.device
        )

        # Add noise to the clean image
        clean_x = clean_x.detach()
        noisy_x, noise = self.make_noisy(clean_x, t_tensor)
        noisy_x = noisy_x.detach()

        mu, logvar = self.infnet(clean_x)
        mu_t, logvar_t = self.infnet(noisy_x)

        # KL between the noisy image posterior and clean image posterior
        kl_cumulative_t0 = self.compute_kl(mu_t, logvar_t, mu, logvar)
        kl_cumulative_0t = self.compute_kl(mu, logvar, mu_t, logvar_t)
        kl_0 = self.compute_kl_against_zero(mu, logvar)
        return kl_cumulative_t0, kl_cumulative_0t, kl_0
