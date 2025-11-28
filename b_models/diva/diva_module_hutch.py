import torch
import torch.nn.functional as F
from b_models.ddpm.ddpm_module import DDPM

class DiVA(DDPM):
    def __init__(
            self, 
            denoiser, 
            infnet, 
            noise_schedule,
            sigma_minmax: tuple[float, float] = (0.0001, .9999),
            timestep_dist: str = 'uniform',
            num_timesteps=1000,
            reduction: str = "mean",
        ):
        super(DiVA, self).__init__(
            denoiser,
            noise_schedule,
            sigma_minmax,
            timestep_dist,
            num_timesteps,
            reduction,
        )
        self.infnet = infnet
        self.register_buffer('kl', torch.tensor(0,))  # KL divergence

    def compute_kl(self, mu, logvar):
        """
        Computes KL divergence between q = N(mu1, diag(var1)) and p = N(0, I).
        Args:
            posterior_params: mean and logvar of q. Tensor of shape (B, 2*z_dim).
        Returns:
            kl: Scalar, KL divergence after reduction (sum over z_dim, mean over batch).
        """
        kl_per_dim = 0.5 * (logvar.exp() + mu.pow(2) - 1 - logvar)
        # if reduction is sum, sum over dimensions, mean over batch
        return kl_per_dim.mean() if self.reduction == "mean" else kl_per_dim.sum(dim=1).mean()
    
    def compute_guidance_score(self, x_t, mu_t, logvar_t, z0_sample, num_samples=1, use_rademacher=True):
        """Computing the guidance score using Hutchinson's trace estimator. 
        Optimized for diagonal covariance matrix in z"""
        assert x_t.requires_grad, "x_t must have requires_grad=True"

        # mean term
        var_t = logvar_t.exp().clamp(min=1e-8)
        delta = z0_sample - mu_t
        weighted_delta = delta / var_t
        mean_scalar = (mu_t * weighted_delta).sum()
        mean_grad = torch.autograd.grad(
            outputs=mean_scalar,
            inputs=x_t, 
            retain_graph=True, 
            create_graph=True
        )[0]  # shape (B, x_dim)

        # trace term
        trace_grad = 0.
        for _ in range(num_samples):
            if use_rademacher:
                # Rademacher: uniform over {-1, +1} (one of two choices)
                v = torch.randint(
                    low=0, high=2, 
                    size=logvar_t.shape, 
                    dtype=logvar_t.dtype, 
                    device=logvar_t.device
                ) * 2.0 - 1.0  # Maps {0,1} -> {-1, +1}
            else:
                # Gaussian: N(0, I)
                v = torch.randn_like(logvar_t)

            weighted_logvar = (v * logvar_t).sum()  # shape (B, z_dim)
            jvp = torch.autograd.grad(
                outputs=weighted_logvar,
                inputs=x_t, 
                retain_graph=True, 
                create_graph=True
            )[0]  # shape (B, x_dim)
            trace_grad += jvp

        logdet_term = trace_grad / num_samples
        guidance_score = mean_grad - 0.5 * logdet_term
        return guidance_score
    
    def get_z_score_and_kl(self, clean_x, noisy_x):
        mu_0, logvar_0 = self.infnet(clean_x)
        z0_sample = self.infnet.sample(mu_0, logvar_0)
        logvar_0 = logvar_0.clamp(min=-20, max=10)
        kl = self.compute_kl(mu_0, logvar_0)
        mu_t, logvar_t = self.infnet(noisy_x)
        logvar_t = logvar_t.clamp(min=-20, max=10)
        guidance_score = self.compute_guidance_score(noisy_x, mu_t, logvar_t, z0_sample)
        return guidance_score, kl

    def forward(self, clean_x):
        '''forward pass of the model. Assume the clean images have already been normalized to [-1, 1].'''
        B, *_ = clean_x.shape  # batch size

        # (1) randomly choose diffusion time-step
        t_tensor = torch.randint(low=0, high=self.num_timesteps, size=(B,), device=clean_x.device, dtype=torch.long)
        # t_tensor = self.sample_timesteps(B, self.num_timesteps, dist=self.timestep_dist, device=clean_x.device)

        # Add noise to the clean image
        # clean_x = clean_x.detach().requires_grad_(True)
        clean_x = clean_x.requires_grad_(True)
        noisy_x, noise = self.make_noisy(clean_x, t_tensor)
        noisy_x = noisy_x.requires_grad_(True)
        
        z_score, self.kl = self.get_z_score_and_kl(clean_x, noisy_x)
        
        # (5) weight the score by sqrt(1-alpha_bar)
        z_score_weight = self.extract(self.sqrt_one_minus_alpha_bars, t_tensor, clean_x.shape)

        # (6) calculate the weighted z-score
        weighted_z_score = z_score_weight * z_score
    
        # (7) estimate the noise epsilon: predict epsilon (noise) given perturbed data at diffusion-timestep t.
        pred_noise = self.denoiser(noisy_x, t_tensor)  # = score of p(x_t|x_t+1)
        pred_noise_guided = pred_noise - weighted_z_score
        
        # (8) set the target and prediction for the model
        target = noise
        prediction = pred_noise_guided
        
        return target, prediction
    
    def compute_loss(self, clean_x, kl_weight:float=1.):

        target, prediction = self.forward(clean_x)
        mse_loss = F.mse_loss(prediction, target, reduction=self.reduction)
        kl_loss = self.kl * kl_weight

        # weigh the two terms appropriately (same ratio as if they were summed)
        if self.reduction == 'mean':
            z_dim = self.infnet.latent_dim
            x_dim = clean_x.shape[1:].numel()
            kl_loss = kl_loss * (z_dim/x_dim)

        total_loss = mse_loss + kl_loss
        return total_loss, mse_loss, self.kl


