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
    
    def compute_log_posterior(self, mu, logvar, z_sample):
        z_dim = mu.shape[-1]
        var = logvar.exp().clamp(min=1e-8)
        quadratic_term = ((z_sample - mu)**2 / var).sum(dim=-1)
        constant = z_dim * torch.log(torch.tensor(2.0 * torch.pi, device=mu.device, dtype=mu.dtype))
        log_p_z = -0.5 * (constant + logvar.sum(dim=-1) + quadratic_term)
        return log_p_z

    def compute_kl(self, mu, logvar):
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
    
    def compute_score(self, log_p_z, noisy_x):
        # get the log posterior score
        rec_score = torch.autograd.grad(log_p_z, noisy_x, torch.ones_like(log_p_z), retain_graph=True, create_graph=True)[0]
        return rec_score
    
    def get_z_score_and_kl(self, clean_x, noisy_x):
        mu, logvar = self.infnet(clean_x)
        z_sample = self.infnet.sample(mu, logvar)
        mu_t, logvar_t = self.infnet(noisy_x)
        log_p_z = self.compute_log_posterior(mu_t, logvar_t, z_sample)
        kl = self.compute_kl(mu, logvar)
        score = self.compute_score(log_p_z, noisy_x)
        return score, kl

    def forward(self, clean_x):
        '''forward pass of the model. Assume the clean images have already been normalized to [-1, 1].'''
        B, *_ = clean_x.shape  # batch size

        # (1) randomly choose diffusion time-step
        t_tensor = torch.randint(low=0, high=self.num_timesteps, size=(B,), device=clean_x.device, dtype=torch.long)
        # t_tensor = self.sample_timesteps(B, self.num_timesteps, dist=self.timestep_dist, device=clean_x.device)

        # Add noise to the clean image
        clean_x = clean_x.detach().requires_grad_(True)
        # noise = torch.randn_like(clean_x)
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


