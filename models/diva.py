import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributions as dist
from models.infnet import HalfUNetInfNetNoTime, HalfUNetInfNet

class DiVA(nn.Module):
    def __init__(
            self, 
            denoiser, 
            infnet, 
            scheduler, 
            kl_reduction="mean", 
            timestep_dist='uniform',
        ):
        super().__init__()
        self.denoiser = denoiser
        self.infnet = infnet
        self.scheduler = scheduler
        
        self.kl_reduction = kl_reduction  # "mean" or 'sum'
        self.kl : torch.Tensor = torch.zeros(1,)  # KL divergence
        self.timestep_dist = timestep_dist

        self.alphas_cumprod = self.scheduler.alphas_cumprod
        self.num_timesteps = self.scheduler.num_inference_steps

    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def compute_log_posterior(self, mu, cov, z_sample):
        # Compute the log probability of z_sample under the multivariate normal distribution
        # cov is the full covariance matrix
        log_p_z = dist.MultivariateNormal(mu, cov).log_prob(z_sample)  # log q(z|x_t)
        return log_p_z
    
    def compute_log_posterior_vectorized(self, mu, var, z_sample):
        """
        Computes log probability of z_sample under q = N(mu, diag(var)).
        
        Args:
            mu: Tensor of shape (B, z_dim), mean of the distribution.
            var: Tensor of shape (B, z_dim), diagonal elements of covariance (variances).
            z_sample: Tensor of shape (B, z_dim), samples to evaluate.
        
        Returns:
            log_p_z: Tensor of shape (B,), log probabilities for each batch element.
        """
        # Ensure var is positive to avoid division by zero or log(0)
        var = torch.clamp(var, min=1e-6)
        z_dim = mu.shape[-1]  # Dimensionality of the latent space
        
        # Compute log probability terms
        # Quadratic term: sum((z - mu)^2 / var)
        quadratic = ((z_sample - mu)**2 / var).sum(dim=-1)  # Shape: (B,)
        
        # Log determinant term: z_dim * log(2pi) + sum(log(var))
        log_det = z_dim * torch.log(torch.tensor(2.0 * torch.pi, device=mu.device)) + torch.log(var).sum(dim=-1)  # Shape: (B,)

        # Log probability: -0.5 * (quadratic + log_det)
        log_p_z = -0.5 * (quadratic + log_det)  # Shape: (B,)
        return log_p_z
    
    def compute_kl(self, mu, cov):
        '''takes in a mu of size (B, z_dim) and cov of size (B, z_dim, z_dim) and computes the KL divergence wrt N(0, I)'''
        q = dist.MultivariateNormal(mu, covariance_matrix=cov)
        p = dist.MultivariateNormal(torch.zeros_like(mu), covariance_matrix=torch.vmap(torch.diag_embed)(torch.ones_like(mu)))

        kl = torch.distributions.kl.kl_divergence(q, p).mean(dim=0)
        self.kl = kl.sum() if self.kl_reduction == 'sum' else kl.mean()
        
    def compute_kl_vectorized(self, mu, var):
        """
        Computes KL divergence between q = N(mu, diag(var)) and p = N(0, I).
        
        Args:
            mu: Tensor of shape (B, z_dim), mean of q.
            var: Tensor of shape (B, z_dim), diagonal elements of covariance (variances).
        
        Returns:
            kl: Scalar, KL divergence after reduction (sum or mean over z_dim).
        """
        var = torch.clamp(var, min=1e-6)
        
        # Compute KL terms: 0.5 * (var + mu^2 - 1 - log(var))
        kl_per_dim = 0.5 * (var + mu**2 - 1 - torch.log(var))  # Shape: (B, z_dim)
        
        # Reduce over z_dim
        kl = kl_per_dim.sum(dim=1)  # Shape: (B,)
        
        # Reduce over batch
        kl = kl.mean(dim=0)  # Scalar
        
        # Apply final reduction (sum or mean over z_dim)
        self.kl = kl.sum() if self.kl_reduction == 'sum' else kl.mean()
    
    def compute_score(self, log_p_z, noisy_x):
        # get the log posterior score
        rec_score = torch.autograd.grad(log_p_z, noisy_x, torch.ones_like(log_p_z), retain_graph=True, create_graph=True)[0]
        return rec_score
    
    def get_z_score_and_kl(self, clean_x, noisy_x, t=None):
        if t is not None:
            zeros = torch.zeros_like(t)
            z_sample, mu, var = self.infnet(clean_x, zeros)
            _, mu_t, var_t = self.infnet(noisy_x, t)
        else:
            z_sample, mu, var = self.infnet(clean_x)
            _, mu_t, var_t = self.infnet(noisy_x)
        log_p_z = self.compute_log_posterior_vectorized(mu_t, var_t, z_sample)
        self.compute_kl_vectorized(mu, var)
        score = self.compute_score(log_p_z, noisy_x)
        return score
    
    def broadcast(self, x, like):
        return x.view(-1, *((1,) * (len(like.shape) - 1)))
    
    def add_noise(self, clean_x, noise, t):
        '''Add noise to the clean image using the forward diffusion kernel.'''
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        self.alphas_cumprod = self.alphas_cumprod.to(device=clean_x.device)
        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alphas_cumprod[t])
        
        # reshape to match clean_x shape
        sqrt_alpha_cumprod = self.broadcast(sqrt_alpha_cumprod, like=clean_x)
        sqrt_one_minus_alpha_cumprod = self.broadcast(sqrt_one_minus_alpha_cumprod, like=clean_x)

        noisy_x = clean_x * sqrt_alpha_cumprod + noise * sqrt_one_minus_alpha_cumprod
        return noisy_x.detach()

    def sample_timesteps(self, B, num_timesteps, min_timestep=0, dist='uniform', device=None):
        """
        Sample timesteps with a bias towards larger values.
        """
        # Generate random samples from a uniform distribution
        uniform_samples = torch.rand(size=(B,), device=device)
        
        if dist == 'linearly_increasing':
            # Apply a transformation to bias towards larger values
            biased_samples = (uniform_samples ** .5) * (num_timesteps - min_timestep) + min_timestep
        elif dist == 'exponentially_increasing':
            # Apply an exponential transformation to bias towards larger values
            biased_samples = (uniform_samples ** .3) * (num_timesteps - min_timestep) + min_timestep
        elif dist == 'hump':
            # Apply a transformation to create a hump shape
            biased_samples = (uniform_samples ** .7) * (num_timesteps - min_timestep) + min_timestep
        else:  # default to uniform distribution
            biased_samples = uniform_samples * (num_timesteps - min_timestep) + min_timestep

        # Convert to integer timesteps
        biased_timesteps = torch.clip(biased_samples.to(torch.long), 0, num_timesteps - 1)
        return biased_timesteps

    def forward(self, clean_x):
        '''forward pass of the model. Assume the clean images have already been normalized to [-1, 1].'''
        B, *_ = clean_x.shape  # batch size

        # (1) randomly choose diffusion time-step
        t_tensor = self.sample_timesteps(B, self.num_timesteps, dist=self.timestep_dist, device=clean_x.device)

        # Add noise to the clean image
        clean_x = clean_x.detach().requires_grad_(True)
        noise = torch.randn_like(clean_x)
        noisy_x = self.add_noise(clean_x, noise, t_tensor)
        noisy_x = noisy_x.detach().requires_grad_(True)
        
        if isinstance(self.infnet, HalfUNetInfNet):
            z_score = self.get_z_score_and_kl(clean_x, noisy_x, t_tensor)
        else:
            z_score = self.get_z_score_and_kl(clean_x, noisy_x)
        
        # (5) weight the score by sqrt(1-alpha_bar)
        z_score_weight = torch.sqrt(1 - self.alphas_cumprod[t_tensor])
        z_score_weight = self.broadcast(z_score_weight, like=z_score)

        # (6) calculate the weighted z-score
        weighted_z_score = z_score_weight * z_score
    
        # (7) estimate the noise epsilon: predict epsilon (noise) given perturbed data at diffusion-timestep t.
        pred_noise = self.denoiser(noisy_x, t_tensor).sample  # = score of p(x_t|x_t+1)
        pred_noise_guided = pred_noise - weighted_z_score
        
        # (8) set the target and prediction for the model
        target = noise
        prediction = pred_noise_guided
        mse_weights = torch.ones_like(clean_x)
        
        return target, prediction, mse_weights
    
    def compute_loss(self, clean_x):
        target, prediction, mse_weights = self.forward(clean_x)
        # Compute the loss
        mse_loss = F.mse_loss(prediction, target, reduction=self.kl_reduction)
        weighted_mse_loss = (mse_loss * mse_weights).mean()
        return weighted_mse_loss, self.kl