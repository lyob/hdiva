import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributions as dist

class DDPM(nn.Module):
    def __init__(
            self, 
            denoiser, 
            scheduler, 
            kl_reduction="mean", 
            timestep_dist='uniform',
        ):
        super().__init__()
        self.denoiser = denoiser
        self.scheduler = scheduler
        
        self.timestep_dist = timestep_dist

        self.alphas_cumprod = self.scheduler.alphas_cumprod
        self.num_timesteps = self.scheduler.num_inference_steps
        self.kl_reduction = kl_reduction  # "mean" or 'sum'

    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
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
    
        # (7) estimate the noise epsilon: predict epsilon (noise) given perturbed data at diffusion-timestep t.
        pred_noise = self.denoiser(noisy_x, t_tensor).sample  # = score of p(x_t|x_t+1)
        
        # (8) set the target and prediction for the model
        target = noise
        prediction = pred_noise
        mse_weights = torch.ones_like(clean_x)
        
        return target, prediction, mse_weights
    
    def compute_loss(self, clean_x):
        target, prediction, mse_weights = self.forward(clean_x)
        # Compute the loss
        mse_loss = F.mse_loss(prediction, target, reduction=self.kl_reduction)
        weighted_mse_loss = (mse_loss * mse_weights).mean()
        return weighted_mse_loss