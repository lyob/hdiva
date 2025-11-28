import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.autograd import grad
import math
import numpy as np
from b_models.ddpm.ddpm_module import DDPM

# ------------------------------ inference class ----------------------------- #
class DDPM_Inference(DDPM):
    def __init__(
            self, 
            ddpm,
            config,
        ):
        super(DDPM_Inference, self).__init__(
            ddpm.denoiser, 
            config.noise_schedule,
            config.sigma_minmax,
            config.timestep_dist,
            config.num_timesteps,
            config.reduction,
        )
        self.config = config
        self.img_C = config.num_channels
        self.img_H = config.image_dim
        self.img_W = config.image_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_times = config.num_timesteps
    
    # ----------------------- inference ----------------------- #
    def denoise_at_t(self, x_t, pred_epsilon, timestep):
        sqrt_alpha_prod = self.extract(self.sqrt_alphas_prod, timestep, x_t.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep, x_t.shape)
        x0_hat = 1 / sqrt_alpha_prod * (x_t - sqrt_one_minus_alpha_bar * pred_epsilon)
        return x0_hat
    
    def predict_mu_t(self, x_t, pred_epsilon, timestep):
        alpha = self.extract(self.alphas, timestep, x_t.shape)
        sqrt_alpha = self.extract(self.sqrt_alphas, timestep, x_t.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep, x_t.shape)
        
        # denoise at time t, utilizing predicted noise
        mu_t_minus_1 = 1 / sqrt_alpha * (x_t - (1-alpha)/sqrt_one_minus_alpha_bar * pred_epsilon)
        return mu_t_minus_1 
    
    def reverse_one_timestep_denoiser_only(self, t):
        '''denoiser only'''
        B, *_ = self.x_t.shape  # batch size
        timestep = torch.Tensor([t]).repeat_interleave(B, dim=0).to(torch.long).to(self.device)
        
        if t > 1:
            z = torch.randn_like(self.x_t, device=self.device)
        else:
            z = torch.zeros_like(self.x_t, device=self.device)
        
        pred_epsilon = self.denoiser(self.x_t, timestep)  # = score of p(x_t|x_t+1)

        # use the total predicted noise to denoise the image
        transition_mean = self.predict_mu_t(self.x_t, pred_epsilon, timestep)
        
        sqrt_beta = self.extract(self.sqrt_betas, timestep, self.x_t.shape)
        
        # and then add noise to this image again
        self.x_t = transition_mean + sqrt_beta*z
    
    @torch.no_grad()
    def unconditional_sample_denoiser_only(self, N):
        '''sampling using only the denoiser'''
        # start from random noise vector, x_0 (for simplicity, x_T declared as x_t instead of x_T)
        self.denoiser.eval()
        
        # start from random noise vector, x_T
        self.x_t = torch.randn((N, self.img_C, self.img_H, self.img_W), device=self.device)
        
        for t in range(self.n_times-1, -1, -1):
            self.reverse_one_timestep_denoiser_only(t)