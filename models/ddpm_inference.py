import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.autograd import grad
import math
import numpy as np
from models.ddpm import DDPM

# ------------------------------ inference class ----------------------------- #
class DDPM_Inference(DDPM):
    def __init__(
            self, 
            denoiser,
            scheduler, 
            kl_reduction="mean"
        ):
        super(DDPM_Inference, self).__init__(
            denoiser, 
            scheduler, 
            kl_reduction
        )
        self.n_times = scheduler.num_inference_steps
        self.device = self.denoiser.device
        self.img_C = 3
        self.img_H = 64
        self.img_W = 64

        self.alphas = scheduler.alphas
        self.alphas_cumprod = scheduler.alphas_cumprod
        self.betas = scheduler.betas
        self.set_schedules_from_alphas()
    
    # ----------------------------------- utils ---------------------------------- #
    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
    def set_schedules_from_alphas(self):
        '''assume you have alphas, alphas_cumprod and betas'''
        self.alphas = self.alphas.to(self.device)
        self.alpha_cumprod = self.alphas_cumprod.to(self.device)
        self.betas = self.betas.to(self.device)
        self.sqrt_alphas = torch.sqrt(self.alphas).to(self.device)
        self.sqrt_alphas_prod = (self.alphas_cumprod ** 0.5).to(self.device)
        self.sqrt_betas = torch.sqrt(self.betas).to(self.device)

        self.one_minus_alphas_prod = (1 - self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_prod = torch.sqrt(self.one_minus_alphas_prod).to(self.device)

    # ----------------------- inference ----------------------- #
    def denoise_at_t(self, pred_epsilon, timestep):
        sqrt_alpha_prod = self.extract(self.sqrt_alphas_prod, timestep, self.x_t.shape)
        sqrt_one_minus_alpha_prod = self.extract(self.sqrt_one_minus_alphas_prod, timestep, self.x_t.shape)
        x0_hat = 1 / sqrt_alpha_prod * (self.x_t - sqrt_one_minus_alpha_prod * pred_epsilon)
        return x0_hat
    
    def predict_mu_t(self, x_t, pred_epsilon, timestep):
        alpha = self.extract(self.alphas, timestep, self.x_t.shape)
        sqrt_alpha = self.extract(self.sqrt_alphas, timestep, self.x_t.shape)
        sqrt_one_minus_alpha_prod = self.extract(self.sqrt_one_minus_alphas_prod, timestep, self.x_t.shape)
        
        # denoise at time t, utilizing predicted noise
        mu_t_minus_1 = 1 / sqrt_alpha * (x_t - (1-alpha)/sqrt_one_minus_alpha_prod * pred_epsilon)
        return mu_t_minus_1 
    
    def reverse_one_timestep_denoiser_only(self, noisy_x, t):
        '''denoiser only'''
        B, *_ = noisy_x.shape  # batch size
        if t > 1:
            z = torch.randn_like(noisy_x, device=self.device)
        else:
            z = torch.zeros_like(noisy_x, device=self.device)
        
        timestep = torch.Tensor([t]).repeat_interleave(B, dim=0).to(torch.long).to(self.device)

        pred_epsilon = self.denoiser(noisy_x, timestep).sample  # = score of p(x_t|x_t+1)

        # use the total predicted noise to denoise the image
        mu_t_minus_1 = self.predict_mu_t(noisy_x, pred_epsilon, timestep)
        
        sigma = self.extract(self.sqrt_betas, timestep, noisy_x.shape)
        
        # and then add noise to this image again
        self.x_t = mu_t_minus_1 + sigma*z
        # return x_tm1
    
    # def reverse_one_timestep_denoiser_only(self, t):
    #     '''denoiser only'''
    #     B, *_ = self.x_t.shape  # batch size
    #     if t > 1:
    #         z = torch.randn_like(self.x_t, device=self.device)
    #     else:
    #         z = torch.zeros_like(self.x_t, device=self.device)
        
    #     timestep = torch.Tensor([t]).repeat_interleave(B, dim=0).to(torch.long).to(self.device)

    #     pred_epsilon = self.denoiser(self.x_t, timestep).sample  # = score of p(x_t|x_t+1)

    #     # use the total predicted noise to denoise the image
    #     mu_t_minus_1 = self.predict_mu_t(pred_epsilon, timestep)
        
    #     sigma = self.extract(self.sqrt_betas, timestep, self.x_t.shape)
        
    #     # and then add noise to this image again
    #     self.x_t = mu_t_minus_1 + sigma*z
    #     # return x_tm1


    def unconditional_sample_denoiser_only(self, N):
        '''sampling using only the denoiser'''
        # start from random noise vector, x_0 (for simplicity, x_T declared as x_t instead of x_T)
        # self.denoiser.eval()
        
        # start from random noise vector, x_T
        self.x_t = torch.randn((N, self.img_C, self.img_H, self.img_W), device=self.device)
        
        for t in range(self.n_times-1, -1, -1):
            self.reverse_one_timestep_denoiser_only(self.x_t, t)

        # return x_t


