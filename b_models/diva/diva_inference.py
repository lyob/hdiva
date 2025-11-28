import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.autograd import grad
import math
import numpy as np
from b_models.diva.diva_module import DiVA

# ------------------------------ inference class ----------------------------- #
class DiVA_Inference(DiVA):
    def __init__(
            self, 
            diva,
            config,
        ):
        super(DiVA_Inference, self).__init__(
            diva.denoiser, 
            diva.infnet, 
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
    
    def compute_score_inference(self, log_p_z, noisy_x):
        # get the log posterior score
        log_p_z.backward(torch.ones_like(log_p_z), retain_graph=True)
        rec_score = noisy_x.grad
        return rec_score
        
    def reverse_one_timestep(self, noisy_x, t, z_sample):
        B, *_ = noisy_x.shape  # batch size
        timestep = torch.Tensor([t]).repeat_interleave(B, dim=0).long().to(self.device)
        
        if t > 1:
            z = torch.randn_like(noisy_x).to(self.device)
        else:
            z = torch.zeros_like(noisy_x).to(self.device)
        
        # get the score of the log posterior
        noisy_x = noisy_x.detach().requires_grad_(True)
        mu_t, logvar_t = self.infnet(noisy_x)
        log_p_z = self.compute_log_posterior(mu_t, logvar_t, z_sample)
        z_score = self.compute_score_inference(log_p_z, noisy_x)

        z_score_weight = self.extract(self.sqrt_one_minus_alpha_bars, timestep, noisy_x.shape)
        weighted_z_score = z_score_weight * z_score
        


        # # get conditional epsilon from the denoiser
        # pred_epsilon = self.denoiser(noisy_x, timestep)  # = score of p(x_t|x_t+1)
        # # pred_epsilon_guided = pred_epsilon - weighted_z_score
        
        # # get x_t-1 ~ N(mu_t-1, beta_t*I) = p(x_t-1|x_t, z)
        # prior_transition_mean = self.predict_mu_t(noisy_x, pred_epsilon, timestep)  # mean of prior transition operator
        # alpha = self.extract(self.alphas, timestep, noisy_x.shape)
        # sqrt_alpha = self.extract(self.sqrt_alphas, timestep, noisy_x.shape)
        # mu_t_minus_1 = prior_transition_mean + (1-alpha)/sqrt_alpha * z_score  # mean of posterior transition operator
        
        # sqrt_beta = self.extract(self.sqrt_betas, timestep, noisy_x.shape)  # std of either transition operator
        # # x_t_minus_1 = mu_t_minus_1 + sqrt_beta*z  # posterior sample
        # self.x_t = mu_t_minus_1 + sqrt_beta*z  # posterior sample
        # # self.x_t = self.x_t.clamp(-1., 1.)  # clamp to [-1, 1] range
        
        # # estimate x0 
        # # x0_hat_unguided = self.denoise_at_t(noisy_x, pred_epsilon, timestep)
        # # x0_hat_guided = self.denoise_at_t(noisy_x, pred_epsilon_guided, timestep)
        
        # # self.prior_score = -pred_epsilon / self.extract(self.sqrt_one_minus_alphas_prod, timestep, noisy_x.shape)
        
        # # return x_t_minus_1.clamp(-1., 1)
        # # return prior_transition_mean, mu_t_minus_1, pred_epsilon, pred_epsilon_guided, z_score, weighted_z_score, x0_hat_unguided, x0_hat_guided



                # get conditional epsilon from the denoiser
        pred_epsilon = self.denoiser(noisy_x, timestep)  # = score of p(x_t|x_t+1)
        pred_epsilon_guided = pred_epsilon - weighted_z_score
        
        # get x_t-1 ~ N(mu_t-1, beta_t*I) = p(x_t-1|x_t, z)
        posterior_transition_mean = self.predict_mu_t(noisy_x, pred_epsilon_guided, timestep)  # mean of prior transition operator
        # alpha = self.extract(self.alphas, timestep, noisy_x.shape)
        # sqrt_alpha = self.extract(self.sqrt_alphas, timestep, noisy_x.shape)
        # mu_t_minus_1 = prior_transition_mean + (1-alpha)/sqrt_alpha * z_score  # mean of posterior transition operator
        
        sqrt_beta = self.extract(self.sqrt_betas, timestep, noisy_x.shape)  # std of either transition operator
        # x_t_minus_1 = mu_t_minus_1 + sqrt_beta*z  # posterior sample
        self.x_t = posterior_transition_mean + sqrt_beta*z  # posterior sample
        # self.x_t = self.x_t.clamp(-1., 1.)  # clamp to [-1, 1] range
        
        # estimate x0 
        # x0_hat_unguided = self.denoise_at_t(noisy_x, pred_epsilon, timestep)
        # x0_hat_guided = self.denoise_at_t(noisy_x, pred_epsilon_guided, timestep)
        
        # self.prior_score = -pred_epsilon / self.extract(self.sqrt_one_minus_alphas_prod, timestep, noisy_x.shape)
        
        # return x_t_minus_1.clamp(-1., 1)
        # return prior_transition_mean, mu_t_minus_1, pred_epsilon, pred_epsilon_guided, z_score, weighted_z_score, x0_hat_unguided, x0_hat_guided




    def conditional_sample(self, N, target_image_input, return_chain=False, x_t=None):
        if x_t is None:
            self.x_t = torch.randn((N, self.img_C, self.img_H, self.img_W)).to(self.device).requires_grad_(True)
            self.initial_x_t = self.x_t
        else:
            self.x_t = x_t.clone().to(self.device).requires_grad_(True)
            self.initial_x_t = self.x_t
        
        # start from random noise vector, x_0 (for simplicity, x_T declared as x_t instead of x_T)
        self.infnet.eval()
        
        target_image = target_image_input.clone()
        target_image = target_image.to(self.device).requires_grad_(True)
        
        # assuming z_given_xt method
        mu, logvar = self.infnet(target_image)  
        z_sample = self.infnet.sample(mu, logvar)
        
        for t in range(self.n_times-1, -1, -1):
            self.reverse_one_timestep(self.x_t, t, z_sample)

    
    def conditional_sample_from_z(self, N:int, z:torch.Tensor, seed:int=0, x_init:torch.Tensor|None=None):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # start from random noise vector, x_0 (for simplicity, x_T declared as x_t instead of x_T)
        if x_init is None:
            self.x_t = torch.randn((N, self.img_C, self.img_H, self.img_W)).to(self.device).requires_grad_(True)
        else:
            self.x_t = x_init.clone().to(self.device).requires_grad_(True)
        self.initial_x_t = self.x_t
        
        for t in range(self.n_times-1, -1, -1):
            self.reverse_one_timestep(self.x_t, t, z)

    
    def conditional_sample_switch(self, N, initial_image_input, target_image_input, return_chain=False, noise_level=999):
        self.x_t = initial_image_input.clone().to(self.device).requires_grad_(True)
        self.initial_x_t = self.x_t
    
        # start from random noise vector, x_0 (for simplicity, x_T declared as x_t instead of x_T)
        self.infnet.eval()
        
        target_image = target_image_input.clone()
        target_image = target_image.to(self.device).requires_grad_(True)
        
        # assuming z_given_xt method
        z_sample, _, _ = self.infnet(target_image)  
        
        for t in range(noise_level, -1, -1):
            self.reverse_one_timestep(self.x_t, t, z_sample)


    def unconditional_sample_back_and_forth(self, N, return_chain=False):
        '''sampling using both the denoiser and infnet'''
        
        # start from random noise vector, x_T
        self.x_t = torch.randn((N, self.img_C, self.img_H, self.img_W)).to(self.device)
        self._history_idx += 1
        x_0_hat = self.x_t.clone()
        x_ts = []
        
        for t in range(self.n_times-1, -1, -1):
            # first, encode x_t into z_t
            z_t = self.infnet(self.x_t.detach())[0]
            # z = z_t + t/self.n_times * torch.randn_like(z_t).to(self.device) 
            
            # then use both x_t and z_t to produce x_t-1
            self.reverse_one_timestep(self.x_t.detach(), t, z_t)

        
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


    def unconditional_sample_denoiser_only(self, N):
        '''sampling using only the denoiser'''
        # start from random noise vector, x_0 (for simplicity, x_T declared as x_t instead of x_T)
        self.denoiser.eval()
        
        # start from random noise vector, x_T
        self.x_t = torch.randn((N, self.img_C, self.img_H, self.img_W), device=self.device)
        
        for t in range(self.n_times-1, -1, -1):
            # x_t, pred_epsilon = self.reverse_one_timestep_denoiser_only(x_t, t)
            self.reverse_one_timestep_denoiser_only(self.x_t, t)


