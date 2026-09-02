import math

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

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
            config.parameterization,
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
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, timestep, x_t.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep, x_t.shape)
        x0_hat = 1 / sqrt_alpha_bar * (x_t - sqrt_one_minus_alpha_bar * pred_epsilon)
        return x0_hat

    def predict_mu_t(self, x_t, pred_epsilon, timestep):
        alpha = self.extract(self.alphas, timestep, x_t.shape)
        sqrt_alpha = self.extract(self.sqrt_alphas, timestep, x_t.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep, x_t.shape)

        # denoise at time t, utilizing predicted noise
        mu_t_minus_1 = 1 / sqrt_alpha * (x_t - (1 - alpha) / sqrt_one_minus_alpha_bar * pred_epsilon)
        return mu_t_minus_1

    def predict_epsilon_from_v(self, v_hat, noisy_x, timestep):
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, timestep, v_hat.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep, v_hat.shape)
        return sqrt_alpha_bar * v_hat + sqrt_one_minus_alpha_bar * noisy_x

    def predict_x0_from_v(self, v_hat, noisy_x, timestep):
        sqrt_alpha_bar           = self.extract(self.sqrt_alpha_bars,           timestep, v_hat.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep, v_hat.shape)
        return sqrt_alpha_bar * noisy_x - sqrt_one_minus_alpha_bar * v_hat

    def predict_epsilon_from_v2(self, v_hat, noisy_x, timestep):
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep, v_hat.shape)
        alpha_bar                = self.extract(self.alpha_bars,                timestep, v_hat.shape)
        return sqrt_one_minus_alpha_bar * (noisy_x + alpha_bar * v_hat)

    def predict_x0_from_v2(self, v_hat, noisy_x, timestep):
        sqrt_alpha_bar      = self.extract(self.sqrt_alpha_bars,      timestep, v_hat.shape)
        one_minus_alpha_bar = self.extract(self.one_minus_alpha_bars, timestep, v_hat.shape)
        return sqrt_alpha_bar * (noisy_x - one_minus_alpha_bar * v_hat)

    def reverse_one_timestep(self, t):
        """denoiser only"""
        B, *_ = self.x_t.shape  # batch size
        timestep = torch.Tensor([t]).repeat_interleave(B, dim=0).to(torch.long).to(self.device)

        if t > 1:
            z = torch.randn_like(self.x_t, device=self.device)
        else:
            z = torch.zeros_like(self.x_t, device=self.device)

        if self.config.parameterization == "noise":
            pred_epsilon = self.denoiser(self.x_t, timestep)  # = score of p(x_t|x_t+1)

            # use the total predicted noise to denoise the image
            transition_mean = self.predict_mu_t(self.x_t, pred_epsilon, timestep)

            sqrt_beta = self.extract(self.sqrt_betas, timestep, self.x_t.shape)

            # and then add noise to this image again
            self.x_t = transition_mean + sqrt_beta * z

        elif self.config.parameterization == "velocity1":
            pred_v = self.denoiser(self.x_t, timestep)

            # convert v -> epsilon, then reuse the standard mu_t machinery
            pred_epsilon = self.predict_epsilon_from_v(pred_v, self.x_t, timestep)
            transition_mean = self.predict_mu_t(self.x_t, pred_epsilon, timestep)

            sqrt_beta = self.extract(self.sqrt_betas, timestep, self.x_t.shape)
            self.x_t = transition_mean + sqrt_beta * z

        elif self.config.parameterization == "velocity2":
            pred_v = self.denoiser(self.x_t, timestep)
            pred_epsilon = self.predict_epsilon_from_v2(pred_v, self.x_t, timestep)
            transition_mean = self.predict_mu_t(self.x_t, pred_epsilon, timestep)
            sqrt_beta = self.extract(self.sqrt_betas, timestep, self.x_t.shape)
            self.x_t = transition_mean + sqrt_beta * z


    @torch.no_grad()
    def unconditional_sample(self, N):
        """sampling using only the denoiser"""
        # start from random noise vector, x_0 (for simplicity, x_T declared as x_t instead of x_T)
        self.denoiser.eval()

        # assert the parameterization is either noise, velocity1 or velocity2
        assert self.config.parameterization in ["noise", "velocity1", "velocity2"], "Invalid parameterization"

        # start from random noise vector, x_T
        self.x_t = torch.randn((N, self.img_C, self.img_H, self.img_W), device=self.device)

        for t in range(self.n_times - 1, -1, -1):
            self.reverse_one_timestep(t)

    def add_noise_and_denoise(self, image_input, t: int = 10):
        assert t > 0, "t must be greater than 0"
        assert t < self.n_times, "t must be less than the number of timesteps"

        # add noise to image
        timestep = torch.Tensor([t]).repeat_interleave(1, dim=0).long().to(self.device)
        noisy_x, noise = self.make_noisy(image_input, timestep)

        if self.config.parameterization == "noise":
            # get conditional epsilon from the denoiser
            pred_epsilon = self.denoiser(noisy_x, timestep)  # = score of p(x_t|x_t+1)
            x0_hat = self.denoise_at_t(noisy_x, pred_epsilon, timestep).detach()

        elif self.config.parameterization == "velocity1":
            pred_v = self.denoiser(noisy_x, timestep)
            x0_hat = self.predict_x0_from_v(pred_v, noisy_x, timestep).detach()

        elif self.config.parameterization == "velocity2":
            pred_v = self.denoiser(noisy_x, timestep)
            x0_hat = self.predict_x0_from_v2(pred_v, noisy_x, timestep).detach()

        return x0_hat, noisy_x.detach()
