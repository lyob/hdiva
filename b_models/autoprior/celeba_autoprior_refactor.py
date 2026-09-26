'''This script contains the implementation of the single latent stage autoprior model'''
import math
# from utils.analysis import plant
from typing import Tuple, Union

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
# import torch.autograd.profiler as profiler
from functorch import grad, vmap
from torch import Size

from b_models.autoprior.celeba_denoisers_t_as_input import Diffusion
# import time
from b_models.autoprior.celeba_encoders import (BF_CNN_RF, HalfUNet,
                                                VariableConvEncoder)

# ----------------------- autoencoding diffusion model ----------------------- #

class AutoPrior_args(Diffusion):
    def __init__(
            self,
            encoder,
            denoiser,
            args,
            device
        ):
        super(AutoPrior_args, self).__init__(
            denoiser=denoiser,
            args=args,
            device=device
        )
    
        self.encoder = encoder  # this instantiates the Encoder class
        self.denoiser = denoiser  # this instantiates the Denoiser class
        
        self.n_times = args.timesteps
        self.img_C = args.num_channels
        self.img_H = args.image_dims
        self.img_W = args.image_dims
        
        self.scale_x0_to_minus_one_to_one:bool = args.rescale
        self.use_weighted_MSE:bool = args.weighted_MSE
        self.kl_reduction = args.kl_reduction
        self.log_posterior_method = args.log_posterior_method
        self.denoiser_target = args.denoiser_target
        self.train_only_denoiser = args.train_only_denoiser
        self.device = device
        self.kl = 0
        
        # define 1/sigma std schedule
        self.noise_dist = 'cosine_in_alpha_bar' if args.noise_dist == None else args.noise_dist
        self.define_noise_schedule(args)
        
        self.complex_reverse_variance = args.complex_reverse_variance if 'complex_reverse_variance' in args else False
        

    def compute_log_posterior(self, mu, cov, z_sample):       
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
        kl = torch.distributions.kl.kl_divergence(
            dist.MultivariateNormal(mu, cov), 
            dist.MultivariateNormal(torch.zeros_like(mu), torch.diag_embed(torch.ones_like(mu)))
            ).mean(dim=0)
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
    
    def get_z_score_and_kl(self, x_zeros, noisy_x):
        z_sample, mu, var = self.encoder(x_zeros)
        _, mu_t, var_t = self.encoder(noisy_x)
        log_p_z = self.compute_log_posterior_vectorized(mu_t, var_t, z_sample)
        self.compute_kl_vectorized(mu, var)
        score = self.compute_score(log_p_z, noisy_x)
        return score
    
        # z_sample, mu, cov = self.encoder(x_zeros)
        # z_sample_t, mu_t, cov_t = self.encoder(noisy_x)
        
        # if self.log_posterior_method == 'z_given_xt':
        #     log_p_z = self.compute_log_posterior(mu_t, cov_t, z_sample)
        # elif self.log_posterior_method == 'z_given_x0':
        #     log_p_z = self.compute_log_posterior(mu, cov, z_sample_t)
        # else:
        #     raise ValueError('log_posterior_method must be either "z_given_xt" or "z_given_x0"')
        # self.compute_kl(mu, cov)
        # score = self.compute_score(log_p_z, noisy_x)
        # return score


    def forward(self, x_zeros:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''forward pass of the model'''
        
        if self.scale_x0_to_minus_one_to_one:
            x_zeros = self.scale_to_minus_one_to_one(x_zeros)  # normalize the input image to -1 ~ 1
        
        B, *_ = x_zeros.shape  # batch size
        
        # (1) randomly choose diffusion time-step
        t = torch.randint(low=0, high=self.n_times, size=(B,), device=self.device, dtype=torch.long)
        
        # (2) forward diffusion process: perturb x_zeros with fixed variance schedule
        x_zeros = x_zeros.detach().requires_grad_(True)
        noisy_x, epsilon = self.make_noisy(x_zeros, t)
        noisy_x = noisy_x.detach().requires_grad_(True)
        
        z_score = self.get_z_score_and_kl(x_zeros, noisy_x)
        
        if self.denoiser_target == 'noise':
            # (5) weight the score by sqrt(1-alpha_bar)
            z_score_weight = self.extract(self.sqrt_one_minus_alpha_bars, t, x_zeros.shape)

            # (6) calculate the weighted z-score (including the learnable gain)
            weighted_z_score = z_score_weight * z_score
        
            # (7) estimate the noise epsilon: predict epsilon (noise) given perturbed data at diffusion-timestep t.
            pred_epsilon = self.denoiser(noisy_x, t)  # = score of p(x_t|x_t+1)
            pred_epsilon_cond = pred_epsilon - weighted_z_score
            
            # (8) set the target and prediction for the model
            target = epsilon
            prediction = pred_epsilon_cond
            
        elif self.denoiser_target == 'image':
            z_score_weight = self.extract(self.one_minus_alpha_bars, t, x_zeros.shape) / self.extract(self.sqrt_alpha_bars, t, x_zeros.shape)
            weighted_z_score = z_score_weight * z_score
            pred_x0 = self.denoiser(noisy_x, t)
            pred_x0_given_z = pred_x0 + weighted_z_score
            
            target = x_zeros
            prediction = pred_x0_given_z
        
        elif self.denoiser_target == 'residual':
            z_score_weight = self.extract(self.one_minus_alpha_bars, t, x_zeros.shape)
            weighted_z_score = z_score_weight * z_score
            pred_residual = self.denoiser(noisy_x, t)
            pred_residual_given_z = pred_residual + weighted_z_score
            
            target = self.extract(self.sqrt_alpha_bars, t, x_zeros.shape) * x_zeros - noisy_x
            prediction = pred_residual_given_z
        
        else: 
            raise ValueError('denoiser_target must be either "noise" or "image" or "residual"')
        
        if self.use_weighted_MSE:
            beta_t = self.extract(self.betas, t, x_zeros.shape)
            alpha_t = self.extract(self.alphas, t, x_zeros.shape)
            one_minus_alpha_bar_t = self.extract(self.one_minus_alpha_bars, t, x_zeros.shape)
            weights = beta_t**2 / (one_minus_alpha_bar_t * alpha_t)
            
            # if self.complex_reverse_variance:
            #     sigma_sq = self.get_sigma_sq(t, x_zeros.shape)
            # else:
            sigma_sq = beta_t
            
            mse_weights = weights/sigma_sq
        else:
            # mse_weights = torch.ones(x_zeros.shape, device=self.device)
            mse_weights = torch.ones_like(x_zeros)
                
        return target, prediction, mse_weights

    
    def compute_loss(self, batch_x):
        '''compute the loss function'''
        # (9) compute the loss
        target, prediction, mse_weights = self.forward(batch_x)
        mse_loss = F.mse_loss(prediction, target, reduction=self.kl_reduction)
        weighted_mse_loss = (mse_loss * mse_weights).mean()
        return weighted_mse_loss, self.kl
    

    

# extend Autopiror_args to include the inference model
class AutoPrior_Inference(AutoPrior_args):
    def __init__(self,
                 encoder:Union[HalfUNet, VariableConvEncoder, BF_CNN_RF],
                 denoiser:torch.nn.Module, 
                 args,
                 device:torch.device
                 ):
        super(AutoPrior_Inference, self).__init__(encoder, denoiser, args, device)
        self._log_p_z = None
        self._initial_x_t = None
        self._x_t = None
        self._mu_t = None
        self._var_t = None
        self._prior_score = None
        self.num_generated_images = 1
        
        self._history_idx = 0
        self.initialize_histories(self.num_generated_images)
    
    def initialize_histories(self, N):
        n_times = self.n_times
        self.log_p_z_history = torch.zeros((n_times, N), device=self.device)
        self.x_t_history = torch.zeros((n_times, N, self.img_C, self.img_H, self.img_W), device=self.device)
        self.mu_t_history = torch.zeros((n_times, N, self.encoder.latent_dims), device=self.device)
        self.var_t_history = torch.zeros((n_times, N, self.encoder.latent_dims), device=self.device)
        self.prior_scores_history = torch.zeros((n_times, N, self.img_C, self.img_H, self.img_W), device=self.device)
        
    def reset_histories(self):
        self._history_idx = 0
        self.log_p_z_history = torch.zeros_like(self.log_p_z_history)
        self.x_t_history = torch.zeros_like(self.x_t_history)
        self.mu_t_history = torch.zeros_like(self.mu_t_history)
        self.var_t_history = torch.zeros_like(self.var_t_history)
        self.prior_scores_history = torch.zeros_like(self.prior_scores_history)
    
    @property
    def log_p_z(self):
        return self._log_p_z
    @property
    def prior_score(self):
        return self._prior_score
    @property
    def initial_x_t(self):
        return self._initial_x_t
    @property
    def x_t(self):
        return self._x_t
    @property
    def mu_t(self):
        return self._mu_t
    @property
    def var_t(self):
        return self._var_t
    
    @log_p_z.setter
    def log_p_z(self, value):
        self._log_p_z = value
        self.log_p_z_history[self._history_idx] = value.detach()  # collect every log_p_z value set
    @prior_score.setter
    def prior_score(self, value):
        self._prior_score = value
        self.prior_scores_history[self._history_idx] = value.detach()
    @initial_x_t.setter
    def initial_x_t(self, value):
        self._initial_x_t = value.detach()
        # self.initial_x_t = value.detach()
    @x_t.setter
    def x_t(self, value):
        self._x_t = value
        self.x_t_history[self._history_idx] = value.detach()
    @mu_t.setter
    def mu_t(self, value):
        self._mu_t = value
        self.mu_t_history[self._history_idx] = value.detach()
    @var_t.setter
    def var_t(self, value):
        self._var_t = value
        self.var_t_history[self._history_idx] = value.detach()
    
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
        mu_t_minus_1 = 1 / sqrt_alpha * (x_t - (1-alpha)/sqrt_one_minus_alpha_bar * pred_epsilon)
        return mu_t_minus_1 
        
    def reverse_one_timestep(self, noisy_x, t, z_sample):
        B, *_ = noisy_x.shape  # batch size
        timestep = torch.Tensor([t]).repeat_interleave(B, dim=0).long().to(self.device)
        
        if t > 1:
            z = torch.randn_like(noisy_x).to(self.device)
        else:
            z = torch.zeros_like(noisy_x).to(self.device)
        
        # get the score of the log posterior
        noisy_x = noisy_x.detach().requires_grad_(True)
        # print(self._history_idx)
        _, self.mu_t, self.var_t = self.encoder(noisy_x)
        self.log_p_z = self.compute_log_posterior_vectorized(self.mu_t, self.var_t, z_sample)
        z_score = self.compute_score(self.log_p_z, noisy_x).detach()
        z_score_weight = self.extract(self.sqrt_one_minus_alpha_bars, timestep, noisy_x.shape)
        weighted_z_score = z_score_weight * z_score
        
        # get conditional epsilon from the denoiser
        pred_epsilon = self.denoiser(noisy_x, timestep)  # = score of p(x_t|x_t+1)
        pred_epsilon_guided = pred_epsilon - weighted_z_score
        
        # get x_t-1 ~ N(mu_t-1, beta_t*I) = p(x_t-1|x_t, z)
        prior_transition_mean = self.predict_mu_t(noisy_x, pred_epsilon, timestep)  # mean of prior transition operator
        alpha = self.extract(self.alphas, timestep, noisy_x.shape)
        sqrt_alpha = self.extract(self.sqrt_alphas, timestep, noisy_x.shape)
        mu_t_minus_1 = prior_transition_mean + (1-alpha)/sqrt_alpha * z_score  # mean of posterior transition operator
        
        sqrt_beta = self.extract(self.sqrt_betas, timestep, noisy_x.shape)  # std of either transition operator
        # x_t_minus_1 = mu_t_minus_1 + sqrt_beta*z  # posterior sample
        self.x_t = mu_t_minus_1 + sqrt_beta*z  # posterior sample
        
        # estimate x0 
        x0_hat_unguided = self.denoise_at_t(noisy_x, pred_epsilon, timestep)
        x0_hat_guided = self.denoise_at_t(noisy_x, pred_epsilon_guided, timestep)
        
        self.prior_score = -pred_epsilon / self.extract(self.sqrt_one_minus_alpha_bars, timestep, noisy_x.shape)
        
        # return x_t_minus_1.clamp(-1., 1)
        # return prior_transition_mean, mu_t_minus_1, pred_epsilon, pred_epsilon_guided, z_score, weighted_z_score, x0_hat_unguided, x0_hat_guided

    def conditional_sample(self, N, target_image_input, return_chain=False, x_t=None):
        self.reset_histories()
        # if not self.histories_initialized:
        self.initialize_histories(N)
        
        if x_t is None:
            self.x_t = torch.randn((N, self.img_C, self.img_H, self.img_W)).to(self.device).requires_grad_(True)
            self.initial_x_t = self.x_t
        else:
            self.x_t = x_t.clone().to(self.device).requires_grad_(True)
            self.initial_x_t = self.x_t
        
        # start from random noise vector, x_0 (for simplicity, x_T declared as x_t instead of x_T)
        self.encoder.eval()
        
        target_image = target_image_input.clone()
        target_image = target_image.to(self.device).requires_grad_(True)
        if self.scale_x0_to_minus_one_to_one:
            target_image = self.scale_to_minus_one_to_one(target_image)
        
        # assuming z_given_xt method
        self.z_sample, _, _ = self.encoder(target_image)  
        
        for t in range(self.n_times-1, -1, -1):
            self.reverse_one_timestep(self.x_t, t, self.z_sample)
            self._history_idx += 1
            
        if self.scale_x0_to_minus_one_to_one:
            self.x_t_history = self.reverse_scale_to_zero_to_one(self.x_t_history)

    
    def conditional_sample_from_z(self, N:int, z:torch.Tensor, seed:int=0, x_init:torch.Tensor|None=None):
        self.reset_histories()
        self.initialize_histories(N)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # start from random noise vector, x_0 (for simplicity, x_T declared as x_t instead of x_T)
        if x_init is None:
            self.x_t = torch.randn((N, self.img_C, self.img_H, self.img_W)).to(self.device).requires_grad_(True)
        else:
            self.x_t = x_init.clone().to(self.device).requires_grad_(True)
        self.initial_x_t = self.x_t
        # self._history_idx += 1

        z = z.clone().to(self.device).requires_grad_(True)
        
        for t in range(self.n_times-1, -1, -1):
            self.reverse_one_timestep(self.x_t, t, z)
            self._history_idx += 1

        if self.scale_x0_to_minus_one_to_one:
            self.x_t_history = self.reverse_scale_to_zero_to_one(self.x_t_history)


    
    def conditional_sample_switch(self, N, initial_image_input, target_image_input, return_chain=False, noise_level=999):
        self.reset_histories()
        # if not self.histories_initialized:
        self.initialize_histories(N)
        
        self.x_t = initial_image_input.clone().to(self.device).requires_grad_(True)
        self.initial_x_t = self.x_t
    
        # start from random noise vector, x_0 (for simplicity, x_T declared as x_t instead of x_T)
        self.encoder.eval()
        
        target_image = target_image_input.clone()
        target_image = target_image.to(self.device).requires_grad_(True)
        if self.scale_x0_to_minus_one_to_one:
            target_image = self.scale_to_minus_one_to_one(target_image)
        
        # assuming z_given_xt method
        z_sample, _, _ = self.encoder(target_image)  
        
        for t in range(noise_level, -1, -1):
            self.reverse_one_timestep(self.x_t, t, z_sample)
            self._history_idx += 1
            
        if self.scale_x0_to_minus_one_to_one:
            self.x_t_history = self.reverse_scale_to_zero_to_one(self.x_t_history)


    def _batch_mahalanobis(self, bL, bx):
        r"""
        Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
        for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.

        Accepts batches for both bL and bx. They are not necessarily assumed to have the same batch
        shape, but `bL` one should be able to broadcasted to `bx` one.
        """
        n = bx.size(-1)
        bx_batch_shape = bx.shape[:-1]

        # Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
        # we are going to make bx have shape (..., 1, j,  i, 1, n) to apply batched tri.solve
        bx_batch_dims = len(bx_batch_shape)
        bL_batch_dims = bL.dim() - 2
        outer_batch_dims = bx_batch_dims - bL_batch_dims
        old_batch_dims = outer_batch_dims + bL_batch_dims
        new_batch_dims = outer_batch_dims + 2 * bL_batch_dims
        # Reshape bx with the shape (..., 1, i, j, 1, n)
        bx_new_shape = bx.shape[:outer_batch_dims]
        for sL, sx in zip(bL.shape[:-2], bx.shape[outer_batch_dims:-1]):
            bx_new_shape += (sx // sL, sL)
        bx_new_shape += (n,)
        bx = bx.reshape(bx_new_shape)
        # Permute bx to make it have shape (..., 1, j, i, 1, n)
        permute_dims = (
            list(range(outer_batch_dims))
            + list(range(outer_batch_dims, new_batch_dims, 2))
            + list(range(outer_batch_dims + 1, new_batch_dims, 2))
            + [new_batch_dims]
        )
        bx = bx.permute(permute_dims)

        flat_L = bL.reshape(-1, n, n)  # shape = b x n x n
        flat_x = bx.reshape(-1, flat_L.size(0), n)  # shape = c x b x n
        flat_x_swap = flat_x.permute(1, 2, 0)  # shape = b x n x c
        M_swap = (
            torch.linalg.solve_triangular(flat_L, flat_x_swap, upper=False).pow(2).sum(-2)
        )  # shape = b x c
        M = M_swap.t()  # shape = c x b

        # Now we revert the above reshape and permute operators.
        permuted_M = M.reshape(bx.shape[:-1])  # shape = (..., 1, j, i, 1)
        permute_inv_dims = list(range(outer_batch_dims))
        for i in range(bL_batch_dims):
            permute_inv_dims += [outer_batch_dims + i, old_batch_dims + i]
        reshaped_M = permuted_M.permute(permute_inv_dims)  # shape = (..., 1, i, j, 1)
        return reshaped_M.reshape(bx_batch_shape)

    def multivariate_normal_log_prob(self, loc, covariance_matrix, value):
        # def log_prob(self, value):
        diff = value - loc
        M = self._batch_mahalanobis(torch.linalg.cholesky(covariance_matrix), diff)
        half_log_det = (
            covariance_matrix.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        )
        return -0.5 * (value.shape[0] * math.log(2 * math.pi) + M) - half_log_det
    
    def reverse_one_timestep_multiple_conds(self, noisy_x, t, z_sample):
        num_conds = z_sample.shape[0]  # num conditions
        C, B, c, h, w = noisy_x.shape  # batch size
        timestep = torch.Tensor([t]).repeat_interleave(B*C, dim=0).long().to(self.device)
        
        if t > 1:
            z = torch.randn_like(noisy_x).to(self.device)
        else:
            z = torch.zeros_like(noisy_x).to(self.device)
        
        # get the score of the log posterior
        noisy_x = noisy_x.detach().requires_grad_(True)
        
        z_sample = z_sample.view(C, B, -1)
        
        

        # Define function that computes log_p_z for a single condition
        def log_prob_fn(noisy_x_c, z_sample_c):
            _, mu_t, cov_t = self.encoder(noisy_x_c)
            log_p_z_c = self.multivariate_normal_log_prob(mu_t, cov_t, z_sample_c)
            # log_p_z_c = dist.MultivariateNormal(mu_t, cov_t).log_prob(z_sample_c)
            return log_p_z_c.sum()  # Sum over batch to ensure it's a scalar

        # Compute gradients using functorch.grad
        grad_fn = grad(log_prob_fn, argnums=0)  # Differentiate w.r.t. noisy_x_c

        # Vectorize over conditions using vmap
        z_score = vmap(grad_fn, randomness='different')(noisy_x, z_sample)  # 13, 5, 1, 64, 64
        
        # from functorch import vjp

        # def log_prob_fn(noisy_x_c, z_sample_c):
        #     _, mu_t, cov_t = self.encoder(noisy_x_c)
        #     log_p_z_c = dist.MultivariateNormal(mu_t, cov_t).log_prob(z_sample_c)
        #     return log_p_z_c.sum()

        # # Get vjp function
        # log_p_vjp_fn, vjp_fn = torch.func.vjp(log_prob_fn, noisy_x, z_sample)

        # # Compute vjp with ones_like(log_p_z)
        # z_score = vjp_fn(torch.ones_like(log_p_vjp_fn))[0]
        
        
        
        
        
        # noisy_x = noisy_x.view(C*B, c, h, w)
        # _, mu_t, cov_t = self.encoder(noisy_x)
        # # mu_t has shape (B*C, z_dim), cov_t has shape (B*C, z_dim, z_dim)
        
        # log_p_z = dist.MultivariateNormal(mu_t, cov_t).log_prob(z_sample)
        # # log_p_z has shape (B*C)
        # log_p_z = log_p_z.view(C, B, 1, 1, 1)
        # noisy_x = noisy_x.view(C, B, c, h, w)

        # # Compute gradients
        # z_score = torch.autograd.grad(log_p_z, noisy_x, torch.ones_like(log_p_z), retain_graph=True, create_graph=True)[0]

        
        
        # Function to compute gradient for one condition (implicitly works on single C index)
        # def grad_fn(log_p_z_c, noisy_x_c):
        #     # print(log_p_z_c.requires_grad)
        #     log_p_z_c.requires_grad_()
        #     return grad(
        #         log_p_z_c, noisy_x_c, torch.ones_like(log_p_z_c), retain_graph=True, create_graph=True
        #     )[0]
        
        # # print(noisy_x.requires_grad)
        # z_score = torch.vmap(grad_fn)(log_p_z, noisy_x)
        
        # z_score_weight = self.extract(self.sqrt_one_minus_alpha_bars, timestep, noisy_x.shape)
        # weighted_z_score = z_score_weight * z_score

        # get conditional epsilon from the denoiser
        noisy_x = noisy_x.view(C*B, c, h, w)
        pred_epsilon = self.denoiser(noisy_x, timestep)
        pred_epsilon = pred_epsilon.view(C, B, c, h, w)
        
        # get x_t-1 ~ N(mu_t-1, beta_t*I) = p(x_t-1|x_t, z)
        prior_transition_mean = self.predict_mu_t(noisy_x, pred_epsilon.view(C*B, c, h, w), timestep)  # mean of prior transition operator
        alpha = self.extract(self.alphas, timestep, noisy_x.shape)
        sqrt_alpha = self.extract(self.sqrt_alphas, timestep, noisy_x.shape)
        mu_t_minus_1 = prior_transition_mean + (1-alpha)/sqrt_alpha * z_score.view(-1, c, h, w)  # mean of posterior transition operator
        
        sqrt_beta = self.extract(self.sqrt_betas, timestep, noisy_x.shape)  # std of either transition operator
        x_t = mu_t_minus_1 + sqrt_beta*z.view(-1, c, h, w)  # posterior sample
        
        return x_t.view(C, B, c, h, w)
    
    def conditional_sample_from_multiple_zs(self, N:int, z:torch.Tensor, seed:int=0):
        '''currently limited to N=1 seed'''
        num_conds = z.shape[0]
        self.reset_histories()
        # self.initialize_histories(1)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # start from random noise vector, x_0 (for simplicity, x_T declared as x_t instead of x_T)
        x_t = torch.randn((N, self.img_C, self.img_H, self.img_W)).to(self.device).requires_grad_(True)
        x_t = x_t.unsqueeze(0).repeat_interleave(num_conds, dim=0)
        
        z = z.repeat_interleave(N, dim=0)
        
        for t in range(self.n_times-1, -1, -1):
            x_t = self.reverse_one_timestep_multiple_conds(x_t, t, z)
            # self._history_idx += 1

        if self.scale_x0_to_minus_one_to_one:
            x_t = self.reverse_scale_to_zero_to_one(x_t)
            # self.x_t_history = self.reverse_scale_to_zero_to_one(self.x_t_history)
        return x_t

    def unconditional_sample_back_and_forth(self, N, return_chain=False):
        '''sampling using both the denoiser and encoder'''
        
        # start from random noise vector, x_T
        self.x_t = torch.randn((N, self.img_C, self.img_H, self.img_W)).to(self.device)
        self._history_idx += 1
        x_0_hat = self.x_t.clone()
        x_ts = []
        
        for t in range(self.n_times-1, -1, -1):
            # first, encode x_t into z_t
            z_t = self.encoder(self.x_t.detach())[0]
            # z = z_t + t/self.n_times * torch.randn_like(z_t).to(self.device) 
            
            # then use both x_t and z_t to produce x_t-1
            self.reverse_one_timestep(self.x_t.detach(), t, z_t)
            # self.x_t, x_0_hat = results[0], results[-1]
            self._history_idx += 1

        if self.scale_x0_to_minus_one_to_one:
            x_ts = self.reverse_scale_to_zero_to_one(x_ts)
        
    '''denoiser only'''
    def reverse_one_timestep_denoiser_only(self, noisy_x, t):
        B, *_ = noisy_x.shape  # batch size
        timestep = torch.Tensor([t]).repeat_interleave(B, dim=0).long().to(self.device)
        if t > 1:
            z = torch.randn_like(noisy_x, device=self.device)
        else:
            z = torch.zeros_like(noisy_x, device=self.device)
        
        # at inference, we estimate the predicted noise (epsilon) in the image
        pred_epsilon = self.denoiser(noisy_x, timestep)  # = score of p(x_t|x_t+1)
        
        # use the total predicted noise to denoise the image
        mu_t_minus_1 = self.predict_mu_t(noisy_x, pred_epsilon, timestep)
        
        # noise at time t
        # if self.complex_reverse_variance:
        #     sigma_sq = self.get_sigma_sq(timestep, noisy_x.shape)
        #     sigma = torch.sqrt(sigma_sq)
        # else:
        sigma = self.extract(self.sqrt_betas, timestep, noisy_x.shape)
        
        # and then add noise to this image again
        self.x_t = mu_t_minus_1 + sigma*z
    
    
    
    def unconditional_sample_denoiser_only(self, N):
        # start from random noise vector, x_0 (for simplicity, x_T declared as x_t instead of x_T)
        self.reset_histories()
        self.initialize_histories(N)
        
        # start from random noise vector, x_T
        self.x_t = torch.randn((N, self.img_C, self.img_H, self.img_W)).to(self.device).requires_grad_(True)
        self.initial_x_t = self.x_t

        self.denoiser.eval()
        # autoregressively denoise from x_T to x_0, i.e., generate image from noise, x_T
        # x_ts = torch.zeros((self.n_times, N, self.img_C, self.img_H, self.img_W))
        
        for t in range(self.n_times-1, -1, -1):
            self.reverse_one_timestep_denoiser_only(self.x_t, t)
            self._history_idx += 1

        # denormalize x_0 into 0 ~ 1 ranged values.
        if self.scale_x0_to_minus_one_to_one:
            self.x_t_history = self.reverse_scale_to_zero_to_one(self.x_t_history)


    """conditional sample with only one latent axis fixed (i.e., only one dimension of z is used for conditioning, while the rest are sampled from the prior)"""

    # def compute_log_posterior_dim(self, mu, var, fixed_z_val, d):
    #     """
    #     Docstring for compute_log_posterior_dim

    #     :param mu: latent mean of the noisy image at time t, with shape (B, z_dim)
    #     :param var: latent var of the noisy image at time t, with shape (B, z_dim)
    #     :param fixed_z_val: latent value to condition on, with shape (B,)
    #     :param d: latent index of fixed_z_val, int scalar
    #     """
    #     # z_sample_d = z_sample[..., d]
    #     var_d = var[..., d].clamp(min=1e-8)
    #     mu_d = mu[..., d]
    #     quadratic_term = (fixed_z_val - mu_d) ** 2 / var_d
        
    #     constant = torch.log(torch.tensor(2.0 * torch.pi, device=mu.device, dtype=mu.dtype))
    #     log_p_z_d = -0.5 * (constant + var_d.log() + quadratic_term)
    #     return log_p_z_d

    # def reverse_one_timestep_single_latent(self, noisy_x, t, fixed_dim, fixed_z_val):
    #     B, *_ = noisy_x.shape  # batch size
    #     timestep = torch.Tensor([t]).repeat_interleave(B, dim=0).long().to(self.device)
        
    #     if t > 1:
    #         z = torch.randn_like(noisy_x).to(self.device)
    #     else:
    #         z = torch.zeros_like(noisy_x).to(self.device)
        
    #     # get the score of the log posterior
    #     noisy_x = noisy_x.detach().requires_grad_(True)
    #     # print(self._history_idx)
    #     _, self.mu_t, self.var_t = self.encoder(noisy_x)
    #     self.log_p_z = self.compute_log_posterior_dim(self.mu_t, self.var_t, fixed_z_val, fixed_dim)

    #     z_score = self.compute_score(self.log_p_z, noisy_x).detach()
    #     z_score_weight = self.extract(self.sqrt_one_minus_alpha_bars, timestep, noisy_x.shape)
    #     weighted_z_score = z_score_weight * z_score
        
    #     # get conditional epsilon from the denoiser
    #     pred_epsilon = self.denoiser(noisy_x, timestep)  # = score of p(x_t|x_t+1)
    #     pred_epsilon_guided = pred_epsilon - weighted_z_score
        
    #     # get x_t-1 ~ N(mu_t-1, beta_t*I) = p(x_t-1|x_t, z)
    #     prior_transition_mean = self.predict_mu_t(noisy_x, pred_epsilon, timestep)  # mean of prior transition operator
    #     alpha = self.extract(self.alphas, timestep, noisy_x.shape)
    #     sqrt_alpha = self.extract(self.sqrt_alphas, timestep, noisy_x.shape)
    #     mu_t_minus_1 = prior_transition_mean + (1-alpha)/sqrt_alpha * z_score  # mean of posterior transition operator
        
    #     sqrt_beta = self.extract(self.sqrt_betas, timestep, noisy_x.shape)  # std of either transition operator
    #     # x_t_minus_1 = mu_t_minus_1 + sqrt_beta*z  # posterior sample
    #     self.x_t = mu_t_minus_1 + sqrt_beta*z  # posterior sample
        
    #     # estimate x0 
    #     x0_hat_unguided = self.denoise_at_t(noisy_x, pred_epsilon, timestep)
    #     x0_hat_guided = self.denoise_at_t(noisy_x, pred_epsilon_guided, timestep)
        
    #     self.prior_score = -pred_epsilon / self.extract(self.sqrt_one_minus_alpha_bars, timestep, noisy_x.shape)
        
    #     # return x_t_minus_1.clamp(-1., 1)
    #     # return prior_transition_mean, mu_t_minus_1, pred_epsilon, pred_epsilon_guided, z_score, weighted_z_score, x0_hat_unguided, x0_hat_guided

    # def conditional_sample_from_z_one_axis_fixed(
    #         self,
    #         target_image:torch.Tensor,
    #         fixed_dim:int, 
    #         fixed_z_val:torch.Tensor, 
    #         seed:int=0, 
    #         x_init:torch.Tensor|None=None
    #     ):
    #     self.reset_histories()
    #     self.initialize_histories(1)
        
    #     torch.manual_seed(seed)
    #     np.random.seed(seed)
        
    #     # start from random noise vector, x_0 (for simplicity, x_T declared as x_t instead of x_T)
    #     if x_init is None:
    #         self.x_t = torch.randn((1, self.img_C, self.img_H, self.img_W)).to(self.device).requires_grad_(True)
    #     else:
    #         self.x_t = x_init.clone().to(self.device).requires_grad_(True)
    #     self.initial_x_t = self.x_t
    #     # self._history_idx += 1

    #     # z = z.clone().to(self.device).requires_grad_(True)
        
    #     for t in range(self.n_times-1, -1, -1):
    #         self.reverse_one_timestep_single_latent(self.x_t, t, fixed_dim, fixed_z_val)
    #         self._history_idx += 1