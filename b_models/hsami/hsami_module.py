from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from b_models.ddpm.ddpm_module import DDPM


class HSAMI(nn.Module):
    def __init__(
        self,
        denoiser_x,
        denoiser_y,
        infnet,
        noise_schedule_x,
        noise_schedule_y,
        sigma_minmax_x: tuple[float, float] = (0.0001, 0.9999),
        sigma_minmax_y: tuple[float, float] = (0.0001, 0.9999),
        timestep_dist_x: str = "uniform",
        timestep_dist_y: str = "uniform",
        num_timesteps_x=100,
        num_timesteps_y=100,
        reduction: Literal["mean", "sum"] = "mean",
        parameterization: Literal["noise", "velocity", "image"] = "velocity",
        rate_type: Literal[
            "grad", "norm", "cumulative", "kl"
        ] = "grad",  # how to compute the rate term in the SAMI loss
        weighted_mse: bool = False,
    ):
        super(HSAMI, self).__init__()
        """
        Args:
            denoiser_x: denoiser model for x
            denoiser_y: denoiser model for y
            infnet: inference network
            noise_schedule: noise schedule
            sigma_minmax: min and max values of sigma
            timestep_dist: distribution of timesteps
            num_timesteps: number of timesteps
            reduction: reduction method
            use_kl: whether to use KL divergence
        """
        self.infnet = infnet
        self.denoiser_x = DDPM(
            denoiser_x,
            noise_schedule_x,
            parameterization,
            sigma_minmax_x,
            timestep_dist_x,
            num_timesteps_x,
            reduction,
        )
        self.denoiser_y = DDPM(
            denoiser_y,
            noise_schedule_y,
            parameterization,
            sigma_minmax_y,
            timestep_dist_y,
            num_timesteps_y,
            reduction,
        )
        self.parameterization = parameterization
        self.rate_type = rate_type
        self.weighted_mse = weighted_mse
        self.reduction = reduction
    
    def broadcast(self, x, like):
        return x.view(-1, *((1,) * (len(like.shape) - 1)))

    def scale_to_minus_one_to_one(self, x):
        # according to the DDPMs paper, normalization seems to be crucial to train reverse process network
        return x * 2 - 1

    def reverse_scale_to_zero_to_one(self, x):
        return (x + 1) * 0.5

    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

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

    def compute_z_rate(self, mu, logvar, clean_x):
        # compute the rate in Z
        kl_z = self.compute_kl_against_zero(mu, logvar)
        return kl_z
    
    def compute_y_score_wrt_z(self, z_sample_up, noisy_y):
        # compute the score in Y with respect to Z
        # ∇_y log p(z|y_\eta)
        mu_z_t, logvar_z_t = self.infnet.y_to_z(noisy_y)
        log_posterior = self.compute_log_posterior(mu_z_t, logvar_z_t, z_sample_up)
        score = self.compute_score_from_logp(log_posterior, noisy_y)
        return score

    def compute_p_y_given_ye_and_x(self, eta, mu_y_up, y_eta, logvar_y_up):
        '''compute p(y|y_eta, x) = p(y_eta|y) q(y|x)'''

        s = logvar_y_up.exp().clamp(min=1e-8)  # variance of q(y|x)
        c = eta / (1-eta)  # rescaled variance of p(y_\eta|y_up) 
        mean_y_given_x = mu_y_up  # mean of q(y|x)
        mean_ye_given_y = y_eta / (1-eta).sqrt()  # rescaled mean of p(y_\eta|y_up) 

        v = (s*c) / (s + c)  # variance of p(y|y_\eta)
        yhat_x = (c * mean_y_given_x + s * mean_ye_given_y) / (s + c)  # mean of p(y|y_\eta, x)
        return yhat_x, v, s, c

    def compute_y_score_wrt_x(self, eta, mu_y_wrt_x, logvar_y_wrt_x, y_eta):
        '''compute the score of y with respect to x
        ∇_{ye} log p(ye|x)'''
        var_y_wrt_x = logvar_y_wrt_x.exp().clamp(min=1e-8)
        score = -((1-eta) * var_y_wrt_x + eta).reciprocal() * (y_eta - (1-eta).sqrt() * mu_y_wrt_x)
        return score

    def compute_y_score_wrt_z_given_x(self, eta, mu_y_wrt_x, y_eta, logvar_y_wrt_x, z_sample_from_x, sample_based=False):
        '''compute the score of z given y_eta and x
        p(z|y_eta, x) = int p(z|y) p(y|y_eta, x) dy = E_{p(y|y_eta, x)} [ p(z|y) ]
        then get ∇_{ye} log p(z|ye, x)
        '''

        # get the mean and variance of p(y|y_eta, x)
        yhat_x, v, s, c = self.compute_p_y_given_ye_and_x(eta, mu_y_wrt_x, y_eta, logvar_y_wrt_x)
        
        a = s / ((s + c) * (1-eta).sqrt())
        
        # compute the score of z given y_eta and x
        if not sample_based:
            # use the mean of p(y|y_eta, x) to compute the score
            mu_z, logvar_z = self.infnet.y_to_z(yhat_x)
            
            # calculate the log posterior of z under p(z|yhat_x)
            log_posterior = self.compute_log_posterior(mu_z, logvar_z, z_sample_from_x)

            # compute the score of z given y_eta and x, estimated by the derivative of log q(z|y) evaluated at y=yhat_x
            score = a * self.compute_score_from_logp(log_posterior, yhat_x)
        else:
            # use samples of z to compute the score
            y_sample = self.infnet.sample(yhat_x, v.log(), num_samples=5)  # sample multiple z's for better estimation
            mu_z, logvar_z = self.infnet.y_to_z(y_sample)

            log_posterior = self.compute_log_posterior(mu_z, logvar_z, z_sample_from_x)
            # compute the score of z given y_eta and x, estimated by the weighted average of the derivative of log q(z|y) evaluated at different y samples
            score_k = self.compute_score_from_logp(log_posterior, y_sample)

            w_k = torch.softmax(log_posterior, dim=0, out=log_posterior)  # convert to weights
            score = a * (w_k * score_k).sum(dim=0)  # weighted average of scores
            
        return score

    def compute_y_score_wrt_x_and_z(self, eta, mu_y_wrt_x, logvar_y_wrt_x, y_eta, z_sample_up):
        '''compute the posterior of Y given X, Z.
        ∇_{ye} log p(ye|x, z) = ∇_{ye} log p(ye|x) + ∇_{ye} log p(z|ye, x) 
        '''

        # compute ∇_{ye} log p(ye|x)
        y_score_wrt_x = self.compute_y_score_wrt_x(eta, mu_y_wrt_x, logvar_y_wrt_x, y_eta)

        # compute ∇_{ye} log p(z|ye, x)
        y_score_wrt_z = self.compute_y_score_wrt_z_given_x(eta, mu_y_wrt_x, y_eta, logvar_y_wrt_x, z_sample_up)

        # combine the two scores to get the posterior score
        y_score_posterior = y_score_wrt_x + y_score_wrt_z
        return y_score_posterior

    def sample_from_posterior_wrt_x_and_z(self, eta, mu_y_wrt_x, logvar_y_wrt_x, y_eta, pred_noise_y_unguided, z_sample_up):
        '''sample a clean estimate of y from the posterior of Y given X, Z
        y~p(y|x, z)'''

        y_score_wrt_xz = self.compute_y_score_wrt_x_and_z(eta, mu_y_wrt_x, logvar_y_wrt_x, y_eta, z_sample_up)
        y_sample_down = 1/(1-eta).sqrt() * (y_eta + eta * y_score_wrt_xz)

        return y_sample_down, y_score_wrt_xz

    def compute_y_rate(self, y_score_wrt_xz, y_score_wrt_z):
        '''compute the rate in Y
        the rate in Y is the norm of the score ∇_{ye} log p(x|ye, z) = ∇_{ye} log p(ye|x, z) - ∇_{ye} log p(z|ye)
        '''
        y_score_of_z_wrt_ye_and_x = y_score_wrt_xz - y_score_wrt_z
        rate_y = torch.norm(y_score_of_z_wrt_ye_and_x, dim=[1, 2, 3]).mean()
        
        return rate_y

    def forward(self, clean_x, unconditional=False):
        """forward pass of the model. Assume the clean images have already been normalized to [-1, 1]."""
        B, _ = shape = clean_x.shape
        device = clean_x.device
        clean_x = clean_x.detach()

        # get latent variables Y and Z from clean image
        mu_y_wrt_x, logvar_y_wrt_x, mu_z_wrt_x, logvar_z_wrt_x = self.infnet.x_to_z(clean_x)
        y_sample_up = self.infnet.sample(mu_y_wrt_x, logvar_y_wrt_x)
        z_sample_up = self.infnet.sample(mu_z_wrt_x, logvar_z_wrt_x)

        ####################### Z rate #######################
        # add noise to the clean image
        t_tensor_x = self.denoiser_x.sample_timesteps(B, self.denoiser_x.num_timesteps, dist=self.denoiser_x.timestep_dist, device=device)
        noisy_x, noise_x = self.denoiser_x.make_noisy(clean_x, t_tensor_x)
        noisy_x = noisy_x.requires_grad_(True)

        # compute the rate in Z
        rate_z = self.compute_z_rate(mu_z_wrt_x, logvar_z_wrt_x, clean_x)

        ####################### Y denoiser with Z guidance #######################
        # get noisy Y 
        t_tensor_y = self.denoiser_y.sample_timesteps(B, self.denoiser_y.num_timesteps, dist=self.denoiser_y.timestep_dist, device=device)
        noisy_y, noise_y = self.denoiser_y.make_noisy(y_sample_up, t_tensor_y)
        noisy_y = noisy_y.requires_grad_(True)
        eta = self.extract(self.denoiser_y.one_minus_alpha_bars, t_tensor_y, shape)

        # get the denoiser in Y
        pred_noise_y_unguided = self.denoiser_y(noisy_y, t_tensor_y)

        # get the posterior score of Y given X, Z
        y_sample_down_wrt_xz, y_score_wrt_xz = self.sample_from_posterior_wrt_x_and_z(eta, mu_y_wrt_x, logvar_y_wrt_x, noisy_y, pred_noise_y_unguided, z_sample_up)

        # get the posterior score of Y given X
        y_score_wrt_z = self.compute_y_score_wrt_z(z_sample_up, noisy_y)

        # compute the rate in Y
        rate_y = self.compute_y_rate(y_score_wrt_xz, y_score_wrt_z)

        ####################### X denoiser with Y guidance #######################
        gamma = self.extract(self.denoiser_x.one_minus_alpha_bars, t_tensor_x, shape)

        # get the denoiser in X
        pred_noise_x = self.denoiser_x(noisy_x, t_tensor_x)

        # get the guidance vector in X wrt Y
        x_score_wrt_y = self.compute_score_from_logp(y_sample_down_wrt_xz, noisy_x)
        # x_guidance_wrt_y = self.compute_score_from_logp(y_sample_up, noisy_x)

        # get the score in X
        # y_score_weight = gamma.sqrt()
        pred_noise_x_guided = pred_noise_x - gamma.sqrt() * x_score_wrt_y

        # optional weight in X
        weight = 1 / (2 * gamma * (1-gamma))

        target = noisy_x
        prediction = pred_noise_x_guided
    
        return target, prediction, weight, rate_y, rate_z


    def compute_loss(self, clean_x, beta=1.0):
        target, prediction, weight, rate_y, rate_z = self.forward(clean_x)
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
