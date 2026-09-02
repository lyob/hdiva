from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from b_models.ddpm.ddpm_module import DDPM
from torch.func import jvp, vjp  # type: ignore[import]

# from torch.autograd.functional import jvp, vjp



class HSAMI(nn.Module):
    def __init__(
        self,
        recnet,
        denoiser_x,
        denoiser_y,
        args
    ):
        super(HSAMI, self).__init__()
        """
        Args:
            recnet: inference network
            denoiser_x: denoiser model for x
            denoiser_y: denoiser model for y
            args: configuration object containing the following attributes:
                noise_schedule: noise schedule
                sigma_minmax: min and max values of sigma
                timestep_dist: distribution of timesteps
                num_timesteps: number of timesteps
                reduction: reduction method
                use_kl: whether to use KL divergence
        """
        self.noise_schedule_x=args.noise_schedule_x
        self.noise_schedule_y=args.noise_schedule_y
        self.sigma_minmax_x=args.sigma_minmax_x
        self.sigma_minmax_y=args.sigma_minmax_y
        self.timestep_dist_x=args.timestep_dist_x
        self.timestep_dist_y=args.timestep_dist_y
        self.num_timesteps_x=args.num_timesteps_x
        self.num_timesteps_y=args.num_timesteps_y
        self.reduction=args.reduction
        self.parameterization=args.parameterization
        self.rate_type=args.rate_type
        self.weighted_mse=args.weighted_mse
        # overall constant for the r_eta^2 weight in the mutual-information rate
        # estimator (absorbs the derivation constant + any dη/dt reweighting).
        self.rate_weight_scale = getattr(args, "rate_weight_scale", 1.0)

        assert self.rate_type in ["grad", "norm", "cumulative", "kl"], "Invalid rate_type"
        assert self.parameterization in ["noise", "image", "velocity"], "Invalid parameterization"
        assert self.reduction in ["mean", "sum"], "Invalid reduction type"
        assert self.timestep_dist_x in ["uniform", "linearly_increasing"], "Invalid timestep distribution for x"
        assert self.timestep_dist_y in ["uniform", "linearly_increasing"], "Invalid timestep distribution for y"
        assert self.noise_schedule_x in ["cosine_in_alpha_bar", "linear_in_alpha_bar"], "Invalid noise schedule for x"
        assert self.noise_schedule_y in ["cosine_in_alpha_bar", "linear_in_alpha_bar"], "Invalid noise schedule for y"
        assert self.num_timesteps_x > 0, "num_timesteps_x must be greater than 0"
        assert self.num_timesteps_y > 0, "num_timesteps_y must be greater than 0"

        self.recnet = recnet
        self.denoiser_x = DDPM(
            denoiser_x,
            args.noise_schedule_x,
            args.parameterization,
            args.sigma_minmax_x,
            args.timestep_dist_x,
            args.num_timesteps_x,
            args.reduction,
        )
        self.denoiser_y = DDPM(
            denoiser_y,
            args.noise_schedule_y,
            args.parameterization,
            args.sigma_minmax_y,
            args.timestep_dist_y,
            args.num_timesteps_y,
            args.reduction,
        )
    
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
        var = logvar.clamp(-8, 8).exp()
        quadratic_term = ((z_sample - mu) ** 2 / var).sum(dim=-1)
        constant = z_dim * torch.log(torch.tensor(2.0 * torch.pi, device=mu.device, dtype=mu.dtype))
        log_p_z = -0.5 * (constant + logvar.sum(dim=-1) + quadratic_term)
        return log_p_z

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
        z_sample = self.recnet.sample(mu, logvar)
        mu_t, logvar_t = self.recnet(noisy_x)
        log_p_z = self.compute_log_posterior(mu_t, logvar_t, z_sample)
        score = self.compute_score_from_logp(log_p_z, noisy_x)
        return score

####################
    def rate_weight(self, denoiser, t_tensor):
        '''r_eta^2 weight from the mutual-information identity
            I(.;._eta|.) = ∫ E[ r_eta^2 ||∇_{._eta} log p(.|._eta,.)||^2 ] d eta.
        Working the I-MMSE change of variables for the VP channel
        (._eta = sqrt(alpha_bar) . + sqrt(1-alpha_bar) eps) gives
            r_eta^2 = 1 / (2 * alpha_bar),
        i.e. the high-noise levels are up-weighted. The leading constant and any
        uniform-t vs uniform-eta (dη/dt) reweighting are folded into
        rate_weight_scale. Returns shape (B,).'''
        alpha_bar = denoiser.alpha_bars.gather(-1, t_tensor)  # (B,)
        return self.rate_weight_scale / (2.0 * alpha_bar)

    def _rate_reduce(self, score, denoiser, t_tensor):
        '''Reduce a per-sample guidance-score field g into a scalar-per-sample rate,
        reducing over all non-batch dims. rate_type selects the functional:
          "grad": r_eta^2 * ||g||^2  -- Monte-Carlo integrand of the mutual information
                                        (this is the quantity the identity integrates)
          "norm": ||g||              -- legacy un-weighted, un-squared score norm
        Returns shape (B,).'''
        g = score.flatten(1)  # (B, D)
        if self.rate_type == "grad":
            sq_norm = g.pow(2).sum(dim=1)                       # ||g||^2, (B,)
            return self.rate_weight(denoiser, t_tensor) * sq_norm
        return torch.linalg.vector_norm(g, dim=1)              # ||g||, (B,)

    def compute_z_rate(self, mu_z_from_xg, logvar_z_from_xg, noisy_x, z_sample_up_from_x, t_tensor_x):
        '''compute the rate in Z from the score ∇_x log p(z|x)'''
        log_posterior = self.compute_log_posterior(mu_z_from_xg, logvar_z_from_xg, z_sample_up_from_x)
        score_z_given_x = self.compute_score_from_logp(log_posterior, noisy_x)
        return self._rate_reduce(score_z_given_x, self.denoiser_x, t_tensor_x)
    
    def compute_score_x_given_ye(self, eta, pred_y_unguided, vjp_fn, mu_y_given_x, logvar_y_given_x):
        '''compute the score of x with respect to y_eta
        ∇_{ye} log p(x|ye) can be computed analytically using the y denoiser and x -> y encoder
        '''
        # denoiser_mean = pred_y_unguided
        # denoiser_var = eta

        # a, sig = (1 - eta).sqrt().clamp(min=1e-3), eta.sqrt()
        a, sig = (1 - eta).sqrt(), eta.sqrt()
        r = mu_y_given_x - pred_y_unguided

        def JT(w, sig, a, vjp_eps):                # J^T w through the eps net
            # vjp_detached = vjp_eps(w)[0].detach()                         # type: ignore[misc]
            vjp_detached = vjp_eps(w)[0]                         # type: ignore[misc]
            return (w - sig * vjp_detached) / a

        # isotropic: C = eta I  (denominator unchanged)
        var_y_given_x = logvar_y_given_x.clamp(-8, 8).exp()
        score_iso = JT(r / (var_y_given_x + eta), sig, a, vjp_fn)
        return score_iso
    
    def denoiser_and_vjp(self, y_denoiser, t_tensor_y, y_eta, eta):
        # a, sig = (1 - eta).sqrt().clamp(min=1e-3), eta.sqrt()
        a, sig = (1 - eta).sqrt(), eta.sqrt()
        f = lambda x: y_denoiser.denoiser(x, t_tensor_y)
        eps_hat, vjp_fn = vjp(f, y_eta)          # eps-pred + J_eps^T(·)  # type: ignore[misc]
        y_bar = (y_eta - sig * eps_hat) / a      # Tweedie clean-signal estimate
        return eps_hat, y_bar, vjp_fn
    
    def compute_score_z_given_ye_and_x(self, mu_z_from_ye, logvar_z_from_ye, z_sample_up_from_x, y_eta):
        '''compute the score of z with respect to y_eta and x
        ∇_{ye} log p(z|ye, x) can be computed analytically using the y -> z encoder and the x -> y encoder
        '''
        log_posterior = self.compute_log_posterior(mu_z_from_ye, logvar_z_from_ye, z_sample_up_from_x)
        score_z_given_ye_and_x = self.compute_score_from_logp(log_posterior, y_eta)
        return score_z_given_ye_and_x

    def compute_score_z_given_ye(self, mu_z_from_ye, logvar_z_from_ye, z_sample_up_from_y, y_eta):
        '''compute the score of z with respect to y_eta
        ∇_{ye} log p(z|ye)
        '''
        log_posterior = self.compute_log_posterior(mu_z_from_ye, logvar_z_from_ye, z_sample_up_from_y)
        score_z_given_ye = self.compute_score_from_logp(log_posterior, y_eta)
        return score_z_given_ye
    
    def compute_y_rate(self, score_z_given_ye, score_x_given_ye, score_z_given_ye_and_x, t_tensor_y):
        '''compute the rate in Y from the score ∇_{ye} log p(x|ye, z)'''
        score_x_given_ye_and_z = score_x_given_ye + score_z_given_ye_and_x - score_z_given_ye
        return self._rate_reduce(score_x_given_ye_and_z, self.denoiser_y, t_tensor_y)
    
    def compute_score_ye_given_x_and_z(self, score_x_given_ye, score_z_given_ye_and_x, score_ye):
        '''compute the score of y with respect to x and z
        ∇_{ye} log p(ye|x, z) = ∇_{ye} log p(ye|x) + ∇_{ye} log p(z|ye, x) + ∇_{ye} log p(ye)
        '''
        # compute the score of y with respect to x and z
        score_y_given_xz = score_x_given_ye + score_z_given_ye_and_x + score_ye
        return score_y_given_xz
    
    def compute_score_y_given_xg(self, mu_y_from_xg, logvar_y_from_xg, y_sample_down_from_x_and_z, noisy_x):
        '''compute the score of x with respect to y_sample_down_from_x_and_z'''
        log_posterior = self.compute_log_posterior(mu_y_from_xg, logvar_y_from_xg, y_sample_down_from_x_and_z)
        score_y_given_x = self.compute_score_from_logp(log_posterior, noisy_x)
        return score_y_given_x
    
    def convert_noise_estimate_to_signal(self, noisy_x, t_tensor, pred_noise):
        '''get the signal estimate from the noise prediction'''
        alpha_bar = self.extract(self.denoiser_x.alpha_bars, t_tensor, noisy_x.shape)
        if self.parameterization == "noise":
            x0_pred = (noisy_x - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
        elif self.parameterization == "x0":
            x0_pred = pred_noise
        elif self.parameterization == "velocity":
            x0_pred = alpha_bar.sqrt() * noisy_x - (1 - alpha_bar).sqrt() * pred_noise
        else:
            raise ValueError(f"Invalid parameterization: {self.parameterization}")
        return x0_pred
    
    def convert_noise_estimate_to_score(self, pred_noise, noisy_x, t_tensor):
        '''compute the score from the noise prediction'''
        alpha_bar = self.extract(self.denoiser_x.alpha_bars, t_tensor, noisy_x.shape)
        if self.parameterization == "noise":
            score = -pred_noise / (1 - alpha_bar).sqrt()
        elif self.parameterization == "x0":
            score = (noisy_x - pred_noise) / (1 - alpha_bar).sqrt()
        elif self.parameterization == "velocity":
            score = (alpha_bar.sqrt() * noisy_x - pred_noise) / (1 - alpha_bar).sqrt()
        else:
            raise ValueError(f"Invalid parameterization: {self.parameterization}")
        return score
    
    def predict_y_using_posterior_score(self, y_eta, t_tensor_y, score_ye_given_x_and_z):
        '''compute the guided prediction in Y via Tweedie: E[y|y_eta,x,z] = (y_eta + (1-alpha_bar)*s) / sqrt(alpha_bar)'''
        alpha_bar = self.extract(self.denoiser_y.alpha_bars, t_tensor_y, y_eta.shape)
        one_minus_alpha_bar = self.extract(self.denoiser_y.one_minus_alpha_bars, t_tensor_y, y_eta.shape)
        pred_y_guided_by_x_and_z = (y_eta + one_minus_alpha_bar * score_ye_given_x_and_z) / alpha_bar.sqrt()
        return pred_y_guided_by_x_and_z
    
    def predict_noise_using_guidance_score(self, y_eta, t_tensor_y, pred_noise_y, score_x_and_z_given_ye):
        '''compute the guided prediction in Y via guidance score: y_guided = y_unguided - (1-alpha_bar)/sqrt(alpha_bar) * guidance_score'''
        one_minus_alpha_bar = self.extract(self.denoiser_y.one_minus_alpha_bars, t_tensor_y, y_eta.shape)
        pred_noise_y_guided = pred_noise_y - one_minus_alpha_bar.sqrt() * score_x_and_z_given_ye
        return pred_noise_y_guided
    

    def forward(self, clean_x, t_y_max=None, unconditional=False, debug=False):
        """forward pass of the model. Assume the clean images have already been normalized to [-1, 1]."""
        shape = clean_x.shape
        B = shape[0]
        device = clean_x.device
        clean_x = clean_x.detach()

        # get latent variables Y and Z from clean image
        # update_stats=True: this clean encoding is the canonical source for the
        # running y-scale used to normalize y before the y-diffusion.
        mu_y_from_x, logvar_y_from_x, mu_z_from_x, logvar_z_from_x = self.recnet.x_to_y_to_z(clean_x, update_stats=True)
        y_sample_up_wrt_x = self.recnet.sample(mu_y_from_x, logvar_y_from_x)
        z_sample_up_from_x = self.recnet.sample(mu_z_from_x, logvar_z_from_x)

        mu_z_wrt_y, logvar_z_wrt_y = self.recnet.y_to_z(y_sample_up_wrt_x)
        z_sample_up_from_y = self.recnet.sample(mu_z_wrt_y, logvar_z_wrt_y)

        ####################### Z rate #######################
        # get noisy x and gamma
        t_tensor_x = self.denoiser_x.sample_timesteps(B, self.denoiser_x.num_timesteps, dist=self.denoiser_x.timestep_dist, device=device)
        noisy_x, noise_x = self.denoiser_x.make_noisy(clean_x, t_tensor_x)
        noisy_x = noisy_x.requires_grad_(True)
        gamma = self.extract(self.denoiser_x.one_minus_alpha_bars, t_tensor_x, shape)

        # compute the rate in Z
        mu_y_from_xg, logvar_y_from_xg, mu_z_from_xg, logvar_z_from_xg = self.recnet.x_to_y_to_z(noisy_x)
        rate_z = self.compute_z_rate(mu_z_from_xg, logvar_z_from_xg, noisy_x, z_sample_up_from_x, t_tensor_x)

        ####################### Y denoiser with Z guidance #######################
        # get noisy Y and eta; t_y_max caps the noisiest timestep (curriculum support)
        _t_y_num = self.num_timesteps_y if t_y_max is None else t_y_max + 1
        t_tensor_y = self.denoiser_y.sample_timesteps(B, _t_y_num, dist=self.denoiser_y.timestep_dist, device=device)
        noisy_y, noise_y = self.denoiser_y.make_noisy(y_sample_up_wrt_x, t_tensor_y)
        noisy_y = noisy_y.requires_grad_(True)
        eta = self.extract(self.denoiser_y.one_minus_alpha_bars, t_tensor_y, y_sample_up_wrt_x.shape)
        mu_z_from_ye, logvar_z_from_ye = self.recnet.y_to_z(noisy_y)

        # get the denoiser output and the VJP of the denoiser
        pred_noise_y, pred_y_unguided, vjp_fn = self.denoiser_and_vjp(self.denoiser_y, t_tensor_y, noisy_y, eta)

        # compute the scores
        score_ye = self.convert_noise_estimate_to_score(pred_noise_y, noisy_y, t_tensor_y)
        score_z_given_ye = self.compute_score_z_given_ye(mu_z_from_ye, logvar_z_from_ye, z_sample_up_from_y, noisy_y)
        score_x_given_ye = self.compute_score_x_given_ye(eta, pred_y_unguided, vjp_fn, mu_y_from_x, logvar_y_from_x)
        score_z_given_ye_and_x = self.compute_score_z_given_ye_and_x(mu_z_from_ye, logvar_z_from_ye, z_sample_up_from_x, noisy_y)

        # compute the rate in Y
        rate_y = self.compute_y_rate(score_z_given_ye, score_x_given_ye, score_z_given_ye_and_x, t_tensor_y)

        # get the posterior score of Y given X, Z
        # score_ye_given_x_and_z = self.compute_score_ye_given_x_and_z(score_x_given_ye, score_z_given_ye_and_x, score_ye)
        # pred_y_guided = self.predict_y_using_posterior_score(noisy_y, t_tensor_y, score_ye_given_x_and_z)

        # get the posterior score of Y given X, Z
        score_x_and_z_given_ye = score_x_given_ye + score_z_given_ye_and_x
        pred_y_noise_guided_by_x_and_z = self.predict_noise_using_guidance_score(noisy_y, t_tensor_y, pred_noise_y, score_x_and_z_given_ye)
        pred_y_guided_by_x_and_z = self.convert_noise_estimate_to_signal(noisy_y, t_tensor_y, pred_y_noise_guided_by_x_and_z)

        y_sample_down_from_x_and_z = pred_y_guided_by_x_and_z

        # get the posterior score of Y given Z
        pred_noise_y_guided_by_z = self.predict_noise_using_guidance_score(noisy_y, t_tensor_y, pred_noise_y, score_z_given_ye)
        # pred_y_guided_by_z = self.convert_noise_estimate_to_signal(noisy_y, t_tensor_y, pred_noise_y_guided_by_z)

        # --- prior-fit path for mse_y: detach the x->y encoder so this loss trains
        #     only denoiser_y (+ the y->z guidance), i.e. fits Delta_Y and NOT H_q(Y|Z) ---
        y_up_detached = y_sample_up_wrt_x.detach()
        noisy_y_pf, noise_y_pf = self.denoiser_y.make_noisy(y_up_detached, t_tensor_y)
        noisy_y_pf = noisy_y_pf.requires_grad_(True)

        pred_noise_y_pf = self.denoiser_y.denoiser(noisy_y_pf, t_tensor_y)
        mu_z_from_ye_pf, logvar_z_from_ye_pf = self.recnet.y_to_z(noisy_y_pf)
        score_z_given_ye_pf = self.compute_score_z_given_ye(
            mu_z_from_ye_pf, logvar_z_from_ye_pf, z_sample_up_from_y, noisy_y_pf
        )
        pred_noise_y_guided_by_z = self.predict_noise_using_guidance_score(
            noisy_y_pf, t_tensor_y, pred_noise_y_pf, score_z_given_ye_pf
        )
        
        target_y = noise_y_pf
        prediction_y = pred_noise_y_guided_by_z

        ####################### X denoiser with Y guidance #######################
        # get the denoiser in X
        pred_noise_x = self.denoiser_x.denoiser(noisy_x, t_tensor_x)

        # get the guidance vector in Xg wrt Y  #NOTE: this is up for debate, technically we want to condition on y~p(y|x,z), but p(y|x) might be more stable
        # score_y_given_xg = self.compute_score_y_given_xg(mu_y_from_xg, logvar_y_from_xg, y_sample_down_from_x_and_z, noisy_x)
        score_y_given_xg = self.compute_score_y_given_xg(mu_y_from_xg, logvar_y_from_xg, y_sample_up_wrt_x, noisy_x)

        # get the score in X
        pred_noise_x_guided = pred_noise_x - gamma.sqrt() * score_y_given_xg

        # optional weight in X
        weight = 1 / (2 * gamma * (1-gamma))

        target_x = noise_x
        prediction_x = pred_noise_x_guided

        # --- debug norms (remove this block when no longer needed) ---
        debug_norms = None
        if debug:
            debug_norms = {
                "raw_mse_x":          F.mse_loss(pred_noise_x, noise_x).item(),
                "raw_mse_y":          F.mse_loss(pred_noise_y, noise_y).item(),
                "pred_noise_y":       pred_noise_y.detach().abs().mean().item(),
                # "a_min":              (1 - eta).sqrt().min().item(),
                "score_x|ye":         score_x_given_ye.detach().abs().mean().item(),
                "score_xz|ye":        score_x_and_z_given_ye.detach().abs().mean().item(),
                "score_z|ye":         score_z_given_ye.detach().abs().mean().item(),
                "score_y|xg":         score_y_given_xg.detach().abs().mean().item(),
                "pred_y_unguided":    pred_y_unguided.detach().abs().mean().item(),
                "pred_y_guided_by_xz":      pred_y_guided_by_x_and_z.detach().abs().mean().item(),
                # "noise_x_guided":     pred_noise_x_guided.detach().abs().mean().item(),
            }
        # --- end debug norms ---

        return target_x, prediction_x, target_y, prediction_y, weight, rate_y, rate_z, debug_norms


    def compute_loss(self, clean_x, beta_y=1.0, beta_z=1.0, t_y_max=None, debug=False):
        target_x, prediction_x, target_y, prediction_y, weight, rate_y, rate_z, debug_norms = self.forward(clean_x, t_y_max=t_y_max, debug=debug)
        # target and prediction shape: (B, D)
        # weight shape: (B, 1)
        # rate shape: (B, D)

        # Compute the MSE loss weighted by weight
        mse_x = (prediction_x - target_x) ** 2
        if self.weighted_mse:
            mse_x = weight * mse_x

        # always average over batch, but sum/avg over other dimensions
        mse_x = (
            mse_x.mean() if self.reduction == "mean" 
            else mse_x.mean(dim=0).sum()
        )

        mse_y = F.mse_loss(prediction_y, target_y, reduction=self.reduction)
        if self.weighted_mse:
            mse_y = weight * mse_y


        if self.rate_type == "grad" or self.rate_type == "norm":
            rate_y_loss = (
                rate_y.mean() if self.reduction == "mean" 
                else rate_y.mean(dim=0).sum()
            )
            rate_z_loss = (
                rate_z.mean() if self.reduction == "mean"
                else rate_z.mean(dim=0).sum()
            )
            # reduce over spatial dimensions, then avg over batch
        elif self.rate_type == "kl":
            rate_y_loss = rate_y.mean()  # already reduced over batch in compute_kl
            rate_z_loss = rate_z.mean()  # already reduced over batch in compute_kl
        else:
            raise ValueError(f"Invalid rate_type: {self.rate_type}")

        total_loss = mse_x + mse_y + beta_y * rate_y_loss + beta_z * rate_z_loss  # average over batch
        # total_loss = mse_x + beta_y * rate_y_loss + beta_z * rate_z_loss  # average over batch

        return total_loss, mse_x, mse_y, rate_y_loss, rate_z_loss, debug_norms






    ################## inference & generation ###################


    def predict_mu_t_from_signal(self, denoiser, x_t, x0_hat, timestep):
        """
        Compute mu_{t-1} using the posterior mean formula.
        timestep: integer tensor with values in [0, num_timesteps-1]
        """

        # Extract values for timestep t
        sqrt_alpha_t = self.extract(denoiser.sqrt_alphas, timestep, x_t.shape)
        alpha_bar_t = self.extract(denoiser.alpha_bars, timestep, x_t.shape)
        beta_t = self.extract(denoiser.betas, timestep, x_t.shape)

        # Handle alpha_bar_{t-1} with boundary condition
        # At t=0, we want alpha_bar_{-1} = 1.0
        alpha_bar_prev = torch.ones_like(alpha_bar_t)

        # For t > 0, get alpha_bar_{t-1}
        mask = timestep > 0  # Boolean tensor of shape [B]
        if mask.any():
            timestep_prev = timestep - 1
            # Only extract for timesteps > 0
            alpha_bar_prev_extracted = self.extract(denoiser.alpha_bars, timestep_prev, x_t.shape)
            # Use where to handle the boundary
            alpha_bar_prev = torch.where(
                mask.view(-1, *((1,) * (len(x_t.shape) - 1))),  # broadcast mask
                alpha_bar_prev_extracted,
                alpha_bar_prev,  # keep 1.0 for t=0
            )

        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)

        # Compute the two coefficients
        coef_x0 = (sqrt_alpha_bar_prev * beta_t) / (1.0 - alpha_bar_t)
        coef_xt = (sqrt_alpha_t * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar_t)

        # Compute posterior mean
        mu_t_minus_1 = coef_x0 * x0_hat + coef_xt * x_t

        return mu_t_minus_1

    def sample_from_reverse_operator(self, denoiser, noisy_signal, pred_signal_guided, t_tensor_y):
        mu_t_minus_1 = self.predict_mu_t_from_signal(denoiser, noisy_signal, pred_signal_guided, t_tensor_y)
        sqrt_beta_tilde_t = self.extract(denoiser.sqrt_posterior_variance, t_tensor_y, noisy_signal.shape)
        return mu_t_minus_1 + sqrt_beta_tilde_t * torch.randn_like(noisy_signal)

    def reverse_one_timestep_y(self, noisy_y, t, mu_y_from_x, logvar_y_from_x, z_sample_up_from_x):
        """
        Reverse one timestep in Y, with Z guidance. This is used for inference/sampling.
        Args:
            noisy_y: the noisy input at the current timestep
            eta: the noise level at the current timestep
            z_sample: the sample of Z to condition on
        Returns:
            pred_y: the predicted clean signal at the current timestep
        """
        # detach from the previous timestep's graph so the reverse loop doesn't
        # accumulate one huge computation graph across all timesteps
        noisy_y = noisy_y.detach().requires_grad_(True)

        B, *_ = noisy_y.shape  # batch size
        t_tensor_y = torch.Tensor([t]).repeat_interleave(B, dim=0).long().to(noisy_y.device)

        if t > 1:
            z = torch.randn_like(noisy_y).to(noisy_y.device)
        else:
            z = torch.zeros_like(noisy_y).to(noisy_y.device)

        eta = self.extract(self.denoiser_y.one_minus_alpha_bars, t_tensor_y, noisy_y.shape)

        # get the denoiser output and the VJP of the denoiser
        pred_noise_y, pred_y_unguided, vjp_fn = self.denoiser_and_vjp(self.denoiser_y, t_tensor_y, noisy_y, eta)

        mu_z_from_ye, logvar_z_from_ye = self.recnet.y_to_z(noisy_y)

        # compute the scores
        # score_z_given_ye = self.compute_score_z_given_ye(mu_z_from_ye, logvar_z_from_ye, z_sample_up_from_y, noisy_y)
        score_x_given_ye = self.compute_score_x_given_ye(eta, pred_y_unguided, vjp_fn, mu_y_from_x, logvar_y_from_x)
        score_z_given_ye_and_x = self.compute_score_z_given_ye_and_x(mu_z_from_ye, logvar_z_from_ye, z_sample_up_from_x, noisy_y)
        score_x_and_z_given_ye = score_x_given_ye + score_z_given_ye_and_x

        # conditional generation
        # pred_noise_y_guided_by_x_and_z = self.predict_noise_using_guidance_score(noisy_y, t_tensor_y, pred_y_unguided, score_x_and_z_given_ye)
        pred_noise_y_guided_by_x_and_z = self.predict_noise_using_guidance_score(noisy_y, t_tensor_y, pred_noise_y, score_x_and_z_given_ye)
        pred_y_guided_by_x_and_z = self.convert_noise_estimate_to_signal(noisy_y, t_tensor_y, pred_noise_y_guided_by_x_and_z)
        
        return self.sample_from_reverse_operator(self.denoiser_y, noisy_y, pred_y_guided_by_x_and_z, t_tensor_y)
        

    def reverse_one_timestep_x(self, noisy_x, t, y_sample):
        """
        Reverse one timestep in X, with Y guidance. This is used for inference/sampling.
        Args:
            noisy_x: the noisy input at the current timestep
            gamma: the noise level at the current timestep
            y_sample: the sample of Y to condition on
        Returns:
            pred_x: the predicted clean signal at the current timestep
        """
        # detach from the previous timestep's graph so the reverse loop doesn't
        # accumulate one huge computation graph across all timesteps
        noisy_x = noisy_x.detach().requires_grad_(True)

        B, *_ = noisy_x.shape  # batch size
        t_tensor_x = torch.Tensor([t]).repeat_interleave(B, dim=0).long().to(noisy_x.device)

        mu_y_from_xg, logvar_y_from_xg = self.recnet.x_to_y(noisy_x)

        gamma = self.extract(self.denoiser_x.one_minus_alpha_bars, t_tensor_x, noisy_x.shape)

        pred_noise_x = self.denoiser_x.denoiser(noisy_x, t_tensor_x)

        # get the guidance vector ∇_{xg} log p(y_clean | xg)
        score_y_given_xg = self.compute_score_y_given_xg(mu_y_from_xg, logvar_y_from_xg, y_sample, noisy_x)

        # get the score in X
        pred_noise_x_guided = pred_noise_x - gamma.sqrt() * score_y_given_xg
        pred_x_guided_by_y = self.convert_noise_estimate_to_signal(noisy_x, t_tensor_x, pred_noise_x_guided)

        return self.sample_from_reverse_operator(self.denoiser_x, noisy_x, pred_x_guided_by_y, t_tensor_x)


    def conditional_generation(self, N:int, target_x: torch.Tensor):
        # get the forward pass through the encoder
        mu_y_from_x, logvar_y_from_x, mu_z_from_x, logvar_z_from_x = self.recnet.x_to_y_to_z(target_x)
        z_sample_up_from_x = self.recnet.sample(mu_z_from_x, logvar_z_from_x)

        # y denoiser with x, z guidance
        self.noisy_y = torch.randn((N, self.recnet.latent_dim_y)).to(target_x.device).requires_grad_(True)
        
        for t in range(self.num_timesteps_y - 1, -1, -1):
            self.noisy_y = self.reverse_one_timestep_y(self.noisy_y, t, mu_y_from_x, logvar_y_from_x, z_sample_up_from_x)

        y_sample_down_from_x_and_z = self.noisy_y

        # x denoiser with y guidance
        self.noisy_x = torch.randn((N, 1, self.recnet.grid_size, self.recnet.grid_size)).to(target_x.device).requires_grad_(True)

        for t in range(self.num_timesteps_x - 1, -1, -1):
            self.noisy_x = self.reverse_one_timestep_x(self.noisy_x, t, y_sample_down_from_x_and_z)

        return self.noisy_x.detach()
    

    #TODO: get the guided prediction in Y, then use it to compute the guided score in Z for better estimation of the Z rate



    ################ unconditional generation ###################


    def predict_mu_t(self, x_t, denoiser, pred_epsilon, timestep):
        alpha = self.extract(denoiser.alphas, timestep, x_t.shape)
        sqrt_alpha = self.extract(denoiser.sqrt_alphas, timestep, x_t.shape)
        sqrt_one_minus_alpha_bar = self.extract(denoiser.sqrt_one_minus_alpha_bars, timestep, x_t.shape)

        # denoise at time t, utilizing predicted noise
        mu_t_minus_1 = 1 / sqrt_alpha * (x_t - (1 - alpha) / sqrt_one_minus_alpha_bar * pred_epsilon)
        return mu_t_minus_1


    def reverse_one_timestep_unconditional(self, noisy_signal, denoiser, t, device):
        """denoiser only"""
        B, *_ = noisy_signal.shape  # batch size
        timestep = torch.Tensor([t]).repeat_interleave(B, dim=0).to(torch.long).to(device)

        if t > 1:
            z = torch.randn_like(noisy_signal, device=device)
        else:
            z = torch.zeros_like(noisy_signal, device=device)

        pred_epsilon = denoiser.denoiser(noisy_signal, timestep)  # = score of p(x_t|x_t+1)

        # use the total predicted noise to denoise the image
        transition_mean = self.predict_mu_t(noisy_signal, denoiser, pred_epsilon, timestep)

        sqrt_beta = self.extract(denoiser.sqrt_betas, timestep, noisy_signal.shape)

        # and then add noise to this image again
        noisy_signal = transition_mean + sqrt_beta * z
        return noisy_signal



    @torch.no_grad()
    def unconditional_sample_x(self, N):
        """generate unconditional samples using the specified denoiser"""
        device = self.denoiser_x.denoiser.input_proj.weight.device
        denoiser = self.denoiser_x

        # x denoiser only
        self.noisy_x = torch.randn((N, 1, self.recnet.grid_size, self.recnet.grid_size)).to(device).requires_grad_(True)

        for t in range(self.num_timesteps_x - 1, -1, -1):
            self.noisy_x = self.reverse_one_timestep_unconditional(self.noisy_x, denoiser, t, device)

        return self.noisy_x.detach()
    
    
    def unconditional_sample_y(self, N):
        """generate unconditional samples using the specified denoiser"""
        device = self.denoiser_x.denoiser.input_proj.weight.device

        denoiser = self.denoiser_y
        dim = self.recnet.latent_dim_y

        # x denoiser only
        self.noisy_y = torch.randn((N, dim)).to(device).requires_grad_(True)

        for t in range(self.num_timesteps_y - 1, -1, -1):
            self.noisy_y = self.reverse_one_timestep_unconditional(self.noisy_y, denoiser, t, device)

        return self.noisy_y.detach()


    ############# stage 1 and stage 2 generation ##############
    def stage2_generation(self, N:int, target_input:torch.Tensor):
        device = target_input.device
        B = target_input.shape[0]

        # sample y from clean x
        mu_y_from_x, logvar_y_from_x = self.recnet.x_to_y(target_input)
        y_sample_up_from_x = self.recnet.sample(mu_y_from_x, logvar_y_from_x)

        # x denoiser with y guidance
        self.noisy_x = torch.randn((N, 1, self.recnet.grid_size, self.recnet.grid_size)).to(target_input.device).requires_grad_(True)

        for t in range(self.num_timesteps_x - 1, -1, -1):
            self.noisy_x = self.reverse_one_timestep_x(self.noisy_x, t, y_sample_up_from_x)

        return self.noisy_x.detach()