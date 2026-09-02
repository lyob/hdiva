import numpy as np
import torch
import torch.nn.functional as F

from b_models.sami.sami_module import SAMI


# ------------------------------ inference class ----------------------------- #
class SAMI_Inference(SAMI):
    def __init__(
        self,
        sami,
        config,
    ):
        super(SAMI_Inference, self).__init__(
            sami.denoiser,
            sami.infnet,
            config.noise_schedule,
            config.parameterization,
            config.sigma_minmax,
            config.timestep_dist,
            config.num_timesteps,
            config.reduction,
            config.rate_type,
            config.weighted_mse,
            config.weighted_rate,
        )

        self.config = config
        self.img_C = config.num_channels
        self.img_H = config.image_dim
        self.img_W = config.image_dim
        self.device: torch.device | str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_times = config.num_timesteps
        self.errors = torch.empty(self.n_times)
        self.prior_score_magnitudes = torch.empty(self.n_times)
        self.posterior_score_magnitudes = torch.empty(self.n_times)
        self.guidance_score_magnitudes = torch.empty(self.n_times)
        self.prior_score_std = torch.empty(self.n_times)
        self.posterior_score_std = torch.empty(self.n_times)
        self.guidance_score_std = torch.empty(self.n_times)

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

    # def predict_mu_t_from_x0_hat(self, x0_hat, x_t, timestep):
    #     """
    #     use the formula mu_t-1 = sqrt_alpha_bar_t-1 * (1-alpha_t)/(1-alpha_bar_t) * x0_hat + sqrt_alpha_t (1-alpha_bar_t-1)/(1-alpha_bar_t) * x_t
    #     but make sure that timestep-1 is
    #     """
    #     sqrt_alpha_bar_t_minus_1 = self.extract(self.sqrt_alpha_bars, timestep - 1, x_t.shape)
    #     sqrt_alpha_t = self.extract(self.sqrt_alphas, timestep, x_t.shape)
    #     sqrt_one_minus_alpha_bar_t = self.extract(self.sqrt_one_minus_alpha_bars, timestep, x_t.shape)
    #     sqrt_one_minus_alpha_bar_t_minus_1 = self.extract(self.sqrt_one_minus_alpha_bars, timestep - 1, x_t.shape)
    #     alpha_t = self.extract(self.alphas, timestep, x_t.shape)
    #     mu_t_minus_1 = (
    #         sqrt_alpha_bar_t_minus_1 * (1 - alpha_t) / sqrt_one_minus_alpha_bar_t * x0_hat
    #         + sqrt_alpha_t * sqrt_one_minus_alpha_bar_t_minus_1 / sqrt_one_minus_alpha_bar_t * x_t
    #     )
    #     return mu_t_minus_1

    def predict_epsilon_from_v(self, v_hat, noisy_x, timestep):
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, timestep, v_hat.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep, v_hat.shape)
        return sqrt_alpha_bar * v_hat + sqrt_one_minus_alpha_bar * noisy_x

    def predict_mu_t_from_x0(self, x_t, x0_hat, timestep):
        """
        Compute mu_{t-1} using the posterior mean formula.
        timestep: integer tensor with values in [0, num_timesteps-1]
        """

        # Extract values for timestep t
        sqrt_alpha_t = self.extract(self.sqrt_alphas, timestep, x_t.shape)
        alpha_bar_t = self.extract(self.alpha_bars, timestep, x_t.shape)
        beta_t = self.extract(self.betas, timestep, x_t.shape)

        # Handle alpha_bar_{t-1} with boundary condition
        # At t=0, we want alpha_bar_{-1} = 1.0
        alpha_bar_prev = torch.ones_like(alpha_bar_t)

        # For t > 0, get alpha_bar_{t-1}
        mask = timestep > 0  # Boolean tensor of shape [B]
        if mask.any():
            timestep_prev = timestep - 1
            # Only extract for timesteps > 0
            alpha_bar_prev_extracted = self.extract(self.alpha_bars, timestep_prev, x_t.shape)
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

    # def compute_score_inference(self, log_p_z, noisy_x):
    #     # get the log posterior score
    #     log_p_z.backward(torch.ones_like(log_p_z), retain_graph=True)
    #     rec_score = noisy_x.grad
    #     return rec_score

    def compute_score_inference(self, log_p_z, noisy_x):
        # get the log posterior score
        rec_score = torch.autograd.grad(log_p_z.sum(), noisy_x, retain_graph=False, create_graph=False)[0]
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

        # get conditional epsilon from the denoiser
        if self.config.parameterization == "noise":
            z_score_weight = self.extract(self.sqrt_one_minus_alpha_bars, timestep, noisy_x.shape)
            weighted_z_score = z_score_weight * z_score

            pred_epsilon = self.denoiser(noisy_x, timestep)  # = score of p(x_t|x_t+1)
            pred_epsilon_guided = pred_epsilon - weighted_z_score

            # """calculating x_t-1 from epsilon directly does not let me clip x0"""
            # # get x_t-1 ~ N(mu_t-1, beta_t*I) = p(x_t-1|x_t, z)
            # posterior_transition_mean = self.predict_mu_t(
            #     noisy_x, pred_epsilon_guided, timestep
            # )  # mean of prior transition operator

            # sqrt_beta = self.extract(self.sqrt_betas, timestep, noisy_x.shape)  # std of either transition operator
            # self.x_t = posterior_transition_mean + sqrt_beta * z  # posterior sample
            # # estimate x0
            # self.x0_hat_guided = self.denoise_at_t(noisy_x, pred_epsilon_guided, timestep)

            """calculating x0 from epsilon first, then using it to calculate x_t-1 does let me clip x0"""
            with torch.no_grad():
                self.x0_hat_guided = self.denoise_at_t(noisy_x, pred_epsilon_guided, timestep)
                # self.x0_hat_guided = torch.clamp(self.x0_hat_guided, -1, 1)
                mu_t_minus_1 = self.predict_mu_t_from_x0(noisy_x, self.x0_hat_guided, timestep)
                # sqrt_beta_t = self.extract(self.sqrt_betas, timestep, noisy_x.shape)
                # self.x_t = mu_t_minus_1 + sqrt_beta_t * z
                sqrt_beta_tilde_t = self.extract(self.sqrt_posterior_variance, timestep, noisy_x.shape)
                self.x_t = mu_t_minus_1 + sqrt_beta_tilde_t * z

            """scores"""
            self.prior_score = -pred_epsilon / self.extract(self.sqrt_one_minus_alpha_bars, timestep, noisy_x.shape)
            self.posterior_score = -pred_epsilon_guided / self.extract(
                self.sqrt_one_minus_alpha_bars, timestep, noisy_x.shape
            )
            self.guidance_score = self.posterior_score - self.prior_score

        elif self.config.parameterization == "image":
            alpha_t = self.extract(self.alphas, timestep, noisy_x.shape)
            one_minus_alpha_t = 1 - alpha_t
            sqrt_alpha_t = self.extract(self.sqrt_alphas, timestep, noisy_x.shape)
            z_score_weight = (one_minus_alpha_t / sqrt_alpha_t).clamp(max=50.0)
            weighted_z_score = z_score_weight * z_score

            x0_hat_unguided = self.denoiser(noisy_x, timestep)  # = score of p(x_t|x_t+1)
            self.x0_hat_guided = x0_hat_unguided - weighted_z_score

            sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, timestep, noisy_x.shape)
            sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep, noisy_x.shape)
            self.x_t = sqrt_alpha_bar * self.x0_hat_guided + sqrt_one_minus_alpha_bar * z

        elif self.config.parameterization == "velocity":
            sqrt_one_minus_alpha_bar_t = self.extract(self.sqrt_one_minus_alpha_bars, timestep, noisy_x.shape)
            sqrt_alpha_bar_t = self.extract(self.sqrt_alpha_bars, timestep, noisy_x.shape)
            z_score_weight = (sqrt_one_minus_alpha_bar_t / sqrt_alpha_bar_t).clamp(max=50.0)

            v_hat_unguided = self.denoiser(noisy_x, timestep)
            v_hat_guided = v_hat_unguided - z_score_weight * z_score

            # x0 = rho_t * x_t - sigma_t * v_t
            self.x0_hat_guided = sqrt_alpha_bar_t * noisy_x - sqrt_one_minus_alpha_bar_t * v_hat_guided

            pred_epsilon = self.predict_epsilon_from_v(v_hat_guided, noisy_x, timestep)
            transition_mean = self.predict_mu_t(noisy_x, pred_epsilon, timestep)
            # sqrt_beta = self.extract(self.sqrt_betas, timestep, noisy_x.shape)
            # self.x_t = transition_mean + sqrt_beta * z
            sqrt_beta_tilde_t = self.extract(self.sqrt_posterior_variance, timestep, noisy_x.shape)
            self.x_t = transition_mean + sqrt_beta_tilde_t * z

        # return x_t_minus_1.clamp(-1., 1)
        # return prior_transition_mean, mu_t_minus_1, pred_epsilon, pred_epsilon_guided, z_score, weighted_z_score, x0_hat_unguided, x0_hat_guided

    def conditional_sample(
        self,
        N: int,
        target_image_input: torch.Tensor,
        return_chain=False,
        x_t: torch.Tensor | None = None,
    ):
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

        self.x0_estimates = torch.empty(self.n_times, N, self.img_C, self.img_H, self.img_W)

        # for t in range(0, 1, self.n_times):
        for t in range(self.n_times - 1, -1, -1):
            self.reverse_one_timestep(self.x_t, t, z_sample)
            with torch.no_grad():
                # target_image_input = target_image_input.expand(N, -1, -1, -1)
                # self.errors[t] = torch.nn.functional.mse_loss(self.x0_hat_guided, target_image_input, reduction="mean")
                self.x0_estimates[t] = self.x0_hat_guided.clone()
            # self.prior_score_magnitudes[t] = torch.mean(torch.norm(self.prior_score.view(N, -1), dim=1))
            # self.posterior_score_magnitudes[t] = torch.mean(torch.norm(self.posterior_score.view(N, -1), dim=1))
            # self.guidance_score_magnitudes[t] = torch.mean(torch.norm(self.guidance_score.view(N, -1), dim=1))
            # self.prior_score_std[t] = torch.std(torch.norm(self.prior_score.view(N, -1), dim=1))
            # self.posterior_score_std[t] = torch.std(torch.norm(self.posterior_score.view(N, -1), dim=1))
            # self.guidance_score_std[t] = torch.std(torch.norm(self.guidance_score.view(N, -1), dim=1))

    def conditional_sample_from_z(self, N: int, z: torch.Tensor, seed: int = 0, x_init: torch.Tensor | None = None):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # start from random noise vector, x_0 (for simplicity, x_T declared as x_t instead of x_T)
        if x_init is None:
            self.x_t = torch.randn((N, self.img_C, self.img_H, self.img_W)).to(self.device).requires_grad_(True)
        else:
            self.x_t = x_init.clone().to(self.device).requires_grad_(True)
        self.initial_x_t = self.x_t

        for t in range(self.n_times - 1, -1, -1):
            self.reverse_one_timestep(self.x_t, t, z)

    def add_noise_and_denoise(self, target_image_input, t: int = 10, x_t=None):
        assert t > 0, "t must be greater than 0"
        assert t < self.n_times, "t must be less than the number of timesteps"

        # add noise to image
        timestep = torch.Tensor([t]).repeat_interleave(1, dim=0).long().to(self.device)
        noisy_x, noise = self.make_noisy(target_image_input, timestep)
        noisy_x = noisy_x.requires_grad_(True)

        # start from random noise vector, x_0 (for simplicity, x_T declared as x_t instead of x_T)
        self.infnet.eval()

        target_image = target_image_input.clone()
        target_image = target_image.to(self.device).requires_grad_(True)

        # assuming z_given_xt method
        mu, logvar = self.infnet(target_image)
        z_sample = self.infnet.sample(mu, logvar)

        mu_t, logvar_t = self.infnet(noisy_x)
        log_p_z = self.compute_log_posterior(mu_t, logvar_t, z_sample)
        z_score = self.compute_score_inference(log_p_z, noisy_x)

        if self.config.parameterization == "noise":
            z_score_weight = self.extract(self.sqrt_one_minus_alpha_bars, timestep, noisy_x.shape)
            weighted_z_score = z_score_weight * z_score

            # get conditional epsilon from the denoiser
            pred_epsilon = self.denoiser(noisy_x, timestep)  # = score of p(x_t|x_t+1)
            pred_epsilon_guided = pred_epsilon - weighted_z_score
            x0_hat = self.denoise_at_t(noisy_x, pred_epsilon_guided, timestep).detach()

        elif self.config.parameterization == "velocity":
            sqrt_one_minus_alpha_bar_t = self.extract(self.sqrt_one_minus_alpha_bars, timestep, noisy_x.shape)
            sqrt_alpha_bar_t = self.extract(self.sqrt_alpha_bars, timestep, noisy_x.shape)
            z_score_weight = sqrt_one_minus_alpha_bar_t / sqrt_alpha_bar_t

            # get conditional epsilon from the denoiser
            v_hat_unguided = self.denoiser(noisy_x, timestep)  # = score of p(x_t|x_t+1)
            v_hat_guided = v_hat_unguided - z_score_weight * z_score
            pred_epsilon_guided = self.predict_epsilon_from_v(v_hat_guided, noisy_x, timestep)

            # get x0 hat from v hat
            x0_hat = sqrt_alpha_bar_t * noisy_x - sqrt_one_minus_alpha_bar_t * v_hat_guided

        return x0_hat.detach(), noisy_x.detach(), pred_epsilon_guided.detach()

    def add_noise_and_iteratively_denoise(self, target_image_input, t: int = 10):
        assert t > 0, "t must be greater than 0"
        assert t < self.n_times, "t must be less than the number of timesteps"

        # add noise to image
        timestep = torch.Tensor([t]).repeat_interleave(1, dim=0).long().to(self.device)
        noisy_x, noise = self.make_noisy(target_image_input, timestep)
        noisy_x = noisy_x.requires_grad_(True)
        self.x_t = noisy_x

        # start from random noise vector, x_0 (for simplicity, x_T declared as x_t instead of x_T)
        self.infnet.eval()

        target_image = target_image_input.clone()
        target_image = target_image.to(self.device).requires_grad_(True)

        # assuming z_given_xt method
        mu, logvar = self.infnet(target_image)
        z_sample = self.infnet.sample(mu, logvar)
        self.x0_estimates = torch.empty(t, 1, self.img_C, self.img_H, self.img_W)

        for t in range(t - 1, -1, -1):
            self.reverse_one_timestep(self.x_t, t, z_sample)
            with torch.no_grad():
                self.x0_estimates[t] = self.x0_hat_guided.clone()

    def conditional_sample_switch(
        self,
        N,
        initial_image_input,
        target_image_input,
        return_chain=False,
        noise_level=999,
    ):
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

    def reverse_one_timestep_denoiser_only(self, noisy_x, t):
        """denoiser only"""
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
        self.x_t = mu_t_minus_1 + sigma * z

    def unconditional_sample_denoiser_only(self, N):
        """sampling using only the denoiser"""
        # start from random noise vector, x_0 (for simplicity, x_T declared as x_t instead of x_T)
        self.denoiser.eval()

        # start from random noise vector, x_T
        self.x_t = torch.randn((N, self.img_C, self.img_H, self.img_W), device=self.device)

        for t in range(self.n_times - 1, -1, -1):
            # x_t, pred_epsilon = self.reverse_one_timestep_denoiser_only(x_t, t)
            self.reverse_one_timestep_denoiser_only(self.x_t, t)
