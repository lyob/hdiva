import numpy as np
import torch

from b_models.sami.sami_module import SAMI


# ------------------------------ inference class ----------------------------- #
class SAMI_Inference_MLP(SAMI):
    """SAMI inference for non-image (vector) data.

    Same sampler as `sami_inference.SAMI_Inference`, but the state is a flat
    vector of shape (B, input_dim) instead of an image (B, C, H, W), and the
    denoiser / infnet are the MLPs in `b_models.base_modules.mlp`
    (`MLPDenoiser`, `MLPEncoder`), which take (x, t) and return plain tensors.
    """

    def __init__(
        self,
        sami,
        config,
        device: torch.device | str | None = None,
    ):
        super(SAMI_Inference_MLP, self).__init__(
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
        self.input_dim = config.input_dim
        self.device: torch.device | str = (
            device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.to(self.device)
        self.n_times = config.num_timesteps
        self.errors = torch.empty(self.n_times)
        self.prior_score_magnitudes = torch.empty(self.n_times)
        self.posterior_score_magnitudes = torch.empty(self.n_times)
        self.guidance_score_magnitudes = torch.empty(self.n_times)
        self.prior_score_std = torch.empty(self.n_times)
        self.posterior_score_std = torch.empty(self.n_times)
        self.guidance_score_std = torch.empty(self.n_times)
        self.gammas = torch.empty(self.n_times)

    # ----------------------- helpers ----------------------- #
    def sample_shape(self, N: int):
        """shape of the diffusion state for N samples: (N, input_dim)"""
        return (N, self.input_dim)

    def make_timestep(self, t: int, B: int):
        return torch.full((B,), int(t), dtype=torch.long, device=self.device)

    def prepare_x(self, x: torch.Tensor):
        """move to device, make it (B, input_dim) and track gradients"""
        x = x.clone().to(self.device).reshape(x.shape[0], -1)
        return x.detach().requires_grad_(True)

    # ----------------------- inference ----------------------- #
    def denoise_at_t(self, x_t, pred_epsilon, timestep):
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, timestep, x_t.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep, x_t.shape)
        x0_hat = 1 / sqrt_alpha_bar * (x_t - sqrt_one_minus_alpha_bar * pred_epsilon)
        return x0_hat

    def predict_epsilon_from_x0(self, x_t, x0_hat, timestep):
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, timestep, x_t.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep, x_t.shape)
        return (x_t - sqrt_alpha_bar * x0_hat) / sqrt_one_minus_alpha_bar

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

    def compute_score_inference(self, log_p_z, noisy_x):
        # get the log posterior score
        rec_score = torch.autograd.grad(log_p_z.sum(), noisy_x, retain_graph=False, create_graph=False)[0]
        return rec_score

    def record_scores(self, pred_epsilon, pred_epsilon_guided, noisy_x, timestep):
        """score = -epsilon / sqrt(1 - alpha_bar)"""
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep, noisy_x.shape)
        self.prior_score = (-pred_epsilon / sqrt_one_minus_alpha_bar).detach()
        self.posterior_score = (-pred_epsilon_guided / sqrt_one_minus_alpha_bar).detach()
        self.guidance_score = self.posterior_score - self.prior_score

    def reverse_one_timestep(self, noisy_x, t, z_sample):
        B, *_ = noisy_x.shape  # batch size
        timestep = self.make_timestep(t, B)

        if t > 1:
            z = torch.randn_like(noisy_x).to(self.device)
        else:
            z = torch.zeros_like(noisy_x).to(self.device)

        # get the score of the log posterior
        noisy_x = noisy_x.detach().requires_grad_(True)
        mu_t, logvar_t = self.encode(self.infnet, noisy_x, timestep)
        log_p_z = self.compute_log_posterior(mu_t, logvar_t, z_sample)
        z_score = self.compute_score_inference(log_p_z, noisy_x)
        self.gamma_t = self.extract(self.one_minus_alpha_bars, timestep, noisy_x.shape)[0].squeeze()
        # print(self.gamma_t)

        # get conditional epsilon from the denoiser
        if self.config.parameterization == "noise":
            z_score_weight = self.extract(self.sqrt_one_minus_alpha_bars, timestep, noisy_x.shape)
            weighted_z_score = z_score_weight * z_score

            pred_epsilon = self.denoiser(noisy_x, timestep)  # = score of p(x_t|x_t+1)
            pred_epsilon_guided = pred_epsilon - weighted_z_score

            # get x0 first, then the posterior transition (so x0 can be clipped if wanted)
            with torch.no_grad():
                self.x0_hat_guided = self.denoise_at_t(noisy_x, pred_epsilon_guided, timestep)
                mu_t_minus_1 = self.predict_mu_t_from_x0(noisy_x, self.x0_hat_guided, timestep)
                sqrt_beta_tilde_t = self.extract(self.sqrt_posterior_variance, timestep, noisy_x.shape)
                self.x_t = mu_t_minus_1 + sqrt_beta_tilde_t * z

        elif self.config.parameterization in ("image", "x0"):
            # matches SAMI.forward's "image" branch: x0_hat_guided = x0_hat + (1-alpha_bar)/sqrt(alpha_bar) * score
            sqrt_alpha_bar_t = self.extract(self.sqrt_alpha_bars, timestep, noisy_x.shape)
            one_minus_alpha_bar_t = self.extract(self.one_minus_alpha_bars, timestep, noisy_x.shape)
            z_score_weight = (one_minus_alpha_bar_t / sqrt_alpha_bar_t).clamp(max=50.0)

            x0_hat_unguided = self.denoiser(noisy_x, timestep)
            x0_hat_guided = x0_hat_unguided + z_score_weight * z_score

            pred_epsilon = self.predict_epsilon_from_x0(noisy_x, x0_hat_unguided, timestep)
            pred_epsilon_guided = self.predict_epsilon_from_x0(noisy_x, x0_hat_guided, timestep)

            with torch.no_grad():
                self.x0_hat_guided = x0_hat_guided
                mu_t_minus_1 = self.predict_mu_t_from_x0(noisy_x, self.x0_hat_guided, timestep)
                sqrt_beta_tilde_t = self.extract(self.sqrt_posterior_variance, timestep, noisy_x.shape)
                self.x_t = mu_t_minus_1 + sqrt_beta_tilde_t * z

        elif self.config.parameterization in ("velocity", "velocity1"):
            sqrt_one_minus_alpha_bar_t = self.extract(self.sqrt_one_minus_alpha_bars, timestep, noisy_x.shape)
            sqrt_alpha_bar_t = self.extract(self.sqrt_alpha_bars, timestep, noisy_x.shape)
            z_score_weight = (sqrt_one_minus_alpha_bar_t / sqrt_alpha_bar_t).clamp(max=50.0)

            v_hat_unguided = self.denoiser(noisy_x, timestep)
            v_hat_guided = v_hat_unguided - z_score_weight * z_score

            pred_epsilon = self.predict_epsilon_from_v(v_hat_unguided, noisy_x, timestep)
            pred_epsilon_guided = self.predict_epsilon_from_v(v_hat_guided, noisy_x, timestep)

            with torch.no_grad():
                # x0 = rho_t * x_t - sigma_t * v_t
                self.x0_hat_guided = sqrt_alpha_bar_t * noisy_x - sqrt_one_minus_alpha_bar_t * v_hat_guided
                transition_mean = self.predict_mu_t(noisy_x, pred_epsilon_guided, timestep)
                sqrt_beta_tilde_t = self.extract(self.sqrt_posterior_variance, timestep, noisy_x.shape)
                self.x_t = transition_mean + sqrt_beta_tilde_t * z

        else:
            raise ValueError(f"Invalid parameterization: {self.config.parameterization}")

        """scores"""
        self.record_scores(pred_epsilon, pred_epsilon_guided, noisy_x, timestep)
        

    def conditional_sample(
        self,
        N: int,
        target_x_input: torch.Tensor,
        x_t: torch.Tensor | None = None,
    ):
        """sample N points conditioned on z ~ q(z|target_x)"""
        if x_t is None:
            self.x_t = torch.randn(self.sample_shape(N), device=self.device).requires_grad_(True)
        else:
            self.x_t = self.prepare_x(x_t)
        self.initial_x_t = self.x_t

        self.infnet.eval()
        self.denoiser.eval()

        target_x = self.prepare_x(target_x_input)

        # assuming z_given_xt method
        mu, logvar = self.encode(self.infnet, target_x, None)
        z_sample = self.infnet.sample(mu, logvar)

        self.x0_estimates = torch.empty(self.n_times, *self.sample_shape(N))

        for t in range(self.n_times - 1, -1, -1):
            self.reverse_one_timestep(self.x_t, t, z_sample)

            with torch.no_grad():
                self.x0_estimates[t] = self.x0_hat_guided.clone()
                self.prior_score_magnitudes[t] = torch.mean(torch.norm(self.prior_score, dim=1))
                self.posterior_score_magnitudes[t] = torch.mean(torch.norm(self.posterior_score, dim=1))
                self.guidance_score_magnitudes[t] = torch.mean(torch.norm(self.guidance_score, dim=1))
                self.prior_score_std[t] = torch.std(torch.norm(self.prior_score, dim=1))
                self.posterior_score_std[t] = torch.std(torch.norm(self.posterior_score, dim=1))
                self.guidance_score_std[t] = torch.std(torch.norm(self.guidance_score, dim=1))
                self.gammas[t] = self.gamma_t

    def conditional_sample_from_z(self, N: int, z: torch.Tensor, seed: int = 0, x_init: torch.Tensor | None = None):
        """sample N points conditioned on a latent z given directly"""
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.infnet.eval()
        self.denoiser.eval()

        if x_init is None:
            self.x_t = torch.randn(self.sample_shape(N), device=self.device).requires_grad_(True)
        else:
            self.x_t = self.prepare_x(x_init)
        self.initial_x_t = self.x_t

        z = z.to(self.device)
        self.x0_estimates = torch.empty(self.n_times, *self.sample_shape(N))

        for t in range(self.n_times - 1, -1, -1):
            self.reverse_one_timestep(self.x_t, t, z)
            with torch.no_grad():
                self.x0_estimates[t] = self.x0_hat_guided.clone()

    def add_noise_and_denoise(self, target_x_input, t: int = 10):
        """one-shot: noise the input to level t, then take a single guided x0 estimate"""
        assert t > 0, "t must be greater than 0"
        assert t < self.n_times, "t must be less than the number of timesteps"

        self.infnet.eval()
        self.denoiser.eval()

        target_x = self.prepare_x(target_x_input)
        B, *_ = target_x.shape
        timestep = self.make_timestep(t, B)

        # add noise to the data
        noisy_x, noise = self.make_noisy(target_x, timestep)
        noisy_x = noisy_x.requires_grad_(True)

        # assuming z_given_xt method
        mu, logvar = self.encode(self.infnet, target_x, None)
        z_sample = self.infnet.sample(mu, logvar)

        mu_t, logvar_t = self.encode(self.infnet, noisy_x, timestep)
        log_p_z = self.compute_log_posterior(mu_t, logvar_t, z_sample)
        z_score = self.compute_score_inference(log_p_z, noisy_x)

        if self.config.parameterization == "noise":
            z_score_weight = self.extract(self.sqrt_one_minus_alpha_bars, timestep, noisy_x.shape)
            weighted_z_score = z_score_weight * z_score

            # get conditional epsilon from the denoiser
            pred_epsilon = self.denoiser(noisy_x, timestep)  # = score of p(x_t|x_t+1)
            pred_epsilon_guided = pred_epsilon - weighted_z_score
            x0_hat = self.denoise_at_t(noisy_x, pred_epsilon_guided, timestep)

        elif self.config.parameterization in ("image", "x0"):
            sqrt_alpha_bar_t = self.extract(self.sqrt_alpha_bars, timestep, noisy_x.shape)
            one_minus_alpha_bar_t = self.extract(self.one_minus_alpha_bars, timestep, noisy_x.shape)
            z_score_weight = one_minus_alpha_bar_t / sqrt_alpha_bar_t

            x0_hat_unguided = self.denoiser(noisy_x, timestep)
            x0_hat = x0_hat_unguided + z_score_weight * z_score
            pred_epsilon_guided = self.predict_epsilon_from_x0(noisy_x, x0_hat, timestep)

        elif self.config.parameterization in ("velocity", "velocity1"):
            sqrt_one_minus_alpha_bar_t = self.extract(self.sqrt_one_minus_alpha_bars, timestep, noisy_x.shape)
            sqrt_alpha_bar_t = self.extract(self.sqrt_alpha_bars, timestep, noisy_x.shape)
            z_score_weight = sqrt_one_minus_alpha_bar_t / sqrt_alpha_bar_t

            # get conditional epsilon from the denoiser
            v_hat_unguided = self.denoiser(noisy_x, timestep)  # = score of p(x_t|x_t+1)
            v_hat_guided = v_hat_unguided - z_score_weight * z_score
            pred_epsilon_guided = self.predict_epsilon_from_v(v_hat_guided, noisy_x, timestep)

            # get x0 hat from v hat
            x0_hat = sqrt_alpha_bar_t * noisy_x - sqrt_one_minus_alpha_bar_t * v_hat_guided

        else:
            raise ValueError(f"Invalid parameterization: {self.config.parameterization}")

        return x0_hat.detach(), noisy_x.detach(), pred_epsilon_guided.detach()

    def add_noise_and_iteratively_denoise(self, target_x_input, t: int = 10):
        """noise the input to level t, then run the guided reverse chain from t down to 0"""
        assert t > 0, "t must be greater than 0"
        assert t < self.n_times, "t must be less than the number of timesteps"

        self.infnet.eval()
        self.denoiser.eval()

        target_x = self.prepare_x(target_x_input)
        B, *_ = target_x.shape

        # add noise to the data
        timestep = self.make_timestep(t, B)
        noisy_x, noise = self.make_noisy(target_x, timestep)
        self.x_t = noisy_x.requires_grad_(True)

        # assuming z_given_xt method
        mu, logvar = self.encode(self.infnet, target_x, None)
        z_sample = self.infnet.sample(mu, logvar)

        self.x0_estimates = torch.empty(t, *self.sample_shape(B))

        for step in range(t - 1, -1, -1):
            self.reverse_one_timestep(self.x_t, step, z_sample)
            with torch.no_grad():
                self.x0_estimates[step] = self.x0_hat_guided.clone()

    def conditional_sample_switch(
        self,
        initial_x_input,
        target_x_input,
        noise_level: int = 999,
    ):
        """start the chain from `initial_x_input` but guide it with z from `target_x_input`"""
        self.infnet.eval()
        self.denoiser.eval()

        self.x_t = self.prepare_x(initial_x_input)
        self.initial_x_t = self.x_t

        target_x = self.prepare_x(target_x_input)

        # assuming z_given_xt method
        mu, logvar = self.encode(self.infnet, target_x, None)
        z_sample = self.infnet.sample(mu, logvar)

        for t in range(min(noise_level, self.n_times - 1), -1, -1):
            self.reverse_one_timestep(self.x_t, t, z_sample)

    def reverse_one_timestep_denoiser_only(self, noisy_x, t):
        """denoiser only"""
        B, *_ = noisy_x.shape  # batch size
        if t > 1:
            z = torch.randn_like(noisy_x, device=self.device)
        else:
            z = torch.zeros_like(noisy_x, device=self.device)

        timestep = self.make_timestep(t, B)

        if self.config.parameterization == "noise":
            pred_epsilon = self.denoiser(noisy_x, timestep)  # = score of p(x_t|x_t+1)
        elif self.config.parameterization in ("image", "x0"):
            pred_epsilon = self.predict_epsilon_from_x0(noisy_x, self.denoiser(noisy_x, timestep), timestep)
        elif self.config.parameterization in ("velocity", "velocity1"):
            pred_epsilon = self.predict_epsilon_from_v(self.denoiser(noisy_x, timestep), noisy_x, timestep)
        else:
            raise ValueError(f"Invalid parameterization: {self.config.parameterization}")

        # use the total predicted noise to denoise the data
        self.x0_hat = self.denoise_at_t(noisy_x, pred_epsilon, timestep)
        mu_t_minus_1 = self.predict_mu_t(noisy_x, pred_epsilon, timestep)

        sigma = self.extract(self.sqrt_posterior_variance, timestep, noisy_x.shape)

        # and then add noise to this sample again
        self.x_t = mu_t_minus_1 + sigma * z

    @torch.no_grad()
    def unconditional_sample_denoiser_only(self, N: int):
        """sampling using only the denoiser"""
        self.denoiser.eval()

        # start from random noise vector, x_T
        self.x_t = torch.randn(self.sample_shape(N), device=self.device)
        self.x0_estimates = torch.empty(self.n_times, *self.sample_shape(N))

        for t in range(self.n_times - 1, -1, -1):
            self.reverse_one_timestep_denoiser_only(self.x_t, t)
            self.x0_estimates[t] = self.x0_hat.clone()
