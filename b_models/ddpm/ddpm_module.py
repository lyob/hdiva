import torch
import torch.nn as nn

# ------------------------------ unconditional diffusion model ------------------------------ #
class DDPM(nn.Module):
    def __init__(
            self,
            denoiser,
            noise_schedule: str = 'cosine_in_alpha_bar',
            sigma_minmax: tuple[float, float] = (0.0001, .9999),
            timestep_dist: str = 'uniform',
            num_timesteps: int = 1000,
            reduction: str = "mean",
    ):
        super(DDPM, self).__init__()
        self.denoiser = denoiser
        self.num_timesteps = num_timesteps
        self.register_buffer('timesteps', torch.linspace(start=0, end=1, steps=self.num_timesteps))
        self.timestep_dist = timestep_dist
        self.sigma_minmax = sigma_minmax
        self.define_noise_schedule(noise_schedule)
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
    
    def set_schedules_from_sigma(self, sigmas):
        sqrt_one_minus_alpha_bars = sigmas
        one_minus_alpha_bars = sqrt_one_minus_alpha_bars ** 2
        alpha_bars = 1 - one_minus_alpha_bars
        sqrt_alpha_bars = torch.sqrt(alpha_bars)
        alphas = torch.ones_like(alpha_bars)
        for i in range(1, len(alphas)):
            alphas[i] = alpha_bars[i] / alpha_bars[i-1]
        alphas[0] = alphas[1] - (alphas[2] - alphas[1])  # linearly extrapolate
        alphas = alphas
        sqrt_alphas = torch.sqrt(alphas)
        betas = 1 - alphas
        sqrt_betas = torch.sqrt(betas)

        # register buffers
        self.register_buffer('alphas', alphas)
        self.register_buffer('sqrt_alphas', sqrt_alphas)
        self.register_buffer('betas', betas)
        self.register_buffer('sqrt_betas', sqrt_betas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sqrt_alpha_bars', sqrt_alpha_bars)
        self.register_buffer('one_minus_alpha_bars', one_minus_alpha_bars)
        self.register_buffer('sqrt_one_minus_alpha_bars', sqrt_one_minus_alpha_bars)

    def define_noise_schedule(self, noise_schedule):
        '''define beta schedule depending on the distribution of the noise (as defined by the std))'''
        sigma_min, sigma_max = self.sigma_minmax
        
        if noise_schedule == 'cosine_in_alpha_bar':
            # using the schedule outlined in the Nichol and Dhariwal (2021) paper 
            s = 0.01
            f_t = torch.cos((self.timesteps + s)/(1 + s) * torch.pi/2) ** 2
            f_0 = np.cos(s/(1 + s) * np.pi/2) ** 2
            alpha_bars = sigma_min + (sigma_max - sigma_min) * (f_t / f_0)
            sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)
            self.set_schedules_from_sigma(sqrt_one_minus_alpha_bars)
        else:
            raise ValueError(f"Unknown noise_schedule: {noise_schedule}, expected 'cosine_in_alpha_bar'")
    
    def make_noisy(self, x_zeros, t):
        '''perturb x_0 into x_t (i.e., take x_0 samples into forward diffusion kernels)'''
        epsilon = torch.randn_like(x_zeros)
        
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, t, x_zeros.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, t, x_zeros.shape)
        
        noisy_sample = x_zeros * sqrt_alpha_bar + epsilon * sqrt_one_minus_alpha_bar
    
        return noisy_sample.detach(), epsilon

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

    def forward(self, x_zeros:torch.Tensor):
        '''forward pass of the model'''

        B, *_ = x_zeros.shape  # batch size
        
        # (1) randomly choose diffusion time-step
        t = self.sample_timesteps(B, self.num_timesteps, dist=self.timestep_dist, device=x_zeros.device)

        # (2) forward diffusion process: perturb x_zeros with fixed variance schedule
        x_zeros = x_zeros.detach().requires_grad_(True)
        noisy_x, epsilon = self.make_noisy(x_zeros, t)
        noisy_x = noisy_x.detach().requires_grad_(True)
    
        # (7) estimate the noise epsilon: predict epsilon (noise) given perturbed data at diffusion-timestep t.
        pred_epsilon = self.denoiser(noisy_x, t)  # = score of p(x_t|x_t+1)
        
        # (8) set the target and prediction for the model
        target = epsilon
        prediction = pred_epsilon
        return target, prediction

    def compute_loss(self, x_zeros):
        target, prediction = self.forward(x_zeros)
        # Compute the loss
        # mse_loss = F.mse_loss(prediction, target, reduction=self.reduction)
        mse_loss = (prediction - target).pow(2)
        mse_loss = mse_loss.mean() if self.reduction == "mean" else mse_loss.sum(dim=1).mean()
        return mse_loss


