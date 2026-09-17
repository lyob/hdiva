import torch
import json
import os
import sys
import numpy as np

notebook_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(notebook_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from b_models.configs.sami_simple_config import Config
from b_models.base_modules.mlp import MLPEncoder, MLPDenoiser
from b_models.sami.sami_module import SAMI

def initialize_sami():
    config = Config()
    denoiser = MLPDenoiser(config)
    infnet = MLPEncoder(config)
    model = SAMI(
        denoiser, 
        infnet, 
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
    return model


def init_sami_from_config(dataset_name:str, model_num:int):
    # init model
    from b_models.configs.sami_simple_config import Config
    from b_models.base_modules.mlp import MLPEncoder, MLPDenoiser
    from b_models.sami.sami_module import SAMI

    load_dir = f"{parent_dir}/c_training/local_weights/{dataset_name}"

    # load config
    config_dir = f"{load_dir}/sami_config_v{model_num:03d}.json" 
    with open(config_dir, 'r') as f:
        config = json.load(f)
    config = Config.from_dict(config)

    # load weights
    weights_dir = f"{load_dir}/sami_weights_v{model_num:03d}.pth"
    # load weights
    weights = torch.load(weights_dir)

    # initialize the model
    denoiser = MLPDenoiser(config)
    infnet = MLPEncoder(config)
    model = SAMI(
        denoiser, 
        infnet, 
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
    return config, model, weights


def get_rd_single_gamma(
        sami_inference, 
        config, 
        target_x, 
        t: int|None = None, 
        rate_eval_method: str|None = 'norm') -> tuple[torch.Tensor, torch.Tensor]:
    """calculate the rate and distortion for a model trained at a specific beta. 
    If `t` is not specified as a parameter, we randomly sample a noise level (i.e. a specific gamma)."""

    sami = sami_inference
    target_x = sami.prepare_x(target_x).detach()
    B, *_ = target_x.shape

    if t != None:
        assert t > 0, "t must be greater than 0"
        assert t < sami.n_times, "t must be less than the number of timesteps"
        timestep = sami.make_timestep(t, B)
    else:
        # timestep = torch.randint(1, config.num_timesteps, (B,))
        timestep = torch.randint(1, config.num_timesteps, (1,)).repeat(target_x.shape[0],)
    # print(timestep)

    sami.infnet.eval()
    sami.denoiser.eval()

    # add noise to the data
    noisy_x, noise = sami.make_noisy(target_x, timestep)
    noisy_x = noisy_x.requires_grad_(True)

    gamma = sami.extract(sami.one_minus_alpha_bars, timestep, target_x.shape)

    # forward pass through encoder
    mu, logvar = sami.encode(sami.infnet, target_x, None)  # clean view == timestep 0
    z_sample = sami.infnet.sample(mu, logvar)

    # get the score via backward pass
    mu_t, logvar_t = sami.encode(sami.infnet, noisy_x, timestep)
    log_posterior = sami.compute_log_posterior(mu_t, logvar_t, z_sample)
    score = sami.compute_score_from_logp(log_posterior, noisy_x)

    # calculate the rate (via norm method)
    if rate_eval_method == 'norm' or rate_eval_method == None:
        rate = score.flatten(1).norm(dim=1).square()
    elif rate_eval_method == 'kl':
        rate = sami.compute_kl_against_zero(mu, logvar)

    # calculate the distortion
    if sami.parameterization == "noise":
        pred_noise = sami.denoiser(noisy_x, timestep)  # = instantaneous entropy
        z_score_weight = gamma.sqrt()
        pred_noise_guided = pred_noise - z_score_weight * score

        target = noise
        prediction = pred_noise_guided

    mse = torch.square(target - prediction).mean(dim=1)

    return mse.detach(), rate.detach(), timestep, gamma.flatten()


def average_vals(input_tuple):
    '''for all elements in the tuple, return their average'''
    return tuple(x.mean().item() for x in input_tuple)

def get_rd(dataloader, sami_inference, model_config, num_iters:int=1, rate_eval_method:str|None='norm'):
    '''calculate the rate and distortion, 
    '''
    all_d = []
    all_r = []

    for i in range(num_iters):
        for idx, batch in enumerate(dataloader):
            batch = batch['data']

            t = None
            d, r, timestep, gamma = get_rd_single_gamma(sami_inference, model_config, batch, t, rate_eval_method)
            mean_d, mean_r = average_vals((d, r))

            all_d.append(mean_d)
            all_r.append(mean_r)

    return float(np.mean(all_d)), float(np.mean(all_r))