from b_models.lvae.lvae3 import LadderVAE
from b_models.vae.vae3 import VAE
from b_models.base_modules.unet import UNet
from b_models.base_modules.half_unet import Half_UNet
from b_models.base_modules.convnet import ConvNet
from b_models.diva.diva_module import DiVA
from b_models.ddpm.ddpm_module import DDPM

def init_lvae(config):
    # Create the model
    model = LadderVAE(
        input_dim=config.input_dim,
        z_dims=config.z_dims,
        enc_channels=config.enc_channels,
        dec_channels=config.dec_channels,
        num_blocks=config.num_blocks,
        dec_skip_connections=config.dec_skip_connections,
        use_nll_loss=config.use_nll_loss,
    )
    return model

def init_vae(config):
    model = VAE(
        input_dim=config.input_dim,
        z_dim=config.z_dim,
        channels=config.channels,
        num_blocks=config.num_blocks,
        dec_skip_connections=config.dec_skip_connections,
        use_nll_loss=config.use_nll_loss,
    )
    return model


def init_diva_model(config):
    denoiser = UNet(config)
    # if hasattr(config, 'num_layers_rec'):
    if 'num_layers_rec' in config.__annotations__:
        infnet = ConvNet(config)
    else:
        infnet = Half_UNet(config)
    model = DiVA(denoiser, infnet, config.noise_schedule, config.sigma_minmax, config.timestep_dist, config.num_timesteps, config.reduction)
    return model


def init_ddpm_model(config):
    denoiser = UNet(config)
    model = DDPM(denoiser, config.noise_schedule, config.sigma_minmax, config.timestep_dist, config.num_timesteps)
    return model
