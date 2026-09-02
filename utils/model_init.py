from b_models.base_modules.convnet import ConvNetComplex, ConvNetSimple, MLPEncoder
from b_models.base_modules.half_unet import Half_UNet
from b_models.base_modules.unet import UNet
from b_models.ddpm.ddpm_module import DDPM
from b_models.diva.diva_module import DiVA
from b_models.lvae.lvae3 import LadderVAE
from b_models.sami.sami_module import SAMI
from b_models.vae.vae3 import VAE


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
        input_channel=config.input_channel,
        hidden_channels=config.hidden_channels,
        z_dim=config.z_dim,
        num_blocks=config.num_blocks,
    )
    return model


def init_diva_model(config):
    denoiser = UNet(config)
    # if hasattr(config, 'num_layers_rec'):
    if "num_layers_rec" in config.__annotations__:
        if config.convnet_type != "simple":
            infnet = ConvNetComplex(config)
        else:
            infnet = ConvNetSimple(config)
    else:
        infnet = Half_UNet(config)
    model = DiVA(
        denoiser,
        infnet,
        config.noise_schedule,
        config.sigma_minmax,
        config.timestep_dist,
        config.num_timesteps,
        config.reduction,
    )
    return model


def init_ddpm_model(config):
    denoiser = UNet(config)
    model = DDPM(
        denoiser,
        config.noise_schedule,
        config.parameterization,
        config.sigma_minmax,
        config.timestep_dist,
        config.num_timesteps,
        config.reduction,
        config.weighted_mse,
    )
    return model


def init_sami_model(config):
    denoiser = UNet(config)
    if "infnet_type" in config.__dataclass_fields__:
        if config.infnet_type == "mlp":
            infnet = MLPEncoder(config)
        elif config.infnet_type == "half_unet":
            infnet = Half_UNet(config)
        elif config.infnet_type == "convnet":
            if config.convnet_type == "complex":
                infnet = ConvNetComplex(config)
            elif config.convnet_type == "simple":
                infnet = ConvNetSimple(config)
            else:
                raise ValueError("invalid convnet type")
        else:
            raise ValueError("invalid infnet type")
    else:
        raise ValueError("infnet_type not in config")
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
