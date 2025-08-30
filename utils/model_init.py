from b_models.lvae import LadderVAE
from b_models.vae3 import VAE

def init_lvae(config):
    # Create the model
    model = LadderVAE(
        input_dim=config.input_dim,
        z_dims=config.z_dims,
        channels=config.channels,
        num_blocks=config.num_blocks
    )
    return model

def init_vae(config):
    model = VAE(
        input_dim=config.input_dim,
        z_dim=config.z_dim,
        channels=config.channels,
        num_blocks=config.num_blocks
    )
    return model