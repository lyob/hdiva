from b_models.lvae.lvae3 import LadderVAE
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
        z_dim=config.z_dim,
        channels=config.channels,
        num_blocks=config.num_blocks,
        dec_skip_connections=config.dec_skip_connections,
        use_nll_loss=config.use_nll_loss,
    )
    return model