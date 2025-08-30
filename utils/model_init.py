from b_models.lvae import LadderVAE

def init_lvae(config):
    # Create the model
    model = LadderVAE(
        input_dim=config.input_dim,
        z_dims=config.z_dims,
        channels=config.channels,
        num_blocks=config.num_blocks
    )
    return model
