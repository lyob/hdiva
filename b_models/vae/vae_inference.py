import numpy as np
import torch
import torch.nn.functional as F

from b_models.vae.vae3 import VAE


# ------------------------------ inference class ----------------------------- #
class VAE_Inference(VAE):
    def __init__(
        self,
        config,
    ):
        super(VAE_Inference, self).__init__(
            input_dim=config.image_dim,
            input_channel=config.input_channel,
            hidden_channels=config.hidden_channels,
            z_dim=config.z_dim,
            num_blocks=config.num_blocks,
            use_kl=config.use_kl,
            reduction=config.reduction,
        )

        self.config = config
        self.device: torch.device | str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------- inference ----------------------- #
    def conditional_sample(
        self,
        target_image_input: torch.Tensor,
    ):
        # d = target_image_input.clone()
        # for layer in self.encoder:
        #     d = layer(d)

        # z_params = self.compress_block(d)

        # # sample from top-level latent
        # z_top = self.reparameterization_trick(z_params)

        # z_top = z_top.unsqueeze(-1).unsqueeze(-1)
        # d = self.expand_block(z_top)

        # for layer in self.decoder:
        #     d = layer(d)
        d = self.forward(target_image_input)
        return d

    def conditional_sample_from_z(self, z: torch.Tensor, seed: int = 0):
        torch.manual_seed(seed)
        np.random.seed(seed)

        z_top = z.unsqueeze(-1).unsqueeze(-1)
        d = self.expand_block(z_top)

        for layer in self.decoder:
            d = layer(d)
        return d

    def encode(self, x: torch.Tensor):
        d = x
        for layer in self.encoder:
            d = layer(d)

        z_params = self.compress_block(d)
        mu, lv = z_params.chunk(2, dim=1)
        var = torch.exp(lv).clamp(min=1e-6)

        # sample from top-level latent
        z_top = self.reparameterization_trick(z_params)

        return z_top, mu, var
