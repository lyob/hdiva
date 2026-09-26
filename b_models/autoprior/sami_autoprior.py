'''SAMI (infomax) training of the pretrained autoprior denoiser + encoder.

The autoprior modules are used as-is so the pretrained weights load without renaming;
this file only adapts their interfaces to what `SAMI` expects.
'''
import os

import torch

from b_models.autoprior.celeba_denoisers_t_as_input import UNet as UNet_t_as_input
from b_models.autoprior.celeba_encoders import HalfUNet
from b_models.sami.sami_module import SAMI


class AutoPriorInfNet(HalfUNet):
    '''autoprior HalfUNet encoder with the (mu, logvar) interface SAMI expects.

    Subclassing (rather than wrapping) keeps the state dict keys identical to the autoprior encoder.
    '''
    def forward(self, x: torch.Tensor):
        _, mu, var = super().forward(x)
        # the autoprior clamped var at 1e-6 when computing log q(z|x_t), so keep the same floor
        logvar = torch.log(var.clamp(min=1e-6))
        return mu, logvar

    def sample(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + torch.randn_like(std) * std


class SAMI_AutoPrior(SAMI):
    '''SAMI with the autoprior's noise schedule (cosine offset s=0.01 instead of 0.008).'''
    def get_cosine_in_alpha_bar(self, timesteps, s=0.008):
        return super().get_cosine_in_alpha_bar(timesteps, s=0.01)


def init_sami_autoprior_model(config):
    denoiser = UNet_t_as_input(config)
    infnet = AutoPriorInfNet(config)

    # the autoprior UNet builds a time_emb Linear in every conv block but never uses it in forward();
    # freeze them so DDP doesn't fail on parameters that never receive a gradient
    for name, param in denoiser.named_parameters():
        if ".time_emb." in name:
            param.requires_grad = False

    model = SAMI_AutoPrior(
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


def get_autoprior_weights_path(config):
    '''path to the pretrained autoprior weights, downloading the wandb artifact if it isn't already local'''
    artifact_name = f"{config.pretrained_model_name}:{config.pretrained_artifact_id}"
    artifact_dir = os.path.join(config.project_dir, "artifacts", artifact_name)
    weights_path = os.path.join(artifact_dir, "final_model_weights.pt")
    if not os.path.exists(weights_path):
        import wandb
        artifact = wandb.Api().artifact(f"{config.pretrained_entity}/{config.pretrained_model_name}/{artifact_name}")
        artifact.download(root=artifact_dir)
    return weights_path


def load_autoprior_weights_into_sami(model: SAMI_AutoPrior, weights_path: str):
    '''load an autoprior state dict ("denoiser.*", "encoder.*") into a SAMI_AutoPrior model ("denoiser.*", "infnet.*")'''
    state_dict = torch.load(weights_path, map_location="cpu")
    denoiser_sd = {k.removeprefix("denoiser."): v for k, v in state_dict.items() if k.startswith("denoiser.")}
    encoder_sd = {k.removeprefix("encoder."): v for k, v in state_dict.items() if k.startswith("encoder.")}
    model.denoiser.load_state_dict(denoiser_sd, strict=True)
    model.infnet.load_state_dict(encoder_sd, strict=True)
    print(f"Loaded autoprior weights from {weights_path}")
    return model
