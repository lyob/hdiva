import math

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """sinusoidal position embedding"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, time_dim, activation):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.act = nn.SiLU() if activation == "silu" else nn.GELU()

    def forward(self, x, t_emb):
        return self.act(self.block(x) + self.time_proj(t_emb))


class Denoiser(nn.Module):
    """MLP denoiser for 2D inputs (e.g. latent variables from a VAE).

    Args:
        input_dim:   dimensionality of the noisy input z (default 2)
        hidden_dim:  width of hidden layers
        num_blocks:  number of residual blocks
        time_dim:    dimensionality of the sinusoidal time embedding
    """

    def __init__(self, input_dim=2, hidden_dim=128, num_blocks=4, time_dim=40, activation="silu"):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, time_dim, activation) for _ in range(num_blocks)])
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, input_dim)  noisy latent
        t: (B,)             noise level / timestep
        returns: (B, input_dim)  predicted noise
        """
        t_emb = self.time_emb(t)       # (B, time_dim)
        h = self.input_proj(x)         # (B, hidden_dim)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.output_proj(h)     # (B, input_dim)
