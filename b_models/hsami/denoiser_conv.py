import math

import torch
import torch.nn as nn
from torch import Tensor


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


class DenoiserConv(nn.Module):
    """
    Minimal UNet denoiser for image inputs.

    args attributes used:
        in_channels             : input image channels
        num_channels_den_x      : number of conv channels throughout the UNet
        num_layers_den_x        : number of encoder/decoder levels
        time_dim_den_x          : dimensionality of the sinusoidal time embedding
        activation_den_x    : "relu" | "gelu" | "silu"
        bias_den_x              : bool
    """

    def __init__(self, args):
        super().__init__()
        C: int = args.num_channels_den_x
        bias: bool = args.bias_den_x
        time_dim: int = args.time_dim_den_x
        num_layers: int = args.num_layers_den_x
        nonlin_name: str = args.activation_den_x

        if nonlin_name == "relu":
            self.nonlin = nn.ReLU()
        elif nonlin_name == "gelu":
            self.nonlin = nn.GELU()
        elif nonlin_name == "silu":
            self.nonlin = nn.SiLU()
        else:
            raise ValueError('activation must be "relu", "gelu", or "silu"')

        self.time_emb = TimeEmbedding(time_dim)
        self.time_proj = nn.Linear(time_dim, C, bias=bias)

        self.input_proj = nn.Conv2d(args.in_channels, C, kernel_size=3, padding=1, bias=bias)

        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1, bias=bias),
                self.nonlin,
            )
            for _ in range(num_layers)
        ])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, padding=1, bias=bias),
            self.nonlin,
        )

        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(C, C, kernel_size=4, stride=2, padding=1, bias=bias),
                self.nonlin,
            )
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Conv2d(C, args.in_channels, kernel_size=1, bias=bias)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        x : (B, in_channels, H, W)  noisy image
        t : (B,)                    timestep
        returns (B, in_channels, H, W)  predicted noise
        """
        t_feat = self.time_proj(self.time_emb(t)).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

        h = self.input_proj(x)

        skips = []
        for enc in self.encoder:
            h = enc(h + t_feat)
            skips.append(h)

        h = self.bottleneck(h + t_feat)

        for dec in self.decoder:
            skip = skips.pop()
            # The stride-2 encoder rounds odd sizes up while ConvTranspose
            # upsamples by exactly 2x, so h can overshoot the skip by one
            # pixel when an intermediate spatial dim was odd. Crop to match.
            h = h[..., : skip.shape[-2], : skip.shape[-1]]
            h = dec(h + skip + t_feat)

        # The final upsample may likewise overshoot the input resolution.
        h = h[..., : x.shape[-2], : x.shape[-1]]

        return self.output_proj(h)
