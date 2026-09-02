import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RecNetConv(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.in_channels: int = args.in_channels
        self.base_channels: int = args.num_channels_rec
        self.max_channels: int = args.max_channels_rec
        self.grid_size: int = args.grid_size
        self.latent_dim_y: int = args.latent_dim_y
        self.latent_dim_z: int = args.latent_dim_z
        self.num_layers_x_to_y: int = args.num_layers_rec_x_to_y
        self.num_layers_y_to_z: int = args.num_layers_rec_y_to_z
        nonlin = args.activation_rec
        bias = args.bias_rec
        num_groups = args.num_groups_rec
        aap_output_size = args.adaptive_avg_pool_output_size
        kernel_size = 4

        if nonlin == "relu":
            self.nonlin = nn.ReLU()
        elif nonlin == "gelu":
            self.nonlin = nn.GELU()
        elif nonlin == "silu":
            self.nonlin = nn.SiLU()
        else:
            raise ValueError('nonlin must be specified, e.g. "relu" or "gelu"')

        self.norm = lambda c: nn.GroupNorm(num_groups, c, affine=bias)

        assert self.num_layers_x_to_y > 0, "num_layers_x_to_y must be greater than 0"
        assert self.num_layers_y_to_z > 0, "num_layers_y_to_z must be greater than 0"

        # geometric channel schedule (doubling per layer, capped) shared across both
        # conv stacks, so the x_to_y -> y_to_z junction channel count is consistent
        # whether reached via the backbone or via the standalone y_to_z path
        total_layers = self.num_layers_x_to_y + self.num_layers_y_to_z
        self.channels: list[int] = [min(self.base_channels * (2**i), self.max_channels) for i in range(total_layers)]

        # conv backbone (x → feature map), each block halves spatial dims
        self.layers_x_to_y = nn.ModuleList()
        for i in range(self.num_layers_x_to_y):
            c_in = self.in_channels if i == 0 else self.channels[i - 1]
            self.layers_x_to_y.append(
                nn.Sequential(
                    nn.Conv2d(c_in, self.channels[i], kernel_size=kernel_size, stride=2, padding=1, bias=bias),
                    self.norm(self.channels[i]),
                    self.nonlin,
                )
            )

        y_channels = self.channels[self.num_layers_x_to_y - 1]

        # pool after layers_x_to_y, feeds directly into the y distribution head
        self.y_pool_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((aap_output_size, aap_output_size)),
            nn.Flatten(),
        )
        y_pooled_dims = y_channels * aap_output_size**2

        # conv backbone (feature map → refined features for z), stride=2 for further downsampling
        self.layers_y_to_z = nn.ModuleList()
        for j in range(self.num_layers_y_to_z):
            i = self.num_layers_x_to_y + j
            self.layers_y_to_z.append(
                nn.Sequential(
                    nn.Conv2d(self.channels[i - 1], self.channels[i], kernel_size=kernel_size, stride=2, padding=1, bias=bias),
                    self.norm(self.channels[i]),
                    self.nonlin,
                )
            )

        z_channels = self.channels[-1]

        # pool after layers_y_to_z, feeds directly into the z distribution head
        self.z_pool_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((aap_output_size, aap_output_size)),
            nn.Flatten(),
        )
        z_pooled_dims = z_channels * aap_output_size**2

        # output heads: single linear layer straight from pooled conv features
        self.head_y = nn.Linear(y_pooled_dims, self.latent_dim_y * 2, bias=bias)
        self.head_z = nn.Linear(z_pooled_dims, self.latent_dim_z * 2, bias=bias)

        # spatial size of the feature map at the x_to_y -> y_to_z junction, i.e. the
        # output of layers_x_to_y, so that the standalone y_to_z path (below) can
        # reconstruct a feature map of matching shape and share layers_y_to_z's weights
        y_grid_size = self.grid_size
        for _ in range(self.num_layers_x_to_y):
            y_grid_size = (y_grid_size + 2 * 1 - kernel_size) // 2 + 1
        self.y_grid_size = y_grid_size

        # projects latent y back to a spatial feature map (matching the backbone's
        # junction shape) for the standalone y_to_z path
        self.input_head_y = nn.Linear(self.latent_dim_y, y_channels * y_grid_size**2, bias=bias)

        # Running scale for unit-second-moment normalization of the y latent before
        # it enters the y-diffusion. Updated as an EMA of the *clean*-encoder second
        # moment (see normalize_y / _update_y_scale) and treated as a constant at
        # apply time (no grad), like BatchNorm running stats. Keeps y ~ O(1) so the
        # y-diffusion noise schedule (which assumes unit-scale data) stays matched
        # under joint training, without imposing a KL/prior on y. A single scalar
        # (not per-dim) so the low-dim y geometry is preserved.
        self.y_scale_momentum: float = getattr(args, "y_scale_momentum", 0.99)
        self.register_buffer("y_scale", torch.ones(()))
        self.register_buffer("y_scale_initialized", torch.zeros((), dtype=torch.bool))

    def _conv_x_to_y(self, x: Tensor) -> Tensor:
        """Run layers_x_to_y; returns feature map (B, num_channels, H', W')."""
        for layer in self.layers_x_to_y:
            x = layer(x)
        return x

    def _conv_y_to_z(self, d: Tensor) -> Tensor:
        """Run layers_y_to_z on a feature map; returns feature map (B, num_channels, H', W')."""
        for layer in self.layers_y_to_z:
            d = layer(d)
        return d

    @torch.no_grad()
    def _update_y_scale(self, mu: Tensor, logvar: Tensor) -> None:
        """EMA update of the y-latent scale from the (clean) encoder second moment.
        RMS = sqrt(E[mu^2 + var]) computed analytically (no sampling)."""
        var = logvar.clamp(-8, 8).exp()
        rms = (mu.pow(2) + var).mean().clamp_min(1e-8).sqrt()
        if self.y_scale_initialized:
            m = self.y_scale_momentum
            self.y_scale.mul_(m).add_((1.0 - m) * rms)
        else:  # initialize directly so we don't ramp slowly from 1.0
            self.y_scale.copy_(rms)
            self.y_scale_initialized.fill_(True)

    def normalize_y(self, mean: Tensor, logvar: Tensor, update_stats: bool = False) -> tuple[Tensor, Tensor]:
        """Scale the y distribution to ~unit second moment: y_norm = y / y_scale,
        i.e. mean -> mean / s, logvar -> logvar - 2 log s. y_scale is a detached
        running stat, so gradients flow through mean/logvar but not the scale.
        Pass update_stats=True only on the *clean*-x encoding (the canonical scale
        source) and only during training."""
        if update_stats and self.training:
            self._update_y_scale(mean, logvar)
        s = self.y_scale.clamp_min(1e-8)
        return mean / s, logvar - 2.0 * torch.log(s)

    def x_to_y(self, x: Tensor, update_stats: bool = False) -> tuple[Tensor, Tensor]:
        """Encode image x to mean and logvar of the (normalized) y."""
        d = self._conv_x_to_y(x)
        y = self.head_y(self.y_pool_proj(d))
        mean, logvar = y.chunk(2, dim=-1)
        return self.normalize_y(mean, logvar, update_stats)

    def y_to_z(self, y: Tensor) -> tuple[Tensor, Tensor]:
        """Encode latent sample y to mean and logvar of z."""
        # project y to a (B, num_channels, y_grid_size, y_grid_size) feature map,
        # matching the shape produced by layers_x_to_y, to feed into conv layers
        B = y.shape[0]
        d = self.input_head_y(y).view(B, -1, self.y_grid_size, self.y_grid_size)
        d = self._conv_y_to_z(d)
        z = self.head_z(self.z_pool_proj(d))
        mean, logvar = z.chunk(2, dim=-1)
        return mean, logvar

    def x_to_y_to_z(self, x: Tensor, update_stats: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Encode image x to (y_mean, y_logvar, z_mean, z_logvar) sharing the backbone.
        y is normalized; z is left in its native scale (it is not diffused)."""
        d = self._conv_x_to_y(x)
        y = self.head_y(self.y_pool_proj(d))
        y_mean, y_logvar = y.chunk(2, dim=-1)
        y_mean, y_logvar = self.normalize_y(y_mean, y_logvar, update_stats)

        d = self._conv_y_to_z(d)
        z = self.head_z(self.z_pool_proj(d))
        z_mean, z_logvar = z.chunk(2, dim=-1)
        return y_mean, y_logvar, z_mean, z_logvar

    def x_to_z(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode image x directly to mean and logvar of z, skipping y sampling."""
        d = self._conv_x_to_y(x)
        d = self._conv_y_to_z(d)
        z = self.head_z(self.z_pool_proj(d))
        mean, logvar = z.chunk(2, dim=-1)
        return mean, logvar

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.x_to_z(x)

    def sample(self, mu: Tensor, logvar: Tensor) -> Tensor:
        # clamp logvar so a runaway encoder variance can't blow up the sampled
        # latent past what the running y-scale can correct. Matches the (-8, 8)
        # range used in the log-posterior / y-scale statistics elsewhere.
        std = (0.5 * logvar.clamp(-8.0, 8.0)).exp()
        eps = torch.randn_like(std)
        return mu + eps * std
