import torch
# import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

from b_models.base_modules.mlp import TimeEmbedding


class LinearInfNet(nn.Module):
    def __init__(self, input_dim, z_dim, bias):
        super(LinearInfNet, self).__init__()
        # self.base_unet = base_unet
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.bias = bias
        self.linear_mu = nn.Linear(input_dim, z_dim, bias=bias)
        self.linear_sig = nn.Linear(input_dim, z_dim, bias=bias)

    def forward(self, x, t):
        # x: [B, 3, 256, 256]
        x = torch.flatten(x, start_dim=1)
        mu = self.linear_mu(x)
        sigma = F.softplus(self.linear_sig(x))
        z = mu + sigma * torch.randn_like(mu)
        return z, mu, sigma


class MLPEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.input_dim = args.image_dim**2 * args.num_channels
        self.hidden_dim = args.hidden_dim
        self.latent_dim = args.latent_dim
        self.bias = args.bias_rec
        self.num_layers = args.num_layers_rec

        if args.activation_rec == "relu":
            self.nonlin = nn.ReLU()
        elif args.activation_rec == "gelu":
            self.nonlin = nn.GELU()
        else:
            raise ValueError('nonlin must be specified, e.g. "relu" or "gelu"')
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Flatten())
        dims = [self.input_dim] + [self.hidden_dim] * self.num_layers  # [input_dim, hidden_dim, hidden_dim, ...]
        for i in range(self.num_layers):
            self.encoder.append(nn.Linear(dims[i], dims[i + 1], bias=self.bias))
            self.encoder.append(self.nonlin)

        self.linear_mu = nn.Linear(dims[-1], self.latent_dim, bias=self.bias)
        self.linear_logvar = nn.Linear(dims[-1], self.latent_dim, bias=self.bias)

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        mu = self.linear_mu(x)
        logvar = self.linear_logvar(x)
        return mu, logvar

    def sample(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std


# --------------------------- convolutional encoder -------------------------- #
class ConvNet(nn.Module):
    def __init__(self, args):
        super(ConvNet, self).__init__()

        self.latent_dim: int = args.latent_dim
        self.num_layers: int = args.num_layers_rec
        kernel_size = args.kernel_size_rec
        stride = args.stride_rec
        padding = args.padding_rec
        nonlin = args.activation_rec
        bias = args.bias_rec

        assert self.num_layers > 0, "num_layers must be greater than 0"
        self.kernels: list[int] = []
        for i in range(self.num_layers):
            self.kernels.append(args.num_kernels_rec)

        if nonlin == "relu":
            self.nonlin = nn.ReLU()
        elif nonlin == "gelu":
            self.nonlin = nn.GELU()
        else:
            raise ValueError('nonlin must be specified, e.g. "relu" or "gelu"')

        """first convolutional layer"""
        self.d1_input_channels = args.num_channels
        self.d1_input_dims = args.image_dim
        # input: 1 x 32 x 32
        # output: 4 x 16 x 16
        self.conv_d1 = nn.Conv2d(
            self.d1_input_channels, self.kernels[0], kernel_size, stride, padding, bias=bias
        )  # in_channels, out_channels, kernel_size, stride, padding

        if self.num_layers > 1:
            for i in range(1, self.num_layers):
                setattr(
                    self,
                    f"conv_d{i + 1}",
                    nn.Conv2d(self.kernels[i - 1], self.kernels[i], kernel_size, stride, padding, bias=bias),
                )
                setattr(self, f"bn_d{i + 1}", BF_BatchNorm(self.kernels[i]))
                setattr(
                    self,
                    f"d{i + 1}_input_dims",
                    self.compute_output_dims(
                        getattr(self, f"d{i}_input_dims"), self.kernels[i - 1], kernel_size, stride, padding
                    ),
                )
                # if i%2 == 0:
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)

        """linear layers"""
        linear_input_dims = self.compute_output_dims(
            getattr(self, f"d{self.num_layers}_input_dims"), self.kernels[-1], kernel_size, stride, padding
        )
        linear_input_dims = linear_input_dims**2 * self.kernels[-1]
        self.linear_mu = nn.Linear(linear_input_dims, self.latent_dim, bias=bias)
        self.linear_logvar = nn.Linear(linear_input_dims, self.latent_dim, bias=bias)

    def compute_output_dims(self, input_dim, output_channels, kernel_size, stride, padding):
        """calculate the output dimensions of a convolutional layer, for a given input dimension and convolutional settings"""
        return (input_dim - kernel_size + 2 * padding) // stride + 1

    def forward(self, x):
        for i in range(1, self.num_layers + 1):
            x = getattr(self, f"conv_d{i}")(x)
            if i != 1:
                x = getattr(self, f"bn_d{i}")(x)
                x = self.pool(x)
            x = self.nonlin(x)

        x = torch.flatten(x, start_dim=1)
        mu = self.linear_mu(x)
        logvar = self.linear_logvar(x)

        return mu, logvar

    def sample(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std


# --------------------------- convolutional encoder -------------------------- #
class ConvNetSimple(nn.Module):
    def __init__(self, args):
        super(ConvNetSimple, self).__init__()

        self.latent_dim: int = args.latent_dim
        self.num_layers: int = args.num_layers_rec
        kernel_size = args.kernel_size_rec
        stride = args.stride_rec
        padding = args.padding_rec
        nonlin = args.activation_rec
        bias = args.bias_rec

        assert self.num_layers > 0, "num_layers must be greater than 0"
        self.num_channels: list[int] = []
        for i in range(self.num_layers):
            self.num_channels.append(args.num_kernels_rec)

        if nonlin == "relu":
            self.nonlin = nn.ReLU()
        elif nonlin == "gelu":
            self.nonlin = nn.GELU()
        else:
            raise ValueError('nonlin must be specified, e.g. "relu" or "gelu"')

        """convolutional layers"""
        self.input_channels = args.num_channels
        self.input_dims = args.image_dim

        # add two of the following: conv then batch norm then nonlinearity
        self.convs = nn.ModuleList()
        current_input_channels = self.input_channels
        for i in range(self.num_layers):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(current_input_channels, self.num_channels[i], kernel_size, stride, padding, bias=bias),
                    BF_BatchNorm(self.num_channels[i]),
                    self.nonlin,
                )
            )
            current_input_channels = self.num_channels[i]

        # self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        """compute the output dimensions"""
        self.output_dims = self.input_dims
        for i in range(self.num_layers):
            self.output_dims = self.compute_output_dims(
                self.output_dims, self.num_channels[i], kernel_size, stride, padding
            )

        """linear layers"""
        linear_input_dims = self.output_dims**2 * self.num_channels[-1]
        self.linear_mu = nn.Linear(linear_input_dims, self.latent_dim, bias=bias)
        self.linear_logvar = nn.Linear(linear_input_dims, self.latent_dim, bias=bias)

        # Register hook to ensure gradients are contiguous
        for param in self.parameters():
            param.register_hook(lambda grad: grad.contiguous())

    def compute_output_dims(self, input_dim, output_channels, kernel_size, stride, padding):
        """calculate the output dimensions of a convolutional layer, for a given input dimension and convolutional settings"""
        return (input_dim - kernel_size + 2 * padding) // stride + 1

    def forward(self, x):
        for i in range(1, self.num_layers + 1):
            x = self.convs[i - 1](x)
            # x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.linear_mu(x)
        logvar = self.linear_logvar(x)

        return mu, logvar

    def sample(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std


class SpatialTimeProjection(nn.Module):
    """project the time embedding to a single image-shaped channel, to be concatenated with x"""

    def __init__(self, args):
        super().__init__()
        self.time_channels = args.time_channels
        self.image_dim = args.image_dim
        self.proj = nn.Linear(self.time_channels, self.image_dim**2)  # (B, C_t) => (B, H*W)

    def forward(self, t):
        return self.proj(t).view(-1, 1, self.image_dim, self.image_dim)


class ConvNetComplex(nn.Module):
    """Convolutional Gaussian encoder, optionally conditioned on the diffusion timestep.

    `args.time_embedding_method_rec` selects the conditioning, mirroring `MLPEncoder`:
      - "none":      no time input at all (default; `forward(x)` as before).
      - "as_input":  project the embedding to an image-shaped map and concatenate it
                     with x as an extra input channel.
      - "per_layer": add a per-block, per-channel projection of the embedding to every
                     normalised pre-activation.
      - "film":      per-block, per-channel scale/shift of the normalised pre-activation,
                     h <- (1 + scale) * h + shift, with zero-initialised heads so it
                     starts as identity conditioning.

    Time conditioning matters here because the encoder is applied to *noisy* inputs
    to form q(z | x_t): the correct posterior width varies by orders of magnitude
    across noise levels, so a t-independent encoder cannot represent that family.
    When `time_conditioned` is True, callers must pass `t`.

    Note on "per_layer": BF_BatchNorm's gammas start very small (|g| <= 0.025), so an
    additive time bias starts out large relative to the features it is added to. "film"
    starts as the identity and is the safer default with bias-free norms.
    """

    def __init__(self, args):
        super(ConvNetComplex, self).__init__()

        self.latent_dim: int = args.latent_dim
        self.num_layers: int = args.num_layers_rec
        self.num_channels: list[int] = []
        kernel_size = args.kernel_size_rec
        stride = args.stride_rec
        padding = args.padding_rec
        nonlin = args.activation_rec
        bias = args.bias_rec
        norm = args.norm_rec
        num_groups = args.num_groups_rec
        aap_output_size = args.adaptive_avg_pool_output_size

        assert self.num_layers > 0, "num_layers must be greater than 0"

        self.time_method = getattr(args, "time_embedding_method_rec", "none")
        if self.time_method not in ("none", "as_input", "per_layer", "film"):
            raise ValueError(
                f"Unknown time_embedding_method_rec: {self.time_method}, "
                'expected "none", "as_input", "per_layer" or "film"'
            )
        self.time_conditioned = self.time_method != "none"

        # for i in range(self.num_layers):
        #     self.num_channels.append(args.num_kernels_rec)

        for i in range(self.num_layers):
            self.num_channels.append(min(args.num_kernels_rec * (2**i), 256))

        if nonlin == "relu":
            self.nonlin = nn.ReLU()
        elif nonlin == "gelu":
            self.nonlin = nn.GELU()
        else:
            raise ValueError('nonlin must be specified, e.g. "relu" or "gelu"')

        if norm == "bn":
            self.norm = lambda x: nn.BatchNorm2d(x) if bias else BF_BatchNorm(x)
        elif norm == "gn":
            self.norm = lambda x: nn.GroupNorm(num_groups, x) if bias else nn.GroupNorm(num_groups, x, affine=False)
        else:
            raise ValueError(f"Unsupported normalization type: {norm}")

        """time conditioning"""
        if self.time_conditioned:
            self.time_channels = args.time_channels
            self.time_emb = TimeEmbedding(args)
            if self.time_method == "as_input":
                self.time_projection = SpatialTimeProjection(args)
            else:
                self.time_mlp = nn.Sequential(
                    nn.Linear(self.time_channels, self.time_channels),
                    nn.SiLU(),
                )
                # one head per conv block, sized to that block's channel count
                self.time_heads = nn.ModuleList(
                    [
                        nn.Linear(self.time_channels, c if self.time_method == "per_layer" else 2 * c)
                        for c in self.num_channels
                    ]
                )
                if self.time_method == "film":
                    # start at scale=0, shift=0 so conditioning begins as the identity
                    for head in self.time_heads:
                        nn.init.zeros_(head.weight)
                        nn.init.zeros_(head.bias)

        """convolutional layers"""
        self.input_channels = args.num_channels
        self.input_dims = args.image_dim

        # add two of the following: conv then batch norm then nonlinearity
        # the (conv, norm, nonlin) layout is kept identical under every time method so
        # that state dicts stay compatible; the conditioned forward indexes into it
        self.convs = nn.ModuleList()
        current_input_channels = self.input_channels + (1 if self.time_method == "as_input" else 0)
        for i in range(self.num_layers):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(current_input_channels, self.num_channels[i], kernel_size, stride, padding, bias=bias),
                    self.norm(self.num_channels[i]),
                    self.nonlin,
                )
            )
            current_input_channels = self.num_channels[i]

        # self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((aap_output_size, aap_output_size))

        """compute the output dimensions"""
        # self.output_dims = self.input_dims
        # for i in range(self.num_layers):
        #     self.output_dims = self.compute_output_dims(
        #         self.output_dims, self.num_channels[i], kernel_size, stride, padding
        #     )
        # self.output_dims = self.output_dims // 2

        """linear layers"""
        # linear_input_dims = self.output_dims**2 * self.num_channels[-1]
        linear_input_dims = aap_output_size**2 * self.num_channels[-1]
        self.linear_mu = nn.Linear(linear_input_dims, self.latent_dim, bias=bias)
        self.linear_logvar = nn.Linear(linear_input_dims, self.latent_dim, bias=bias)

        """logvar initialisation"""
        # The guidance score carries a 1/var factor, so a posterior that starts vague
        # (logvar ~ 0) makes sigma_t * grad log q(z|x_t) a ~0.1% correction to the
        # denoiser's prediction, and the only gradient that can grow it is itself
        # proportional to its current size. Starting from a sharp, input-independent
        # posterior gives the guidance a usable scale from step 0.
        self.logvar_init_bias = getattr(args, "logvar_init_bias", 0.0)
        if self.logvar_init_bias != 0.0:
            if self.linear_logvar.bias is None:
                raise ValueError("logvar_init_bias requires bias_rec=True; linear_logvar has no bias to set")
            with torch.no_grad():
                self.linear_logvar.weight.zero_()
                self.linear_logvar.bias.fill_(self.logvar_init_bias)

        # Register hook to ensure gradients are contiguous
        for param in self.parameters():
            param.register_hook(lambda grad: grad.contiguous())

    def compute_output_dims(self, input_dim, output_channels, kernel_size, stride, padding):
        """calculate the output dimensions of a convolutional layer, for a given input dimension and convolutional settings"""
        return (input_dim - kernel_size + 2 * padding) // stride + 1

    def forward(self, x, t=None):
        if not self.time_conditioned:
            for block in self.convs:
                x = block(x)
            return self.head(x)

        if t is None:
            raise ValueError(
                "ConvNetComplex was built with time_embedding_method_rec="
                f'"{self.time_method}" and requires a timestep t'
            )
        emb = self.time_emb(t.float())

        if self.time_method == "as_input":
            x = torch.cat([x, self.time_projection(emb)], dim=1)
            for block in self.convs:
                x = block(x)
            return self.head(x)

        emb = self.time_mlp(emb)
        for block, time_head in zip(self.convs, self.time_heads):
            conv, norm, nonlin = block[0], block[1], block[2]
            x = norm(conv(x))
            if self.time_method == "per_layer":
                x = x + time_head(emb)[:, :, None, None]
            else:  # film
                scale, shift = time_head(emb).chunk(2, dim=1)
                x = (1 + scale[:, :, None, None]) * x + shift[:, :, None, None]
            x = nonlin(x)
        return self.head(x)

    def head(self, x):
        """pool, flatten and read out the Gaussian parameters"""
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.linear_mu(x), self.linear_logvar(x)

    def sample(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std


# ----------------------------- previous convnet ----------------------------- #











# --------------------------- residual convolutional encoder -------------------------- #
def _apply_time_mod(h, time_mod, time_method):
    """FiLM / additive modulation of a normalised pre-activation."""
    if time_mod is None:
        return h
    if time_method == "per_layer":
        return h + time_mod[:, :, None, None]
    if time_method == "film":
        scale, shift = time_mod.chunk(2, dim=1)
        return (1 + scale[:, :, None, None]) * h + shift[:, :, None, None]
    return h


class DownBlock(nn.Module):
    """Strided conv -> norm -> (time modulation) -> activation. Halves resolution."""

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, norm_fn, nonlin, bias, time_method):
        super().__init__()
        self.time_method = time_method
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
        self.norm = norm_fn(out_ch)
        self.nonlin = nonlin

    def forward(self, x, time_mod=None):
        h = self.norm(self.conv(x))
        h = _apply_time_mod(h, time_mod, self.time_method)
        return self.nonlin(h)


class ResBlock(nn.Module):
    """Stride-1 residual block: x <- x + alpha * f(x). Adds depth without downsampling.

    `alpha` is a learnable scalar. With `zero_init=True` it starts at 0, so the block is
    exactly the identity at initialisation and the network *begins* as its downsampling
    trunk, growing into the extra depth only as training finds a use for it.

    This matters for SAMI specifically. The guidance term is sigma_t * grad_x log q(z|x_t),
    whose magnitude scales with d mu / d x, so every layer that contributes at
    initialisation inflates it. At step 0 that score is random, and a larger random score
    is a larger corruption of the (pretrained) denoiser's prediction -- which gradient
    descent removes by driving the score toward zero, i.e. collapsing the latent. Zero
    init decouples depth from the initial score magnitude, so `logvar_init_bias` does not
    have to be retuned every time the encoder's depth changes.

    Measured at initialisation on dsprites (64x64, film conditioning, logvar_init_bias=-2),
    ||score|| / encoder gradient SNR:
        3 down, 0 res                  1.15 / 3.27
        5 down, 0 res                  4.03 / 1.92
        3 down + 2 res, standard init  3.61 / 2.38
        3 down + 2 res, zero init      1.15 / 3.62
    """

    def __init__(self, channels, kernel_size, padding, norm_fn, nonlin, bias, zero_init, time_method):
        super().__init__()
        self.time_method = time_method
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, 1, padding, bias=bias)
        self.norm1 = norm_fn(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, 1, padding, bias=bias)
        self.norm2 = norm_fn(channels)
        self.nonlin = nonlin
        self.alpha = nn.Parameter(torch.zeros(1) if zero_init else torch.ones(1))

    def forward(self, x, time_mod=None):
        h = self.norm1(self.conv1(x))
        h = _apply_time_mod(h, time_mod, self.time_method)
        h = self.nonlin(h)
        h = self.norm2(self.conv2(h))
        return x + self.alpha * h


class ResNet(nn.Module):
    """Residual convolutional Gaussian encoder for q(z | x_t).

    Layout: `num_layers_rec` strided `DownBlock`s, then `num_res_blocks_rec` stride-1
    `ResBlock`s at the final resolution, then pool -> (mu, logvar). Depth therefore comes
    from the residual stage rather than from further downsampling, which keeps the spatial
    map large and the guidance score small while still extending the receptive field.
    For 64x64 inputs, 3 down + 2 res reaches ~79px of receptive field (full image) on an
    8x8 map, versus 63px on a 2x2 map for 5 downsampling stages.

    `args.time_embedding_method_rec` selects the conditioning, matching `ConvNetComplex`:
      - "none":      no time input at all; `forward(x)`.
      - "as_input":  project the embedding to an image-shaped map and concatenate it with
                     x as an extra input channel.
      - "per_layer": add a per-block, per-channel projection of the embedding to every
                     normalised pre-activation, in both the trunk and the residual stage.
      - "film":      per-block, per-channel scale/shift of the normalised pre-activation,
                     with zero-initialised heads so conditioning starts as the identity.

    Every block gets its own time head, and the modulations are built as explicit
    per-block lists whose lengths are asserted against the block lists -- a `zip` over
    mismatched lists would silently skip blocks instead of failing.
    """

    def __init__(self, args):
        super().__init__()

        self.latent_dim: int = args.latent_dim
        self.num_layers: int = args.num_layers_rec
        self.num_res_blocks: int = getattr(args, "num_res_blocks_rec", 0)
        self.res_zero_init: bool = getattr(args, "res_zero_init", True)
        kernel_size = args.kernel_size_rec
        stride = args.stride_rec
        padding = args.padding_rec
        bias = args.bias_rec
        norm = args.norm_rec
        num_groups = args.num_groups_rec
        aap_output_size = args.adaptive_avg_pool_output_size

        assert self.num_layers > 0, "num_layers_rec must be greater than 0"
        assert self.num_res_blocks >= 0, "num_res_blocks_rec must be >= 0"

        self.time_method = getattr(args, "time_embedding_method_rec", "none")
        if self.time_method not in ("none", "as_input", "per_layer", "film"):
            raise ValueError(
                f"Unknown time_embedding_method_rec: {self.time_method}, "
                'expected "none", "as_input", "per_layer" or "film"'
            )
        self.time_conditioned = self.time_method != "none"

        self.num_channels: list[int] = [min(args.num_kernels_rec * (2**i), 256) for i in range(self.num_layers)]

        if args.activation_rec == "relu":
            self.nonlin = nn.ReLU()
        elif args.activation_rec == "gelu":
            self.nonlin = nn.GELU()
        else:
            raise ValueError('activation_rec must be specified, e.g. "relu" or "gelu"')

        if norm == "bn":
            norm_fn = lambda c: nn.BatchNorm2d(c) if bias else BF_BatchNorm(c)
        elif norm == "gn":
            norm_fn = lambda c: nn.GroupNorm(num_groups, c) if bias else nn.GroupNorm(num_groups, c, affine=False)
        else:
            raise ValueError(f"Unsupported normalization type: {norm}")

        """time conditioning"""
        modulated = self.time_method in ("per_layer", "film")
        if self.time_conditioned:
            self.time_channels = args.time_channels
            self.time_emb = TimeEmbedding(args)
            if self.time_method == "as_input":
                self.time_projection = SpatialTimeProjection(args)
            else:
                self.time_mlp = nn.Sequential(
                    nn.Linear(self.time_channels, self.time_channels),
                    nn.SiLU(),
                )
                width = (lambda c: c if self.time_method == "per_layer" else 2 * c)
                self.down_time_heads = nn.ModuleList(
                    [nn.Linear(self.time_channels, width(c)) for c in self.num_channels]
                )
                self.res_time_heads = nn.ModuleList(
                    [nn.Linear(self.time_channels, width(self.num_channels[-1])) for _ in range(self.num_res_blocks)]
                )
                if self.time_method == "film":
                    # start at scale=0, shift=0 so conditioning begins as the identity
                    for head in list(self.down_time_heads) + list(self.res_time_heads):
                        nn.init.zeros_(head.weight)
                        nn.init.zeros_(head.bias)

        """downsampling trunk"""
        self.input_channels = args.num_channels
        self.input_dims = args.image_dim
        block_time_method = self.time_method if modulated else "none"

        self.down_blocks = nn.ModuleList()
        current_input_channels = self.input_channels + (1 if self.time_method == "as_input" else 0)
        for i in range(self.num_layers):
            self.down_blocks.append(
                DownBlock(
                    current_input_channels, self.num_channels[i], kernel_size, stride, padding,
                    norm_fn, self.nonlin, bias, block_time_method,
                )
            )
            current_input_channels = self.num_channels[i]

        """residual stage (stride 1, final resolution)"""
        self.res_blocks = nn.ModuleList(
            [
                ResBlock(
                    self.num_channels[-1], kernel_size, padding, norm_fn, self.nonlin, bias,
                    self.res_zero_init, block_time_method,
                )
                for _ in range(self.num_res_blocks)
            ]
        )

        if modulated:
            assert len(self.down_time_heads) == len(self.down_blocks)
            assert len(self.res_time_heads) == len(self.res_blocks)

        self.pool = nn.AdaptiveAvgPool2d((aap_output_size, aap_output_size))

        """linear layers"""
        linear_input_dims = aap_output_size**2 * self.num_channels[-1]
        self.linear_mu = nn.Linear(linear_input_dims, self.latent_dim, bias=bias)
        self.linear_logvar = nn.Linear(linear_input_dims, self.latent_dim, bias=bias)

        """logvar initialisation"""
        # see ConvNetComplex: the guidance score carries a 1/var factor, so a posterior
        # that starts vague leaves the guidance too small for its own gradient to grow it
        self.logvar_init_bias = getattr(args, "logvar_init_bias", 0.0)
        if self.logvar_init_bias != 0.0:
            if self.linear_logvar.bias is None:
                raise ValueError("logvar_init_bias requires bias_rec=True; linear_logvar has no bias to set")
            with torch.no_grad():
                self.linear_logvar.weight.zero_()
                self.linear_logvar.bias.fill_(self.logvar_init_bias)

        # Register hook to ensure gradients are contiguous
        for param in self.parameters():
            param.register_hook(lambda grad: grad.contiguous())

    def forward(self, x, t=None):
        if self.time_conditioned and t is None:
            raise ValueError(
                f'ResNet was built with time_embedding_method_rec="{self.time_method}" and requires a timestep t'
            )

        down_mods = [None] * len(self.down_blocks)
        res_mods = [None] * len(self.res_blocks)

        if self.time_conditioned:
            emb = self.time_emb(t.float())
            if self.time_method == "as_input":
                x = torch.cat([x, self.time_projection(emb)], dim=1)
            else:
                emb = self.time_mlp(emb)
                down_mods = [head(emb) for head in self.down_time_heads]
                res_mods = [head(emb) for head in self.res_time_heads]

        for block, mod in zip(self.down_blocks, down_mods):
            x = block(x, mod)
        for block, mod in zip(self.res_blocks, res_mods):
            x = block(x, mod)
        return self.head(x)

    def head(self, x):
        """pool, flatten and read out the Gaussian parameters"""
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.linear_mu(x), self.linear_logvar(x)

    def sample(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std


# ------------------------------------ bf batchnorm ------------------------------------ #
# class BF_batchNorm(nn.Module):
#     def __init__(self, num_kernels):
#         super(BF_batchNorm, self).__init__()
#         self.register_buffer("running_sd", torch.ones(1, num_kernels, 1, 1))
#         g = (torch.randn((1, num_kernels, 1, 1)) * (2.0 / 9.0 / 64.0)).clamp_(-0.025, 0.025)
#         self.gammas = nn.Parameter(g, requires_grad=True)

#     def forward(self, x):
#         training_mode = self.training
#         sd_x = torch.sqrt(x.var(dim=(0, 2, 3), keepdim=True, unbiased=False) + 1e-05)
#         if training_mode:
#             x = x / sd_x.expand_as(x)
#             with torch.no_grad():
#                 self.running_sd.copy_((1 - 0.1) * self.running_sd.data + 0.1 * sd_x)

#             x = x * self.gammas.expand_as(x)

#         else:
#             x = x / self.running_sd.expand_as(x)
#             x = x * self.gammas.expand_as(x)

#         return x


class BF_BatchNorm(nn.Module):
    def __init__(self, num_kernels):
        super().__init__()
        # Helper constant for numerical stability
        self.eps: float = 1e-5
        self.momentum: float = 0.1

        # Initialize buffer and parameter
        self.register_buffer("running_sd", torch.ones(1, num_kernels, 1, 1))

        g = (torch.randn((1, num_kernels, 1, 1)) * (2.0 / 9.0 / 64.0)).clamp_(-0.025, 0.025)
        self.gammas = nn.Parameter(g)

    def forward(self, x):
        if self.training:
            # Calculate std (PyTorch handles the dimensions efficiently)
            # unbiased=False matches your original code
            var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            sd_x = (var + self.eps).sqrt()

            # Update buffer without tracking gradients
            # We use .detach() to ensure this never leaks into the graph
            with torch.no_grad():
                self.running_sd.lerp_(sd_x.to(self.running_sd.dtype), self.momentum)  # ty:ignore[call-non-callable]
                # lerp_ is "Linear Interpolation": (1-w)*start + w*end

            # Normalize
            x = x / sd_x
        else:
            # Use running stats
            x = x / self.running_sd

        # Apply Gamma (Broadcasting handles dimensions automatically)
        return x * self.gammas
