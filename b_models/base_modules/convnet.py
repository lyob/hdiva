import torch

# import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F


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


class ConvNetComplex(nn.Module):
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

        # Register hook to ensure gradients are contiguous
        for param in self.parameters():
            param.register_hook(lambda grad: grad.contiguous())

    def compute_output_dims(self, input_dim, output_channels, kernel_size, stride, padding):
        """calculate the output dimensions of a convolutional layer, for a given input dimension and convolutional settings"""
        return (input_dim - kernel_size + 2 * padding) // stride + 1

    def forward(self, x):
        for i in range(1, self.num_layers + 1):
            x = self.convs[i - 1](x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.linear_mu(x)
        logvar = self.linear_logvar(x)

        return mu, logvar

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
