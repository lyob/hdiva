import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RecNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_dim: int = args.input_dim
        self.latent_dim_y: int = args.latent_dim_y
        self.latent_dim_z: int = args.latent_dim_z
        self.num_layers_x_to_y: int = args.num_layers_rec_x_to_y
        self.num_layers_y_to_z: int = args.num_layers_rec_y_to_z
        self.hidden_dim: int = args.hidden_dim_rec
        # self.norm = lambda x: nn.LayerNorm(self.hidden_dim)
        nonlin = args.activation_rec
        bias = args.bias_rec

        if nonlin == "relu":
            self.nonlin = nn.ReLU()
        elif nonlin == "gelu":
            self.nonlin = nn.GELU()
        elif nonlin == "silu":
            self.nonlin = nn.SiLU()
        else:
            raise ValueError('nonlin must be specified, e.g. "relu" or "gelu"')

        assert self.num_layers_x_to_y > 0, "num_layers_x_to_y must be greater than 0"
        assert self.num_layers_y_to_z > 0, "num_layers_y_to_z must be greater than 0"

        # deterministic backbone layers
        self.layers_x_to_y = nn.ModuleList()
        for i in range(self.num_layers_x_to_y):
            self.layers_x_to_y.append(
                nn.Sequential(
                    nn.Linear(self.input_dim if i == 0 else self.hidden_dim, self.hidden_dim, bias=bias),
                    # self.norm(self.hidden_dim),
                    self.nonlin,
                )
            )
        self.layers_y_to_z = nn.ModuleList()
        for i in range(self.num_layers_y_to_z):
            self.layers_y_to_z.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias),
                    # self.norm(self.hidden_dim),
                    self.nonlin,
                )
            )

        # output heads
        self.head_y = nn.Linear(self.hidden_dim, self.latent_dim_y*2, bias=bias)
        self.head_z = nn.Linear(self.hidden_dim, self.latent_dim_z*2, bias=bias)

        # input heads
        self.input_head_y = nn.Linear(self.latent_dim_y, self.hidden_dim, bias=bias)

    def x_to_y(self, x: Tensor) -> tuple[Tensor, Tensor]:
        '''Encode a sample of x to mean and logvar of y.'''
        for i in range(self.num_layers_x_to_y):
            x = self.layers_x_to_y[i](x)
        y = self.head_y(x)
        mean, logvar = y.chunk(2, dim=-1)
        return mean, logvar

    def y_to_z(self, y: Tensor) -> tuple[Tensor, Tensor]:
        '''Encode sample of y to mean and logvar of z.'''
        y = self.input_head_y(y)
        for i in range(self.num_layers_y_to_z):
            y = self.layers_y_to_z[i](y)
        z = self.head_z(y)
        mean, logvar = z.chunk(2, dim=-1)
        return mean, logvar

    def x_to_y_to_z(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        '''Encode a sample of x to mean and logvar of y, and then to mean and logvar of z.'''
        d = x
        for i in range(self.num_layers_x_to_y):
            d = self.layers_x_to_y[i](d)
        y = self.head_y(d)
        y_mean, y_logvar = y.chunk(2, dim=-1)
        
        for i in range(self.num_layers_y_to_z):
            d = self.layers_y_to_z[i](d)
        z = self.head_z(d)
        z_mean, z_logvar = z.chunk(2, dim=-1)
        return y_mean, y_logvar, z_mean, z_logvar

    def x_to_z(self, x: Tensor) -> tuple[Tensor, Tensor]:
        '''Encode a sample of x to mean and logvar of z. No sampling of y.'''
        d = x
        for i in range(self.num_layers_x_to_y):
            d = self.layers_x_to_y[i](d)
        for i in range(self.num_layers_y_to_z):
            d = self.layers_y_to_z[i](d)
        z = self.head_z(d)
        mean, logvar = z.chunk(2, dim=-1)
        return mean, logvar

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.x_to_z(x)

    def sample(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std
