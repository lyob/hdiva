import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributions as dist

# ---------------------------- 2 latent layer LVAE --------------------------- #
class ConvBlock(nn.Module):
    def __init__(
            self,
            c_in: int,
            c_out: int,
            num_blocks=2,
            mode='bottom_up'
    ):
        super(ConvBlock, self).__init__()
        '''
        takes in (B, c_in, h_in, w_in), returns (B, c_out, h_out, w_out)
        '''

        assert mode in ['bottom_up', 'top_down']

        modules = []
        if mode == 'bottom_up':
            self.pre_conv = nn.Conv2d(in_channels=c_in, 
                                      out_channels=c_out, 
                                      kernel_size=3, 
                                      padding=1, 
                                      stride=2)
        elif mode == 'top_down':
            self.pre_conv = nn.ConvTranspose2d(in_channels=c_in, 
                                               out_channels=c_out, 
                                               kernel_size=3, 
                                               padding=1, 
                                               stride=2, 
                                               output_padding=1)
        
        # self.conv = nn.Conv2d(c_out, c_out, kernel_size=3, stride=2, padding=1)
        for i in range(num_blocks):
            modules.append(nn.BatchNorm2d(c_out))
            modules.append(nn.ReLU())
            modules.append(nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1))  # avoid down/up-sampling

        self.block = nn.Sequential(*modules)

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.block(x)
        return x
    
class BottomUpBlock(nn.Module):
    '''ConvBlock + CompressBlock. Takes in (B, c_in, w_in, h_in), returns (B, c_out, w_out, h_out), (B, z_out), (B, z_out)'''
    def __init__(self, c_in, c_out, num_blocks=2):
        super(BottomUpBlock, self).__init__()
        self.conv_block = ConvBlock(c_in, c_out, num_blocks=num_blocks, mode='bottom_up')

    def forward(self, x):
        d = self.conv_block(x)
        return d


class TopDownBlock(nn.Module):
    '''ExpandBlock + ConvBlock. 
    Takes in (B, z_in) and optionally a top-down input (B, c_in, w_in, h_in). 
    Converts the latent (B, z_in) with an ExpandBlock into (B, c_in, w_in, h_in). 
    Returns (B, c_out, w_out, h_out).
    If `return_z_params` is True, uses CompressBlock to return latent params.
    '''
    def __init__(self, c_in, c_out, num_blocks=2):
        super(TopDownBlock, self).__init__()
        # dynamic kernel_size in the expand block to match the final HxW of the image from the bottom-up blocks
        self.conv_block = ConvBlock(c_in, c_out, num_blocks=num_blocks,mode='top_down')
        
    def forward(self, z):
        d = self.conv_block(z)
        return d

class VAE(nn.Module):
    # def __init__(self, input_dim, z_dims:list[int], c_in:list[int], c_out:list[int], num_blocks=2):
    def __init__(self, input_dim, z_dim:int, channels:list[int], num_blocks=2):
        super(VAE, self).__init__()
        encoder_layers = []
        for i in range(len(channels)-1):
            encoder_layers.append(BottomUpBlock(channels[i], channels[i+1], num_blocks=num_blocks))
        self.encoder = nn.ModuleList(encoder_layers)

        self.compress_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], 2*z_dim)
        )

        # the model divides the spatial dimensions by 2 every latent stage
        final_dim = int(input_dim / (2**(len(channels)-1)))
        self.expand_block = nn.ConvTranspose2d(z_dim, channels[-1], kernel_size=final_dim, stride=1, padding=0)

        decoder_layers = []
        for i in range(len(channels)-1):
            decoder_layers.append(TopDownBlock(channels[i+1], channels[i], num_blocks=num_blocks))
        self.decoder = nn.ModuleList(decoder_layers[::-1])  # go in order of top to bottom

        self.kl = 0

    def reparameterization_trick(self, params):
        """
        Reparameterization trick for sampling from the latent space.
        Args:
            params: Tensor of shape (B, 2*z_dim), mean and logvar of the latent space.
        Returns:
            z: Tensor of shape (B, z_dim), sampled latent variables.
        """
        mu, lv = params.chunk(2, dim=1)
        return mu + torch.exp(0.5 * lv) * torch.randn_like(lv)

    def compute_kl(self, posterior_params,):
        """
        Computes KL divergence between q = N(mu1, diag(var1)) and p = N(mu2, diag(var2)).
        Args:
            posterior_params: Tensor of shape (B, 2*z_dim), mean and logvar of q.
            prior_params: Tensor of shape (B, 2*z_dim), mean and logvar of p.
        Returns:
            kl: Scalar, KL divergence after reduction (sum over z_dim, mean over batch).
        """
        mu_q, lv_q = posterior_params.chunk(2, dim=1)
        var_q = torch.exp(lv_q).clamp(min=1e-6)

        # KL per dimension
        kl_per_dim = 0.5 * (
            - lv_q                               # log(sigma_p^2 / sigma_q^2)
            + (var_q + (mu_q).pow(2))            # variance + squared diff
            - 1.0
        )

        # sum over dimensions, mean over batch
        self.kl = kl_per_dim.sum(dim=1).mean()

    def forward(self, x):
        d = x
        for layer in self.encoder:
            d = layer(d)

        z_params = self.compress_block(d)

        # sample from top-level latent
        self.compute_kl(z_params)
        z_top = self.reparameterization_trick(z_params)

        z_top = z_top.unsqueeze(-1).unsqueeze(-1)
        d = self.expand_block(z_top)

        for layer in self.decoder:
            d = layer(d)
        return d
    
    def criterion(self, clean_x, img_dim):
        prediction = self.forward(clean_x)
        # Compute the loss
        mse_loss = F.mse_loss(prediction, clean_x, reduction="mean")
        weighted_mse_loss = mse_loss.mean() * img_dim ** 2 / 2
        return weighted_mse_loss, self.kl
