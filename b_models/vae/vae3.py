import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributions as dist

'''Variable Convolutional Encoder'''
class EncoderConvBlock(nn.Module):
    def __init__(
            self, 
            num_channels,
            img_dim,
            latent_dim, 
            channels=[4, 8, 16], 
            bias=True,
        ):
        super(EncoderConvBlock, self).__init__()
        '''
        image_dim: int, the dimension of the input image height or width'''
        
        self.channels = [num_channels, *channels]
        self.ksp = [3, 2, 1]

        '''first convolutional layer'''
        # input: 1 x 32 x 32
        # output: 4 x 16 x 16

        self.conv_network = nn.ModuleList()
        for idx in range(len(self.channels)-1):
            self.conv_network.append(nn.Conv2d(self.channels[idx], self.channels[idx+1], *self.ksp, bias=bias))
            self.conv_network.append(nn.ReLU())

        self.layer_input_dim = img_dim
        self.compute_projection_dim()
        linear_input_dim = self.layer_input_dim**2 * self.channels[-1]
        self.linear_mu = nn.Linear(linear_input_dim, latent_dim, bias=bias)
        self.linear_sig = nn.Linear(linear_input_dim, latent_dim, bias=bias)

    def compute_projection_dim(self):
        for i in range(len(self.channels)-1):
            self.layer_input_dim = self.compute_output_dims(self.layer_input_dim, *self.ksp)
        
    def compute_output_dims(self, input_dim, kernel, stride, padding):
        '''calculate the output dimensions of a convolutional layer, for a given input dimension and convolutional settings'''
        return ((input_dim - kernel + 2*padding)//stride + 1)
    
    def forward(self, x):
        for i in range(len(self.conv_network)):
            x = self.conv_network[i](x)
        
        x = torch.flatten(x, start_dim=1)
        mu = self.linear_mu(x)
        sigma = F.softplus(self.linear_sig(x)) + 1e-6  # ensure sigma is positive
        z = mu + sigma*torch.randn_like(mu)
        var = sigma**2
        return z, mu, var


'''Variable Convolutional Encoder'''
class SpatialEncoderConvBlock(nn.Module):
    def __init__(
            self, 
            input_dim,
            z_dim, 
            c_in,
            channels=[8, 16],
            bias=True,
        ):
        super(SpatialEncoderConvBlock, self).__init__()
        '''
        image_dim: int, the dimension of the input image height or width'''
        
        self.channels = [c_in, *channels]
        self.ksp = [3, 2, 1]

        '''first convolutional layer'''
        # input: 1 x 32 x 32
        # output: 4 x 16 x 16

        self.conv_network = nn.ModuleList()
        for idx in range(len(self.channels)-1):
            self.conv_network.append(nn.Conv2d(self.channels[idx], self.channels[idx+1], *self.ksp, bias=bias))
            self.conv_network.append(nn.ReLU())

        self.spatial_dim = input_dim
        self.compute_projection_dim()

        # now provide linear readouts that average over the spatial dimensions so we're only left with (B, latent_dim, 1, 1)
        self.mu_readout = nn.Conv2d(self.channels[-1], z_dim, self.spatial_dim, bias=bias)
        self.var_readout = nn.Conv2d(self.channels[-1], z_dim, self.spatial_dim, bias=bias)

    def compute_projection_dim(self):
        for i in range(len(self.channels)-1):
            self.spatial_dim = self.compute_output_dims(self.spatial_dim, *self.ksp)

    def compute_output_dims(self, input_dim, kernel, stride, padding):
        '''calculate the output dimensions of a convolutional layer, for a given input dimension and convolutional settings'''
        return ((input_dim - kernel + 2*padding)//stride + 1)

    def forward(self, x):
        d = x
        for i in range(len(self.conv_network)):
            d = self.conv_network[i](d)

        mu = torch.flatten(self.mu_readout(d), start_dim=1)
        var = torch.flatten(self.var_readout(d), start_dim=1)
        return d, mu, var




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

class MergeBlock(nn.Module):
    '''takes in parameters of the likelihood and prior, outputs parameters of merged Gaussian posterior'''
    def __init__(self):
        super(MergeBlock, self).__init__()

    def merge_gaussians(self, mu1, var1, mu2, var2):
        precision1 = 1 / (var1 + 1e-8)
        precision2 = 1 / (var2 + 1e-8)
        
        # combined mu
        new_mu = (mu1 * precision1 + mu2 * precision2) / (precision1 + precision2)

        # combined variance
        new_var = 1 / (precision1 + precision2)
        return new_mu, new_var

    def forward(self, lh_params, p_params):
        lh_mu, lh_lv = lh_params.chunk(2, dim=1)
        p_mu, p_lv = p_params.chunk(2, dim=1)

        lh_var = torch.exp(lh_lv)
        p_var = torch.exp(p_lv)

        mu, var = self.merge_gaussians(lh_mu, lh_var, p_mu, p_var)
        lv = torch.log(var + 1e-8)
        return torch.cat([mu, lv], dim=1)
    

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

        self.merge_block = MergeBlock()

        self.kl = 0

    def reparameterization_trick(self, mu, lv):
        return mu + torch.exp(0.5 * lv) * torch.randn_like(lv)
    

    def compute_kl_vectorized(self, mu, var):
        """
        Computes KL divergence between q = N(mu, diag(var)) and p = N(0, I).
        
        Args:
            mu: Tensor of shape (B, z_dim), mean of q.
            var: Tensor of shape (B, z_dim), diagonal elements of covariance (variances).
        
        Returns:
            kl: Scalar, KL divergence after reduction (sum or mean over z_dim).
        """
        var = torch.clamp(var, min=1e-6)
        
        # Compute KL terms: 0.5 * (var + mu^2 - 1 - log(var))
        kl_per_dim = 0.5 * (var + mu**2 - 1 - torch.log(var))  # Shape: (B, z_dim)
        
        # Reduce over z_dim
        kl = kl_per_dim.sum(dim=1)  # Shape: (B,)
        
        # Reduce over batch
        kl = kl.mean(dim=0)  # Scalar
        
        # Apply final reduction (sum or mean over z_dim)
        self.kl = kl.mean()
        # self.kl = kl.sum() if self.kl_reduction == 'sum' else kl.mean()

    def forward(self, x):
        d = x
        for layer in self.encoder:
            d = layer(d)

        z_params = self.compress_block(d)

        # sample from top-level latent
        mu, lv = z_params.chunk(2, dim=1)
        self.compute_kl_vectorized(mu, torch.exp(lv))
        
        z_top = self.reparameterization_trick(mu, lv)
        z_top = z_top.unsqueeze(-1).unsqueeze(-1)

        d = self.expand_block(z_top)

        for layer in self.decoder:
            d = layer(d)
        return d
    
    def criterion(self, clean_x):
        prediction = self.forward(clean_x)
        # Compute the loss
        mse_loss = F.mse_loss(prediction, clean_x, reduction="mean")
        weighted_mse_loss = mse_loss.mean()
        return weighted_mse_loss, self.kl
