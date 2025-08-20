import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributions as dist

'''This contains the implementation of a hierarchical VAE model in the style of ladder VAEs'''
def merge_gaussians(mu1, var1, mu2, var2):
    precision1 = 1 / (var1 + 1e-8)
    precision2 = 1 / (var2 + 1e-8)
    
    # combined mu
    new_mu = (mu1 * precision1 + mu2 * precision2) / (precision1 + precision2)

    # combined variance
    new_var = 1 / (precision1 + precision2)
    return new_mu, new_var


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


class DecoderConvBlock(nn.Module):
    def __init__(self,
                 latent_in_dim:int,
                 latent_out_dim:int,
                 output_dim:int,
                #  out_channels:int=1,
                 channels:list[int]=[8, 4, 2, 1],
                 bias:bool=True,
                 ):
        super(DecoderConvBlock, self).__init__()
        
        # self.channels = [out_channels, *channels]
        self.channels = [*channels]
        self.ksp = [4, 2, 1]
        self.activation_fn = nn.ReLU()

        # first project the latent variable into the input dimensions of the conv network
        self.projection_dim = output_dim
        self.compute_projection_dim()
        self.in_project = nn.Linear(latent_in_dim, self.channels[-1]*self.projection_dim**2, bias=bias)

        # deconvolutional network to generate the likelihood score
        self.deconv_network = nn.ModuleList()
        for idx in range(len(self.channels)-1, -1, -1):
            self.deconv_network.append(nn.ConvTranspose2d(self.channels[idx], self.channels[idx-1], *self.ksp, bias=bias))
            self.deconv_network.append(self.activation_fn)

        self.linear_mu = nn.Linear(output_dim**2 * self.channels[-1], latent_out_dim)
        self.linear_sig = nn.Linear(output_dim**2 * self.channels[-1], latent_out_dim)

    def compute_projection_dim(self):
        '''calculate the input dimensions of the conv network'''
        for i in range(1, len(self.channels)+1):
            self.projection_dim = self.compute_conv_output_dims(self.projection_dim, *self.ksp)

    def compute_conv_output_dims(self, input_dim, kernel, stride, padding):
        '''calculate the output dimensions of a convolutional layer, for a given input dimension and convolutional settings'''
        return ((input_dim - kernel + 2*padding)//stride + 1)
    
    def forward(self, z):
        z = self.in_project(z)
        z = z.view(-1, self.channels[-1], self.projection_dim, self.projection_dim)

        for i in range(len(self.deconv_network)):
            z = self.deconv_network[i](z)

        z = torch.flatten(z, start_dim=1)

        z2_mu = self.linear_mu(z)
        z2_sig = F.softplus(self.linear_sig(z)) + 1e-6  # ensure variance is positive
        z2 = z2_mu + z2_sig*torch.randn_like(z2_mu)
        return z2, z2_mu, z2_sig**2


class LadderVAE(nn.Module):
    def __init__(
            self, 
            num_channels:int,
            image_dim:int,
            latent_dims:list[int], 
            channels:list[int]=[4, 8, 16], 
            bias=True,
            ):
        self.image_dim = image_dim
        self.hidden_dims = hidden_dims
        self.z_dims = z_dims
        self.n_layers = len(z_dims)

        dims = [image_dim, *hidden_dims]
        encoder_layers = nn.ModuleList([EncoderMLPBlock(dims[i], dims[i+1], z_dims[i]) for i in range(self.n_layers)])
        decoder_layers = nn.ModuleList([DecoderMLPBlock(z_dims[i], dims[i+1], z_dims[i]) for i in range(self.n_layers-1)])[::-1]

        self.xhat = FinalDecoder(z_dims[0], dims[0], input_dim)




class VAE(nn.Module):
    def __init__(self, 
                 encoder: nn.Module,  # half UNet encoder
                 decoder: nn.Module,  # UNet decoder
                 kl_reduction: str = 'mean',
                 ):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.kl_reduction = kl_reduction
        self.kl = torch.tensor(0.0, requires_grad=True)
        # self.img_C, self.img_H, self.img_W = image_resolution
    
    
    def scale_to_minus_one_to_one(self, x):
        # according to the DDPMs paper, normalization seems to be crucial to train reverse process network
        return x * 2 - 1
    
    def reverse_scale_to_zero_to_one(self, x):
        return (x + 1) * 0.5

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
        self.kl = kl.sum() if self.kl_reduction == 'sum' else kl.mean()
        

    def forward(self, x):
        z, mu, var = self.encoder(x)
        xhat = self.decoder(z)
        self.compute_kl_vectorized(mu, var)
        return xhat, x, torch.ones_like(x)

    # --------------------------------- inference -------------------------------- #
    def conditional_sample(self, N, target_image):
        self.encoder.eval()
        self.decoder.eval()
        target_image = target_image.repeat(N, 1, 1, 1)
        z_sample, z_mean, z_sd = self.encoder(target_image)
        xhat = self.decoder(z_sample)
        
        return xhat, z_mean, z_sd
    
    def unconditional_sample(self, N):
        self.decoder.eval()
        
        z = torch.randn(N, self.encoder.latent_dims, device=self.device)
        z_mean = torch.zeros_like(z)
        z_sd = torch.ones_like(z)
        xhat = self.decoder(z)
        
        return xhat, z_mean, z_sd
    