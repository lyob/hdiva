import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributions as dist

'''This contains the implementation of a hierarchical VAE model in the style of ladder VAEs'''

class EncoderBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=True):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(output_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


class ConvEncoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(ConvEncoder, self).__init__()
        self.enc1 = EncoderBlock(input_channels, hidden_channels)
        self.enc2 = EncoderBlock(hidden_channels, output_channels)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        return x


'''Variable Convolutional Encoder'''
class VariableConvEncoder(nn.Module):
    def __init__(
            self, 
            num_channels,
            image_dims,
            latent_dims, 
            channels=[4, 8, 16], 
            bias=True,
        ):
        super(VariableConvEncoder, self).__init__()
        '''
        image_dims: int, the dimension of the input image height or width'''
        
        self.channels = channels
        self.kernel_size = 3
        self.stride = 2
        self.padding = 1

        '''first convolutional layer'''
        self.d1_input_channels = num_channels
        self.d1_input_dims = image_dims
        # input: 1 x 32 x 32
        # output: 4 x 16 x 16
        self.conv_d1 = nn.Conv2d(self.d1_input_channels, self.channels[0], self.kernel_size, self.stride, self.padding)

        if len(self.channels) > 1:
            for i in range(1, len(self.channels)):
                setattr(self, f'conv_d{i+1}', nn.Conv2d(self.channels[i-1], self.channels[i], self.kernel_size, self.stride, self.padding))
                setattr(self, f'd{i+1}_input_dims', self.compute_output_dims(getattr(self, f'd{i}_input_dims'), self.channels[i-1], self.kernel_size, self.stride, self.padding))

        '''linear layers'''
        linear_input_dims = self.compute_output_dims(getattr(self, f'd{len(self.channels)}_input_dims'), self.channels[-1], self.kernel_size, self.stride, self.padding)
        linear_input_dims = linear_input_dims**2 * self.channels[-1]
        self.linear_mu = nn.Linear(linear_input_dims, latent_dims, bias=bias)
        self.linear_sig = nn.Linear(linear_input_dims, latent_dims, bias=bias)
        
    def compute_output_dims(self, input_dim, output_channels, kernel, stride, padding):
        '''calculate the output dimensions of a convolutional layer, for a given input dimension and convolutional settings'''
        return ((input_dim - kernel + 2*padding)//stride + 1)
    
    def forward(self, x):
        for i in range(1, len(self.channels)+1):
            x = F.relu(getattr(self, f'conv_d{i}')(x))
        
        x = torch.flatten(x, start_dim=1)
        mu = self.linear_mu(x)
        sigma = F.softplus(self.linear_sig(x)) + 1e-6  # ensure sigma is positive
        z = mu + sigma*torch.randn_like(mu)
        var = sigma**2
        return z, mu, var


class VariableConvDecoder(nn.Module):
    def __init__(self,
                 num_channels:int,
                 img_dims:int,
                 latent_dims:int,
                 channels=[8, 4, 2],
                 output_channels:int=1,
                 bias:bool=True,
                 ):
        super(VariableConvDecoder, self).__init__()
        
        self.img_C = num_channels
        self.img_H = self.img_W = img_dims
        self.latent_dims = latent_dims
        self.channels = channels
        self.output_channels = output_channels
        self.kernel_size = 4
        self.stride = 2
        self.padding = 1
        self.activation_fn = nn.ReLU()

        # first project the latent variable into the input dimensions of the conv network
        self.projection_dim = self.img_H
        self.compute_projection_dim()
        self.in_project = nn.Linear(self.latent_dims, self.channels[-1]*self.projection_dim**2, bias=bias)

        # deconvolutional network to generate the likelihood score
        self.deconv_network = nn.ModuleList()
        self.deconv_network.append(nn.ConvTranspose2d(self.channels[-1], self.channels[-2], self.kernel_size, self.stride, self.padding, bias=bias))
        self.deconv_network.append(self.activation_fn)
        for idx in range(len(self.channels)-1, 1, -1):
            self.deconv_network.append(nn.ConvTranspose2d(self.channels[idx-1], self.channels[idx-2], self.kernel_size, self.stride, self.padding, bias=bias))
            self.deconv_network.append(self.activation_fn)
        self.deconv_network.append(nn.ConvTranspose2d(self.channels[0], output_channels, self.kernel_size, self.stride, self.padding, bias=bias))
    
    def compute_projection_dim(self):
        '''calculate the input dimensions of the conv network'''
        for i in range(1, len(self.channels)+1):
            self.projection_dim = self.compute_conv_output_dims(self.projection_dim, self.channels[i-1], self.kernel_size, self.stride, self.padding)

    def compute_conv_output_dims(self, input_dim, output_channels, kernel, stride, padding):
        '''calculate the output dimensions of a convolutional layer, for a given input dimension and convolutional settings'''
        return ((input_dim - kernel + 2*padding)//stride + 1)
    
    def compute_deconv_output_dims(self, input_dim, output_channels, kernel, stride, padding):
        '''calculate the output dimensions of a deconvolutional layer, for a given input dimension and convolutional settings'''
        return (input_dim - 1) * stride - 2*padding + kernel
    
    def forward(self, z):
        z = self.in_project(z)
        z = z.view(-1, self.channels[-1], self.projection_dim, self.projection_dim)

        for i in range(len(self.deconv_network)):
            z = self.deconv_network[i](z)
        return z
        

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
    