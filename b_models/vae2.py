import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributions as dist

'''This script contains the implementation of the single latent VAE model with convolutional encoders and deconvolutional decoders'''

'''Variable Convolutional Encoder'''
class SpatialConvEncoder(nn.Module):
    def __init__(
            self, 
            input_dim,
            z_dim, 
            c_in,
            channels=[8, 16],
            bias=True,
        ):
        super(SpatialConvEncoder, self).__init__()
        '''
        image_dim: int, the dimension of the input image height or width'''
        
        self.channels = [c_in, *channels]
        self.ksp = [3, 2, 1]

        '''first convolutional layer'''
        # input: 1 x 32 x 32
        # output: 4 x 16 x 16

        modules = []
        for idx in range(len(self.channels)-1):
            modules.append(nn.BatchNorm2d(self.channels[idx]))
            modules.append(nn.Conv2d(self.channels[idx], self.channels[idx+1], *self.ksp, bias=bias))
            modules.append(nn.ReLU())

        self.conv_network = nn.Sequential(*modules)
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
        print(d.shape)
        for i in range(len(self.conv_network)):
            d = self.conv_network[i](d)
            print(d.shape)

        mu = torch.flatten(self.mu_readout(d), start_dim=1)
        var = torch.flatten(self.var_readout(d), start_dim=1)
        return mu, var
    

'''Variable Convolutional Decoder'''
class SpatialConvDecoder(nn.Module):
    def __init__(
            self, 
            output_dim,
            z_dim, 
            c_out = 1,
            channels=[16, 8],
            bias=True,
        ):
        super(SpatialConvDecoder, self).__init__()
        '''
        image_dim: int, the dimension of the input image height or width'''
        
        self.channels = [*channels]
        self.ksp = [3, 2, 1]

        '''first convolutional layer'''
        # input: 1 x 32 x 32
        # output: 4 x 16 x 16

        final_dim = int(output_dim / (2**(len(channels))))
        self.expand_block = nn.ConvTranspose2d(z_dim, self.channels[-1], kernel_size=final_dim, stride=1, padding=0)


        modules = []
        for idx in range(len(self.channels)-1):
            modules.append(nn.ConvTranspose2d(self.channels[idx], self.channels[idx+1], *self.ksp, output_padding=1, bias=bias))
            modules.append(nn.BatchNorm2d(self.channels[idx]))
            modules.append(nn.ReLU())
            modules.append(nn.Conv2d(self.channels[idx+1], self.channels[idx+1], kernel_size=3, stride=1, padding=1))
        self.deconv_network = nn.Sequential(*modules)

        self.final_deconv = nn.ConvTranspose2d(self.channels[-1], self.channels[-1], *self.ksp, output_padding=1, bias=bias)
        self.output_readout = nn.Conv2d(self.channels[-1], c_out, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        d = z.view(z.size(0), z.size(1), 1, 1)  # reshape to (B, latent_dim, 1, 1)
        print(d.shape)
        d = self.expand_block(d)
        print(d.shape)

        for i in range(len(self.deconv_network)):
            d = self.deconv_network[i](d)
            print(d.shape)

        d = self.final_deconv(d)
        xhat = self.output_readout(d)
        # xhat = self.output_activation(self.output_readout(d))
        return xhat
        

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
        mu, var = self.encoder(x)
        z = mu + torch.sqrt(var) * torch.randn_like(var)
        print(z.shape)
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
    

# ------------------------------------ bf batchnorm ------------------------------------ #
class BF_batchNorm(nn.Module):
    def __init__(self, num_kernels):
        super(BF_batchNorm, self).__init__()
        self.register_buffer("running_sd", torch.ones(1,num_kernels,1,1))
        g = (torch.randn( (1,num_kernels,1,1) )*(2./9./64.)).clamp_(-0.025,0.025)
        self.gammas = nn.Parameter(g, requires_grad=True)

    def forward(self, x):
        training_mode = self.training       
        sd_x = torch.sqrt(x.var(dim=(0,2,3) ,keepdim = True, unbiased=False)+ 1e-05)
        if training_mode:
            x = x / sd_x.expand_as(x)
            with torch.no_grad():
                self.running_sd.copy_((1-.1) * self.running_sd.data + .1 * sd_x)

            x = x * self.gammas.expand_as(x)

        else:
            x = x / self.running_sd.expand_as(x)
            x = x * self.gammas.expand_as(x)

        return x


