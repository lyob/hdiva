import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributions as dist

'''This script contains the implementation of the single latent VAE model with convolutional encoders and deconvolutional decoders'''

    
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
    




################################################# UNet stuff #################################################
class FirstBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv = nn.Conv2d(args.num_channels, args.num_kernels, args.kernel_size, padding=args.padding, bias=args.bias)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x:torch.Tensor):
        x = self.conv(x)
        x = self.act(x)
        return x
        
class DownBlock(nn.Module):
    def __init__(self, args, b, l):
        super().__init__()
        self.in_channels = args.num_kernels*(2**(b-1)) if l==0 else args.num_kernels*(2**b)
        self.out_channels = args.num_kernels*(2**b)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, args.kernel_size, padding=args.padding, bias=args.bias)
        self.bn = nn.BatchNorm2d(self.out_channels) if args.bias else BF_batchNorm(self.out_channels)
        self.act = nn.ReLU(inplace=True)
        # self.time_emb = nn.Linear(args.time_channels, self.out_channels)
        # self.time_emb = TimeProjection(args, self.out_channels)
        
    def forward(self, x:torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class MiddleBlock(nn.Module):
    def __init__(self, args, l):
        super().__init__()
        b = args.num_blocks-1
        self.in_channels = args.num_kernels*(2**b) if l==0 else args.num_kernels*(2**(b+1))
        self.out_channels = args.num_kernels*(2**(b+1))
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, args.kernel_size, padding=args.padding , bias=args.bias)
        self.bn = nn.BatchNorm2d(self.out_channels) if args.bias else BF_batchNorm(self.out_channels)
        self.act = nn.ReLU(inplace=True)
        # self.time_emb = nn.Linear(args.time_channels, self.out_channels)
        
    def forward(self, x:torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, args, b, l):
        super().__init__()
        self.in_channels = args.num_kernels*(2**(b+1)) if l==0 else args.num_kernels*(2**b)
        self.out_channels = args.num_kernels*(2**b)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, args.kernel_size, padding=args.padding, bias=args.bias)
        self.bn = nn.BatchNorm2d(self.out_channels) if args.bias else BF_batchNorm(self.out_channels)
        self.act = nn.ReLU(inplace=True)
        # self.time_emb = nn.Linear(args.time_channels, self.out_channels)
                
    def forward(self, x:torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    
class FinalBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv = nn.Conv2d(args.num_kernels, args.num_channels, kernel_size=args.kernel_size, padding=args.padding, bias=False)
        
    def forward(self, x:torch.Tensor):
        x = self.conv(x)
        return x
    
class DownSample(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pool =  nn.AvgPool2d(kernel_size=args.pool_window, stride=2, padding=int((args.pool_window-1)/2) ) 
        
    def forward(self, x:torch.Tensor):
        x = self.pool(x)
        return x
    
class UpSample(nn.Module):
    def __init__(self, args, b):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(args.num_kernels*(2**(b+1)), args.num_kernels*(2**b), kernel_size=2, stride=2, bias=False)
        
    def forward(self, x:torch.Tensor):
        x = self.upsample(x)
        return x



class UNetDecoder(nn.Module): 
    def __init__(self, args): 
        super(UNetDecoder, self).__init__()
        # args: num_channels, num_kernels, kernel_size, padding, bias, num_enc_conv, num_dec_conv, pool_window, num_blocks, num_mid_conv, time_channels
        
        self.pool_window = args.pool_window
        self.num_blocks = args.num_blocks
        self.num_enc_conv = args.num_enc_conv
        self.num_mid_conv = args.num_mid_conv
        self.num_dec_conv = args.num_dec_conv

        ########## Latent Embedding ##########
        self.num_channels = args.num_channels
        self.image_dims = args.image_dims
        self.latent_embedding = nn.Linear(args.latent_dims, args.num_channels*args.image_dims*args.image_dims)
        
        ########## First Block ##########
        self.first = FirstBlock(args)
        
        ########## Down Blocks ##########
        # self.down = nn.ModuleDict([])
        self.down = nn.ModuleList([])
        for b in range(self.num_blocks):
            self.init_encoder_block(b, args)
            self.down.append(DownSample(args))

        ########## Mid-layers ##########
        self.mid = nn.ModuleList([])
        for l in range(args.num_mid_conv):
            self.mid.append(MiddleBlock(args, l))
                                    
        ########## Up Blocks ##########
        self.up = nn.ModuleList([])
        for b in range(self.num_blocks-1,-1,-1):
            self.up.append(UpSample(args,b))
            self.init_decoder_block(b,args)
        
        ########## Final Block ##########
        self.final = FinalBlock(args)
    
    
    def init_encoder_block(self, b, args):
        if b==0:
            for l in range(1,args.num_enc_conv):
                self.down.append(DownBlock(args,b,l))
        else:
            for l in range(args.num_enc_conv):
                self.down.append(DownBlock(args,b,l))
    
    def init_decoder_block(self, b, args):
        if b==0:
            for l in range(args.num_dec_conv-1):
                self.up.append(UpBlock(args,b,l))
        else:
            for l in range(args.num_dec_conv):
                self.up.append(UpBlock(args,b,l))

    def forward(self, z:torch.Tensor):
        '''now we don't have a time embedding, we'll use these parameters to map the latent into the input space'''
        z = self.latent_embedding(z)  # maps from (B, latent_dims) to (B, C, H, W)
        z = z.view(-1, self.num_channels, self.image_dims, self.image_dims)  # reshape to (B, C, H, W)

        ########## Encoder ##########
        x = self.first(z)
        unpooled = []
        for d in self.down:
            if type(d) == DownSample:
                unpooled.append(x) 
            x = d(x)
            
        ########## Mid-layers ##########
        for m in self.mid:
            x = m(x)
        
        ########## Decoder ##########
        for u in self.up:
            x = u(x)
            if type(u) == UpSample:
                # this is where the residual connection comes in!!
                x = torch.cat([x, unpooled.pop()], dim = 1)
        
        x = self.final(x)

        return x


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


