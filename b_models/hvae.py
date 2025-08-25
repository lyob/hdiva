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
        print(x.shape)
        for i in range(len(self.conv_network)):
            x = self.conv_network[i](x)
            print(x.shape)
        
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
        print(d.shape)
        for i in range(len(self.conv_network)):
            d = self.conv_network[i](d)
            print(d.shape)

        mu = torch.flatten(self.mu_readout(d), start_dim=1)
        var = torch.flatten(self.var_readout(d), start_dim=1)
        print(mu.shape)
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
        print(x.shape)
        x = self.pre_conv(x)
        print(x.shape)
        x = self.block(x)
        print(x.shape)
        return x
    
class BottomUpBlock(nn.Module):
    '''ConvBlock + CompressBlock. Takes in (B, c_in, w_in, h_in), returns (B, c_out, w_out, h_out), (B, z_out), (B, z_out)'''
    def __init__(self, c_in, c_out, z_out, num_blocks=2):
        super(BottomUpBlock, self).__init__()
        self.conv_block = ConvBlock(c_in, c_out, num_blocks=num_blocks, mode='bottom_up')
        self.compress_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c_out, 2*z_out)
        )

    def forward(self, x):
        d = self.conv_block(x)
        z_params = self.compress_block(d)
        # mu, lv = z_params.chunk(2, dim=1)
        # return d, mu, lv
        return d, z_params


class TopDownBlock(nn.Module):
    '''ExpandBlock + ConvBlock. 
    Takes in (B, z_in) and optionally a top-down input (B, c_in, w_in, h_in). 
    Converts the latent (B, z_in) with an ExpandBlock into (B, c_in, w_in, h_in). 
    Returns (B, c_out, w_out, h_out).
    If `return_z_params` is True, uses CompressBlock to return latent params.
    '''
    def __init__(self, z_in, c_in, c_out, input_dim, num_blocks=2, return_z_params=False, z_out=None,):
        super(TopDownBlock, self).__init__()
        self.return_z_params = return_z_params
        # dynamic kernel_size in the expand block to match the final HxW of the image from the bottom-up blocks
        self.expand_block = nn.ConvTranspose2d(z_in, c_in, kernel_size=input_dim, stride=1, padding=0)
        self.conv_block = ConvBlock(c_in, c_out, num_blocks=num_blocks,mode='top_down')
        if self.return_z_params:
            assert z_out is not None and z_out > 0, "z_out must be specified and greater than 0 if return_z_params is True"
            self.compress_block = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c_out, 2 * z_out)
            )

    def forward(self, z, top_down_input=None):
        print('z input shape:', z.shape)
        z = z.unsqueeze(-1).unsqueeze(-1)  # reshape to (B, z_in, 1, 1)
        d = self.expand_block(z)
        print('post expand shape:', d.shape)

        if top_down_input is not None:
            d += top_down_input

        d = self.conv_block(d)
        print('post conv block', d.shape)

        prior_params = self.compress_block(d) if self.return_z_params else None
        return d, prior_params

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
    

class LadderVAE(nn.Module):
    # def __init__(self, input_dim, z_dims:list[int], c_in:list[int], c_out:list[int], num_blocks=2):
    def __init__(self, input_dim, z_dims:list[int], channels:list[int], num_blocks=2):
        super(LadderVAE, self).__init__()
        assert len(channels) == len(z_dims) + 1, "channels must be one more than z_dims"
        encoder_layers = []
        for i in range(len(z_dims)):
            encoder_layers.append(BottomUpBlock(channels[i], channels[i+1], z_dims[i], num_blocks=num_blocks))
        self.encoder = nn.ModuleList(encoder_layers)

        final_dim = int(input_dim / (2**len(z_dims)))

        decoder_layers = []
        for i in range(len(z_dims)):
            input_dim = final_dim * (len(z_dims)-i)
            print(input_dim)
            decoder_layers.append(TopDownBlock(z_dims[i], channels[i+1], channels[i], 
                                               input_dim=input_dim, 
                                               num_blocks=num_blocks, 
                                               return_z_params=(i>0), 
                                               z_out=z_dims[i-1] if i>0 else None)
                                               )
        self.decoder = nn.ModuleList(decoder_layers[::-1])  # go in order of top to bottom

        self.merge_block = MergeBlock()

    def reparameterization_trick(self, mu, lv):
        return mu + torch.exp(0.5 * lv) * torch.randn_like(lv)

    def forward(self, x):
        d = x
        bu_params = []
        i = 0
        for layer in self.encoder:
            print('i:', i)
            d, z_params = layer(d)
            bu_params.append(z_params)
            i += 1

        # sample from top-level latent
        z_top = self.reparameterization_trick(*bu_params[-1].chunk(2, dim=1))

        posterior_params = []
        for i, layer in enumerate(self.decoder):
            print('j:', i)
            # for the top block, take only the top-level latent
            if i == 0:
                d, td_params = layer(z_top)
                posterior_params.append(self.merge_block(bu_params[-i-2], td_params))

            # for all top-down blocks in the middle that are neither top nor bottom,
            # use the merged latent and top-down output of block above, and return 
            # top-down prior params to send to the layer below
            elif i > 0 and i < len(self.decoder) - 1:
                z_merged = self.reparameterization_trick(*posterior_params[-1].chunk(2, dim=1))
                d, td_params = layer(z_merged, d)
                posterior_params.append(self.merge_block(bu_params[-i-2], td_params))

            # for the bottom block, use the merged latent and top-down output of block above
            else:
                z_merged = self.reparameterization_trick(*posterior_params[-1].chunk(2, dim=1))
                d, _ = layer(z_merged, d)
        return d
    





class DecoderConvBlock(nn.Module):
    '''
    Top-down layer, including stochastic sampling, KL computation, and small
    deterministic ResNet with upsampling.

    The architecture when doing inference is roughly as follows:
        p_params = output of top-down layer above
        bu = inferred bottom-up value at this layer
        q_params = merge(bu, p_params)
        z = stochastic_layer(q_params)
        possibly get skip connection from previous top-down layer
        top-down deterministic ResNet

    When doing generation only, the value bu is not available, the
    merge layer is not used, and z is sampled directly from p_params.

    If this is the top layer, at inference time, the uppermost bottom-up value
    is used directly as q_params, and p_params are defined in this layer
    (while they are usually taken from the previous layer), and can be learned.
    '''
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
        print(z.shape)
        z = self.in_project(z)
        print(z.shape)
        z = z.view(-1, self.channels[-1], self.projection_dim, self.projection_dim)

        print(z.shape)
        for i in range(len(self.deconv_network)):
            z = self.deconv_network[i](z)
            print(z.shape)
        print('end of deconv network')

        z = torch.flatten(z, start_dim=1)

        z2_mu = self.linear_mu(z)
        z2_sig = F.softplus(self.linear_sig(z)) + 1e-6  # ensure variance is positive
        z2 = z2_mu + z2_sig*torch.randn_like(z2_mu)
        return z2, z2_mu, z2_sig**2


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
    