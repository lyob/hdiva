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
    def __init__(self, c_in, c_out, z_out, num_blocks=2):
        super(BottomUpBlock, self).__init__()
        self.conv_block = ConvBlock(c_in, c_out, num_blocks=num_blocks, mode='bottom_up')
        self.compress_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c_out, 2*z_out, bias=False)
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
                nn.Linear(c_out, 2 * z_out, bias=False)
            )

    def forward(self, z, top_down_input=None):
        z = z.unsqueeze(-1).unsqueeze(-1)  # reshape to (B, z_in, 1, 1)
        d = self.expand_block(z)

        if top_down_input is not None:
            d += top_down_input

        d = self.conv_block(d)

        prior_params = self.compress_block(d) if self.return_z_params else None
        return d, prior_params

class MergeBlock(nn.Module):
    '''takes in parameters of the likelihood and prior, outputs parameters of merged Gaussian posterior'''
    def __init__(self):
        super(MergeBlock, self).__init__()

    def merge_gaussians(self, mu1, var1, mu2, var2):
        var1 = torch.clamp(var1, min=1e-6)
        var2 = torch.clamp(var2, min=1e-6)
        precision1 = 1 / var1
        precision2 = 1 / var2
        
        # combined mu
        new_mu = (mu1 * precision1 + mu2 * precision2) / (precision1 + precision2)

        # combined variance
        new_var = 1 / (precision1 + precision2)
        new_var = torch.clamp(new_var, min=1e-6)  # ensure numerical stability
        return new_mu, new_var

    def forward(self, lh_params, p_params):
        lh_mu, lh_lv = lh_params.chunk(2, dim=1)
        p_mu, p_lv = p_params.chunk(2, dim=1)

        lh_var = torch.exp(lh_lv)
        p_var = torch.exp(p_lv)

        mu, var = self.merge_gaussians(lh_mu, lh_var, p_mu, p_var)
        lv = torch.log(var)
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

        # the model divides the spatial dimensions by 2 every latent stage
        final_dim = int(input_dim / (2**(len(z_dims))))

        decoder_layers = []
        for i in range(len(z_dims)):
            input_dim = final_dim * (len(z_dims)-i)
            decoder_layers.append(TopDownBlock(z_dims[i], channels[i+1], channels[i], 
                                               input_dim=input_dim, 
                                               num_blocks=num_blocks, 
                                               return_z_params=(i>0), 
                                               z_out=z_dims[i-1] if i>0 else None)
                                               )
        self.decoder = nn.ModuleList(decoder_layers[::-1])  # go in order of top to bottom

        self.merge_block = MergeBlock()

        # self.kl = 0
        self.kl_per_layer = []


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

    def compute_kl(self, posterior_params, prior_params):
        """
        Computes KL divergence between q = N(mu1, diag(var1)) and p = N(mu2, diag(var2)).
        Args:
            posterior_params: Tensor of shape (B, 2*z_dim), mean and logvar of q.
            prior_params: Tensor of shape (B, 2*z_dim), mean and logvar of p.
        Returns:
            kl: Scalar, KL divergence after reduction (sum over z_dim, mean over batch).
        """
        mu_q, lv_q = posterior_params.chunk(2, dim=1)
        mu_p, lv_p = prior_params.chunk(2, dim=1)

        var_q = torch.exp(lv_q).clamp(min=1e-6)
        var_p = torch.exp(lv_p).clamp(min=1e-6)

        # KL per dimension
        kl_per_dim = 0.5 * (
            lv_p - lv_q                                     # log(sigma_p^2 / sigma_q^2)
            + (var_q + (mu_q - mu_p).pow(2)) / var_p        # variance + squared diff
            - 1.0
        )

        # sum over dimensions, mean over batch
        kl = kl_per_dim.sum(dim=1).mean()
        return kl

    def forward(self, x):
        d = x
        bu_params = []
        i = 0
        for layer in self.encoder:
            d, z_params = layer(d)
            bu_params.append(z_params)
            i += 1

        # calculate the KL divergence for each layer
        # self.kl = 0
        self.kl_per_layer = []

        zeros = torch.zeros((x.size(0), bu_params[-1].size(1)//2), device=x.device)
        ones = torch.ones((x.size(0), bu_params[-1].size(1)//2), device=x.device)
        iso = torch.cat([zeros, ones], dim=1)

        # KL against an isotropic Gaussian
        layer_kl = self.compute_kl(bu_params[-1], iso)
        
        # self.kl += layer_kl
        self.kl_per_layer.append(layer_kl)

        # sample from top-level latent
        z_top = self.reparameterization_trick(bu_params[-1])

        for i, layer in enumerate(self.decoder):
            if i == 0:
                # for the top block, take only the top-level latent sample
                # and return the posterior sample
                d, td_params = layer(z_top)
                
                posterior_params = self.merge_block(bu_params[-2], td_params)
                
                # kl between posterior and top-down prior
                layer_kl = self.compute_kl(posterior_params, td_params)
                self.kl_per_layer.append(layer_kl)

                z_merged = self.reparameterization_trick(posterior_params)

            elif i > 0 and i < len(self.decoder) - 1:
                # for all top-down blocks in the middle, take the posterior sample from the
                # layer above, and then return this layer's posterior sample
                d, td_params = layer(z_merged, d)
                
                posterior_params = self.merge_block(bu_params[-2-i], td_params)
                
                layer_kl = self.compute_kl(posterior_params, td_params)
                self.kl_per_layer.append(layer_kl)

                z_merged = self.reparameterization_trick(posterior_params)

            else:
                # for the bottom block, take the posterior sample from the layer above
                # and return the final image
                d, _ = layer(z_merged, d)

        self.kl_per_layer = self.kl_per_layer[::-1]  # reverse to match order of z1, z2, ...

        return d

    def criterion(self, clean_x, img_dim):
        # kl_weights = kl_weights.to(clean_x.device)
        prediction = self.forward(clean_x)
        # Compute the loss
        mse_loss = F.mse_loss(prediction, clean_x, reduction="mean")
        weighted_mse_loss = mse_loss.mean() * img_dim ** 2 / 2
        # return weighted_mse_loss, torch.dot(self.kl_per_layer, kl_weights)
        return weighted_mse_loss, self.kl_per_layer
