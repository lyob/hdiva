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

    def merge_gaussians(self, mu1, lv1, mu2, lv2):
        precision1 = torch.exp(-lv1)
        precision2 = torch.exp(-lv2)

        # combined precision
        new_precision = precision1 + precision2

        # combined mu
        new_mu = (mu1 * precision1 + mu2 * precision2) / new_precision

        # combined variance
        new_var = (1. / new_precision).clamp(min=1e-6)  # ensure numerical stability
        new_lv = torch.log(new_var)
        return new_mu, new_lv

    def forward(self, lh_params, p_params):
        lh_mu, lh_lv = lh_params.chunk(2, dim=1)
        p_mu, p_lv = p_params.chunk(2, dim=1)

        mu, lv = self.merge_gaussians(lh_mu, lh_lv, p_mu, p_lv)
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

        # KL against an isotropic Gaussian
        zeros = torch.zeros_like(bu_params[-1])  # both the mean and logvar should be 0 for isotropic gaussians
        layer_kl = self.compute_kl(bu_params[-1], zeros)
        
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
                z_merged = self.reparameterization_trick(posterior_params)  # sample from posterior
                
                # kl between bottom-up likelihood and top-down prior
                layer_kl = self.compute_kl(posterior_params, td_params)
                self.kl_per_layer.append(layer_kl)


            elif i > 0 and i < len(self.decoder) - 1:
                # for all top-down blocks in the middle, take the posterior sample from the
                # layer above, and then return this layer's posterior sample
                d, td_params = layer(z_merged, d)
                
                posterior_params = self.merge_block(bu_params[-2-i], td_params)
                z_merged = self.reparameterization_trick(posterior_params)
                
                # kl computation
                layer_kl = self.compute_kl(posterior_params, td_params)
                self.kl_per_layer.append(layer_kl)

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
