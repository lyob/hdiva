import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributions as dist

"""Ladder Variational Autoencoder (LVAE) with two latent layers. Differs from lvae2 in two ways: 
you can configure different number of channels for the encoder and decoder, and the decoder outputs a log variance
as well as the mean, so we get a proper variance weighted MSE (= negative log likelihood error).  
"""

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
    def __init__(self, z_in, c_in, c_out, input_dim, num_blocks=2, return_z_params=False, z_out=None, dec_skip_connections=False):
        super(TopDownBlock, self).__init__()
        self.return_z_params = return_z_params
        self.dec_skip_connections = dec_skip_connections
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

        if top_down_input is not None and self.dec_skip_connections:
            d += top_down_input

        d = self.conv_block(d)

        prior_params = self.compress_block(d) if self.return_z_params else None
        return d, prior_params

class FinalDecoderBlock(nn.Module):
    def __init__(self, z_in, c_in, c_out, input_dim, num_blocks=2, dec_skip_connections=False, use_nll_loss=True):
        super(FinalDecoderBlock, self).__init__()
        self.dec_skip_connections = dec_skip_connections
        self.expand_block = nn.ConvTranspose2d(z_in, c_in, kernel_size=input_dim, stride=1, padding=0)
        self.use_nll_loss = use_nll_loss
        if use_nll_loss:
            c_out = c_out * 2  # output both mean and log variance for NLL loss
        self.conv_block = ConvBlock(c_in, c_out, num_blocks=num_blocks, mode='top_down')

    def forward(self, z, top_down_input=None):
        z = z.unsqueeze(-1).unsqueeze(-1)  # reshape to (B, z_in, 1, 1)
        d = self.expand_block(z)
        if top_down_input is not None and self.dec_skip_connections:
            d += top_down_input
        d = self.conv_block(d)  # (B, c_out*2, 1, 1)
        if self.use_nll_loss:
            mean, logvar = d.chunk(2, dim=1)
            return mean, logvar
        else:
            return d, None


class MergeBlock(nn.Module):
    '''takes in parameters of the likelihood and prior, outputs parameters of merged Gaussian posterior'''
    def __init__(self):
        super(MergeBlock, self).__init__()

    def merge_gaussians(self, mu1, ls1, mu2, ls2):
        j1 = (-2 * ls1).exp()
        j2 = (-2 * ls2).exp()

        # new ls
        new_ls = -0.5 * torch.log(j1 + j2 + 1e-6)  # ensure numerical stability
        
        # combined mu
        new_mu = (mu1 * j1 + mu2 * j2) / (j1 + j2 + 1e-6)  # ensure numerical stability

        return new_mu, new_ls

    def forward(self, lh_params, p_params):
        lh_mu, lh_ls = lh_params.chunk(2, dim=1)
        p_mu, p_ls = p_params.chunk(2, dim=1)

        mu, ls = self.merge_gaussians(lh_mu, lh_ls, p_mu, p_ls)
        return torch.cat([mu, ls], dim=1)


class LadderVAE(nn.Module):
    def __init__(self, input_dim, z_dims:list[int], enc_channels:list[int], dec_channels:list[int], num_blocks=2, dec_skip_connections=False, use_nll_loss=True):
        super(LadderVAE, self).__init__()
        assert len(enc_channels) == len(z_dims) + 1, "enc_channels must be one more than z_dims"
        assert len(dec_channels) == len(z_dims) + 1, "dec_channels must be one more than z_dims"
        assert len(enc_channels) == len(dec_channels), "enc_channels and dec_channels must be the same length"
        
        self.use_nll_loss = use_nll_loss

        encoder_layers = []
        for i in range(len(z_dims)):
            encoder_layers.append(BottomUpBlock(enc_channels[i], enc_channels[i+1], z_dims[i], num_blocks=num_blocks))
        self.encoder = nn.ModuleList(encoder_layers)

        # the model divides the spatial dimensions by 2 every latent stage
        final_dim = int(input_dim / (2**(len(z_dims))))

        decoder_layers = []
        for i in range(len(z_dims)):
            input_dim = final_dim * (i+1)
            if i < len(z_dims)-1:
                decoder_layers.append(TopDownBlock(z_dims[-i-1], dec_channels[i], dec_channels[i+1], 
                                                input_dim=input_dim, 
                                                num_blocks=num_blocks, 
                                                return_z_params=(i<len(z_dims)-1), 
                                                z_out=z_dims[-i-2] if i<len(z_dims)-1 else None,
                                                dec_skip_connections=dec_skip_connections,
                                                ))
            else:
                decoder_layers.append(FinalDecoderBlock(z_dims[-i-1], dec_channels[i], dec_channels[i+1], 
                                                        input_dim=input_dim, num_blocks=num_blocks,
                                                        dec_skip_connections=dec_skip_connections,
                                                        use_nll_loss=use_nll_loss,
                                                        ))
        self.decoder = nn.ModuleList(decoder_layers)  # go in order of top to bottom

        self.merge_block = MergeBlock()

        # self.kl = 0
        self.kl_per_layer = []


    def reparameterization_trick(self, params):
        """
        Reparameterization trick for sampling from the latent space.

        Args:
            params: Tensor of shape (B, 2*z_dim), mean and logstd of the latent space.

        Returns:
            z: Tensor of shape (B, z_dim), sampled latent variables.
        """
        mu, ls = params.chunk(2, dim=1)
        return mu + ls.exp() * torch.randn_like(ls)

    def log_standard_gaussian(self, z):
        """
        Evaluates the log pdf of a standard normal distribution at x. (Univariate distribution)
        Args:
            :param x: point to evaluate. shape (B, z_dim)
        :return: log N(x|0,I), shape (B,)
        """
        log_pdf = -0.5 * (math.log(2 * math.pi) + z.pow(2))
        return log_pdf.sum(dim=-1)

    def log_gaussian(self, z, mu, lv):
        """
        Calculates the log probability density function (PDF) of a normal
        distribution parametrised by mu and logvar.
        Args:
            :param x: point to evaluate. shape (B, z_dim)
            :param mu: mean of distribution. shape (B, z_dim)
            :param logvar: log-variance of distribution. shape (B, z_dim)
        :return: log N(x|mu, exp(logvar)), shape (B,)
        """
        var = torch.exp(lv).clamp(min=1e-6)  # ensure numerical stability
        log_pdf = -0.5 * (math.log(2 * math.pi) + lv + (z - mu).pow(2) / var)
        return log_pdf.sum(dim=-1)

    def compute_kl_with_samples(self, z_sample, posterior_params, prior_params=None):
        """
        Computes KL divergence between q = N(mu1, diag(var1)) and p = N(mu2, diag(var2)).
        Args:
            z_sample: Tensor of shape (B, z_dim), sampled from posterior q.
            posterior_params: Tensor of shape (B, 2*z_dim), mean and logvar of q.
            prior_params: Tensor of shape (B, 2*z_dim), mean and logvar of p.
        Returns:
            kl: Scalar, KL divergence after reduction (sum over z_dim, mean over batch).
        """
        mu_q, lv_q = posterior_params.chunk(2, dim=1)
        qz = self.log_gaussian(z_sample, mu_q, lv_q)
        
        if prior_params == None:
            pz = self.log_standard_gaussian(z_sample)
        else:
            mu_p, lv_p = prior_params.chunk(2, dim=1)
            pz = self.log_gaussian(z_sample, mu_p, lv_p)
        kl_per_dim = qz - pz

        # sum over dimensions, mean over batch
        kl = kl_per_dim.sum(dim=1).mean()
        return kl
    
    def compute_kl(self, posterior_params, prior_params=None):
        """
        Computes KL divergence between q = N(mu1, diag(var1)) and p = N(mu2, diag(var2)).
        Args:
            posterior_params: mean and logvar of q. Tensor of shape (B, 2*z_dim).
            prior_params: mean and logvar of p. Tensor of shape (B, 2*z_dim).
        Returns:
            kl: Scalar, KL divergence after reduction (sum over z_dim, mean over batch).
        """
        mu_q, ls_q = posterior_params.chunk(2, dim=1)
        if prior_params is None:
            kl_per_dim = 0.5 * ( (2 * ls_q).exp() + mu_q.pow(2) ) - ls_q - 0.5
        else:
            mu_p, ls_p = prior_params.chunk(2, dim=1)
            # KL per dimension, expressed in terms of log standard deviations
            kl_per_dim = 0.5 * ( (2 * ls_q).exp() + (mu_q - mu_p).pow(2) ) * (-2 * ls_p).exp() + ls_p - ls_q - 0.5

        # sum over dimensions, mean over batch
        kl = kl_per_dim.sum(dim=1).mean()
        return kl

    def forward(self, x):
        d = x
        bu_lh_params = []
        i = 0
        for layer in self.encoder:
            d, lh_params = layer(d)
            bu_lh_params.append(lh_params)
            i += 1

        # calculate the KL divergence for each layer
        self.kl_per_layer = []

        # sample from top-level latent posterior
        top_posterior_params = bu_lh_params[-1]
        z_top = self.reparameterization_trick(top_posterior_params)

        for i, layer in enumerate(self.decoder):
            if i == 0:
                # KL against an isotropic Gaussian
                layer_kl = self.compute_kl(top_posterior_params, prior_params=None)
                self.kl_per_layer.append(layer_kl)

                # for the top block, take only the top-level latent sample
                d, td_prior_params = layer(z_top)
                posterior_params = self.merge_block(bu_lh_params[-2], td_prior_params)
                z_merged = self.reparameterization_trick(posterior_params)  # sample from posterior

            elif i > 0 and i < len(self.decoder) - 1:
                # kl between bottom-up likelihood and top-down prior
                layer_kl = self.compute_kl(posterior_params, td_prior_params)
                self.kl_per_layer.append(layer_kl)

                # for all top-down blocks in the middle, take the posterior sample from the layer above
                d, td_prior_params = layer(z_merged, d)
                posterior_params = self.merge_block(bu_lh_params[-2-i], td_prior_params)
                z_merged = self.reparameterization_trick(posterior_params)

            else:
                # kl computation
                layer_kl = self.compute_kl(posterior_params, td_prior_params)
                self.kl_per_layer.append(layer_kl)

                # for the bottom block, take the posterior sample from the layer above and return the final image
                mean, lv = layer(z_merged, d)

        self.kl_per_layer = self.kl_per_layer[::-1]  # reverse to match order of z1, z2, ...

        return mean, lv

    def criterion(self, clean_x, current_kl_weights):
        pred_mu, pred_ls = self.forward(clean_x)  # output the mean and the log standard deviation
        
        # Compute the squared error
        squared_error_loss = (pred_mu - clean_x).pow(2)  # (B, C, H, W)
        
        if self.use_nll_loss:
            # compute the negative log likelihood
            dim = clean_x.shape[1:].numel()  # C*H*W
            loss_nll_by_dim = dim * (.5 * squared_error_loss / (2*pred_ls).exp().clamp(min=1e-6) + pred_ls)  # (B, C, H, W)
            
            # mean over batches, sum over dimensions -- important for relative weighing against the KL (Rybkin et al., 2021)
            x_loss = loss_nll_by_dim.mean(dim=0).sum()
        else:
            x_loss = squared_error_loss.mean(dim=0).sum()  # mean over batches, sum over dimensions

        weighted_kl = []
        for loss, weight in zip(self.kl_per_layer, current_kl_weights):
            weighted_kl.append(loss * weight)  # numpy scalar is auto-converted
        weighted_kl = sum(weighted_kl)

        total_loss = x_loss + weighted_kl
        
        return total_loss, x_loss, weighted_kl
