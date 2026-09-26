from typing import Tuple

# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.celeba_denoisers import UpSample
# from utils.model import calc_kl_divergence

# --------------------------- convolutional encoder -------------------------- #
class VariableConvEncoder(nn.Module):
    def __init__(self, 
                 args):
        super(VariableConvEncoder, self).__init__()
        '''
        image_dims: int, the dimension of the input image height or width
        conv_cfg: list of lists, each list contains [out_channels, kernel_size, stride, padding]'''
        
        self.latent_dims:int = args.latent_dims
        conv_cfg = args.conv_cfg
        nonlin = args.activation_rec
        bias = args.bias_rec
        
        assert len(conv_cfg) > 0, 'conv_cfg must contain at least 2 convolutional layers'
        self.conv_cfg = conv_cfg
        
        if nonlin == 'relu':
            self.nonlin = nn.ReLU() 
        else:
            raise ValueError('nonlin must be specified, e.g. "relu"')
        
        '''first convolutional layer'''
        self.d1_input_channels = args.num_channels
        self.d1_input_dims = args.image_dims
        # input: 1 x 32 x 32
        # output: 4 x 16 x 16
        self.conv_d1 = nn.Conv2d(self.d1_input_channels, *conv_cfg[0])  # in_channels, out_channels, kernel_size, stride, padding
        
        if len(conv_cfg) > 1:
            for i in range(1, len(conv_cfg)):
                setattr(self, f'conv_d{i+1}', nn.Conv2d(conv_cfg[i-1][0], *conv_cfg[i]))
                setattr(self, f'bn_d{i+1}', BF_batchNorm(conv_cfg[i][0]))
                setattr(self, f'd{i+1}_input_dims', self.compute_output_dims(getattr(self, f'd{i}_input_dims'), *conv_cfg[i-1]))
                # if i%2 == 0:
                    # setattr(self, f'pool_d{i//2}', nn.AvgPool2d(kernel_size=2, stride=2, padding=1 ))
        
        '''linear layers'''
        linear_input_dims = self.compute_output_dims(getattr(self, f'd{len(conv_cfg)}_input_dims'), *conv_cfg[-1])
        linear_input_dims = linear_input_dims**2 * conv_cfg[-1][0]
        self.linear_mu = nn.Linear(linear_input_dims, self.latent_dims, bias=bias)
        self.linear_sig = nn.Linear(linear_input_dims, self.latent_dims, bias=bias)
        
        self.kl_reduction = args.kl_reduction
        self.kl = 0
        
    def compute_output_dims(self, input_dim, output_channels, kernel, stride, padding):
        '''calculate the output dimensions of a convolutional layer, for a given input dimension and convolutional settings'''
        return ((input_dim - kernel + 2*padding)//stride + 1)
        
    def forward(self, x):
        for i in range(1, len(self.conv_cfg)+1):
            x = getattr(self, f'conv_d{i}')(x)
            if i!=1:
                x = getattr(self, f'bn_d{i}')(x)
            x = self.nonlin(x)
        
        x = torch.flatten(x, start_dim=1)
        mu = self.linear_mu(x)
        sigma = F.softplus(self.linear_sig(x))
        z = mu + sigma*torch.randn_like(mu)
        
        # self.kl = calc_kl_divergence(mu, torch.zeros_like(mu), sigma**2, torch.ones_like(sigma)).mean()
        # kl = torch.distributions.kl.kl_divergence(torch.distributions.Normal(mu, sigma**2), torch.distributions.Normal(0, 1))
        # self.kl = kl.sum() if self.kl_reduction == 'sum' else kl.mean()

        return z, mu, sigma
    





# ------------------------ Half a UNet for an encoder ----------------------- #
class FirstBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv = nn.Conv2d(args.num_channels, args.num_kernels_rec, args.kernel_size_rec, padding=args.padding_rec, bias=args.bias_rec)

        if args.activation_rec == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif args.activation_rec == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

    def forward(self, x:torch.Tensor):
        x = self.conv(x)
        x = self.act(x)
        return x
        
class DownBlock(nn.Module):
    def __init__(self, args, b, l):
        super().__init__()
        self.in_channels = args.num_kernels_rec*(2**(b-1)) if l==0 else args.num_kernels_rec*(2**b)
        self.out_channels = args.num_kernels_rec*(2**b)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, args.kernel_size_rec, padding=args.padding_rec, bias=args.bias_rec)
        self.bn = BF_batchNorm(self.out_channels)
        
        if args.activation_rec == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif args.activation_rec == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation function: {args.activation_rec}")
        
    def forward(self, x:torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class MiddleBlock(nn.Module):
    def __init__(self, args, l):
        super().__init__()
        b = args.num_blocks_rec-1
        self.in_channels = args.num_kernels_rec*(2**b) if l==0 else args.num_kernels_rec*(2**(b+1))
        self.out_channels = args.num_kernels_rec*(2**(b+1)) if l!=args.num_mid_conv_rec-1 else args.num_kernels_rec
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, args.kernel_size_rec, padding=args.padding_rec, bias=args.bias_rec)
        self.bn = BF_batchNorm(self.out_channels)

        if args.activation_rec == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif args.activation_rec == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation function: {args.activation_rec}")

    def forward(self, x:torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class DownSample(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pool =  nn.AvgPool2d(kernel_size=args.pool_window_rec, stride=2, padding=int((args.pool_window_rec-1)/2) ) 
        
    def forward(self, x:torch.Tensor):
        x = self.pool(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, args, b, l):
        super().__init__()
        self.in_channels = args.num_kernels_rec*(2**(b+1)) if l==0 else args.num_kernels_rec*(2**b)
        self.out_channels = args.num_kernels_rec*(2**b)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, args.kernel_size_rec, padding=args.padding_rec, bias=args.bias_rec)
        self.bn = BF_batchNorm(self.out_channels)
        
        if args.activation_rec == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif args.activation_rec == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation function: {args.activation_rec}")
                
    def forward(self, x:torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

################################################# network class #################################################
class HalfUNet(nn.Module): 
    def __init__(self, args): 
        super(HalfUNet, self).__init__()
        # args: num_channels, num_kernels, kernel_size, padding, bias, num_enc_conv, num_dec_conv, pool_window, num_blocks, num_mid_conv, time_channels
        
        self.activation = args.activation_rec
        self.kernel_size = args.kernel_size_rec
        self.padding = args.padding_rec
        self.num_kernels = args.num_kernels_rec
        self.num_blocks = args.num_blocks_rec
        self.num_enc_conv = args.num_enc_conv_rec
        self.num_mid_conv = args.num_mid_conv_rec
        self.pool_window = args.pool_window_rec
        self.downsample_in_mid_block = args.downsample_in_mid_block_rec if hasattr(args, 'downsample_in_mid_block_rec') else False
        
        self.latent_dims:int = args.latent_dims
        self.use_full_cov = args.use_full_cov if hasattr(args, 'use_full_cov') else False
        self.epsilon = args.epsilon if hasattr(args, 'epsilon') else 1e-6
        
        ########## First Block ##########
        self.first = FirstBlock(args)
        
        ########## Down Blocks ##########
        self.down = nn.ModuleList([])
        for b in range(self.num_blocks):
            self.init_encoder_block(b, args)
            self.down.append(DownSample(args))

        ########## Mid-layers ##########
        self.mid = nn.ModuleList([])
        for l in range(self.num_mid_conv):
            if self.downsample_in_mid_block and l>0:
                self.mid.append(DownSample(args))
            self.mid.append(MiddleBlock(args, l))
            
        self.up = nn.ModuleList([])
        
        linear_input_dims = self.compute_output_dims(args)
        self.linear_mu = nn.Linear(linear_input_dims, self.latent_dims, bias=args.bias_rec)
        
        # diagonal covariance or full covariance
        if self.use_full_cov:
            self.rank = args.latent_rank if 'latent_rank' in args else 4
            self.linear_sig = nn.Linear(linear_input_dims, self.latent_dims*(self.rank + 1), bias=args.bias_rec)
        else:
            self.linear_sig = nn.Linear(linear_input_dims, self.latent_dims, bias=args.bias_rec)

    def compute_output_dims(self, args):
        '''calculate the output dim of a convolutional layer, for given input dims and convolutional settings'''
        def config_of(layer):
            if type(layer.kernel_size)==tuple:
                return layer.kernel_size[0], layer.stride[0], layer.padding[0]
            else:
                return layer.kernel_size, layer.stride, layer.padding
        
        def layerwise_output_dims(input_dim, kernel, stride, padding):
            return ((input_dim - kernel + 2*padding)//stride + 1)
        
        # first
        output_dims = layerwise_output_dims(args.image_dims, *config_of(self.first.conv))
        
        # down
        for b in self.down:
            if hasattr(b, 'pool'):
                output_dims = layerwise_output_dims(output_dims, *config_of(b.pool))
            elif hasattr(b, 'conv'):
                output_dims = layerwise_output_dims(output_dims, *config_of(b.conv))
            
        # mid
        for m in self.mid:
            if hasattr(m, 'conv'):
                output_dims = layerwise_output_dims(output_dims, *config_of(m.conv))
            elif hasattr(m, 'pool'):
                output_dims = layerwise_output_dims(output_dims, *config_of(m.pool))
                
        return output_dims**2 * self.num_kernels
    
    def init_encoder_block(self, b, args):
        if b==0:
            for l in range(1,self.num_enc_conv):
                self.down.append(DownBlock(args,b,l))
        else:
            for l in range(self.num_enc_conv):
                self.down.append(DownBlock(args,b,l))
                
    def init_decoder_block(self, b, args):
        if b==0:
            for l in range(args.num_dec_conv_rec-1):
                self.up.append(UpBlock(args,b,l))
        else:
            for l in range(args.num_dec_conv_rec):
                self.up.append(UpBlock(args,b,l))
    
    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ########## Encoder ##########
        x = self.first(x)
        for d in self.down:
            x = d(x)
        ########## Mid-layers ##########
        for m in self.mid:
            x = m(x)
        ########## Decoder ##########
        x = torch.flatten(x, start_dim=1)
        mu = self.linear_mu(x)
        
        if self.use_full_cov:
            # print('full covariance method')
            x = self.linear_sig(x)
            L = x[:, :self.rank * self.latent_dims].reshape(-1, self.latent_dims, self.rank)
            # L = L - torch.diag_embed(torch.diag(L)) + torch.diag_embed(F.softplus(torch.diag(L)))
            low_rank_cov = torch.bmm(L, L.transpose(1, 2))
            
            diagonal = F.softplus(x[:, -self.latent_dims:].reshape(-1, self.latent_dims)) + self.epsilon
            D = torch.diag_embed(diagonal)
            
            cov = low_rank_cov + D
            mvn = torch.distributions.MultivariateNormal(mu, cov)
            z = mvn.sample()
        else:
            sigma = F.softplus(self.linear_sig(x))
            z = mu + sigma*torch.randn_like(mu)
            # cov = torch.diag_embed(sigma**2)  # turn into matrix form
            var = sigma**2
        
        return z, mu, var



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
    

# ---------------------------------- BF_CNN ---------------------------------- #
class BF_CNN_RF(nn.Module):
    def __init__(self, args):
        super(BF_CNN_RF, self).__init__()        
        self.num_layers = args.num_layers_rec #21
        self.RF = args.RF_rec 
        self.num_channels = args.num_channels
        self.num_kernels = args.num_kernels_rec
        self.kernel_size = args.kernel_size_rec
        self.padding = args.padding_rec
        self.latent_dims:int = args.latent_dims
        
        if self.num_layers != 21:
            raise ValueError('number of layers must be 21 ')

        if self.RF not in [5,8,9,13,23,43]:
            raise ValueError('choose a receptive field in [5,8,9,13,23,43]')

        #this creates RF=9x9, because of the way interspersing 3x3 layers in my code work. Improve code later 
        if self.RF == 9:
            self.RF = 8

        self.conv_layers = nn.ModuleList([])
        self.BN_layers = nn.ModuleList([])
        self.conv_layers.append(nn.Conv2d(self.num_channels,self.num_kernels, self.kernel_size, padding=self.padding , bias=False))

        for l in range(1,self.num_layers-1):
            if l%((self.num_layers - 1)/ (((self.RF-1)/2)-1)) != 0: ### set some of kernel sizes to 1x1
                kernel_size = 1
                padding = 0
            else:
                kernel_size = self.kernel_size
                padding = self.padding
            self.conv_layers.append(nn.Conv2d(self.num_kernels ,self.num_kernels, kernel_size, padding=padding , bias=False))
            self.BN_layers.append(BF_batchNorm(self.num_kernels ))
        self.conv_layers.append(nn.Conv2d(self.num_kernels, self.num_channels, self.kernel_size, padding=self.padding , bias=False))

        # linear layers
        linear_input_dims = self.compute_output_dims(args)
        self.linear_mu = nn.Linear(linear_input_dims, args.latent_dims, bias=args.bias_rec)
        self.use_diagonal_cov = args.use_diagonal_cov if hasattr(args, 'use_diagonal_cov') else True
        if self.use_diagonal_cov:
            self.linear_sig = nn.Linear(linear_input_dims, args.latent_dims, bias=args.bias_rec)
        else: 
            self.rank = args.latent_rank
            self.linear_sig = nn.Linear(linear_input_dims, args.latent_dims*(self.rank*2 + 1), bias=args.bias_rec)


    def compute_output_dims(self, args):
        '''calculate the output dim of a convolutional layer, for given input dims and convolutional settings'''
        def config_of(layer):
            if type(layer.kernel_size)==tuple:
                return layer.kernel_size[0], layer.stride[0], layer.padding[0]
            else:
                return layer.kernel_size, layer.stride, layer.padding
        
        def layerwise_output_dims(input_dim, kernel, stride, padding):
            return ((input_dim - kernel + 2*padding)//stride + 1)
        
        # first
        output_dims = layerwise_output_dims(args.image_dims, *config_of(self.conv_layers[0]))
        
        # conv_layers
        for l in self.conv_layers[1:]:
            output_dims = layerwise_output_dims(output_dims, *config_of(l))
                
        return output_dims**2 * self.num_kernels
    

    def forward(self, x):
        # activations = []
        relu = nn.ReLU(inplace=True)
        x = self.conv_layers[0](x) #first layer linear

        for l in range(1,self.num_layers-1):
            x = self.conv_layers[l](x)
            x = self.BN_layers[l-1](x)
            x = relu(x)

        x = self.conv_layers[-1](x)
        
        x = torch.flatten(x, start_dim=1)
        mu = self.linear_mu(x)
        return x