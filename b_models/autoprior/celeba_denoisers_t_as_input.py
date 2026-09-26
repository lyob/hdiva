'''autoprior v2 denoisers for celeba dataset, with 80x80 images. 
Based on the UNet architecture.
'''
import math

import numpy as np
import torch
import torch.nn as nn


# ------------------------ conditional diffusion model ----------------------- #
class TimeEmbedding(nn.Module):
    '''sinusoidal position embedding'''
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class LinearTimeProjection(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.time_channels = args.time_channels
        self.proj = nn.Sequential(
            nn.Linear(self.time_channels, self.time_channels**2), # (B, 40) => (B, 40**2)
            # nn.ReLU(),
        )
        
    def forward(self, t):
        return self.proj(t).view(-1, 1, self.time_channels, self.time_channels)


class FirstBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv = nn.Conv2d(args.num_channels+1, args.num_kernels, args.kernel_size, padding=args.padding, bias=args.bias)
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
        self.time_emb = nn.Linear(args.time_channels, self.out_channels)
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
        self.time_emb = nn.Linear(args.time_channels, self.out_channels)
        
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
        self.time_emb = nn.Linear(args.time_channels, self.out_channels)
                
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

################################################# network class #################################################
class UNet(nn.Module): 
    def __init__(self, args): 
        super(UNet, self).__init__()
        # args: num_channels, num_kernels, kernel_size, padding, bias, num_enc_conv, num_dec_conv, pool_window, num_blocks, num_mid_conv, time_channels
        
        self.pool_window = args.pool_window
        self.num_blocks = args.num_blocks
        self.num_enc_conv = args.num_enc_conv
        self.num_mid_conv = args.num_mid_conv
        self.num_dec_conv = args.num_dec_conv
        self.time_channels = args.time_channels
        
        ########## Time Embedding ##########
        self.time_emb = TimeEmbedding(self.time_channels)
        self.time_projection = LinearTimeProjection(args)
        
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

    def forward(self, x:torch.Tensor, t:torch.Tensor):
        ########## Time embedding ###########
        t = self.time_emb(t)
        t = self.time_projection(t)
        x = torch.cat([x, t], dim=1)

        ########## Encoder ##########
        x = self.first(x)
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




# ------------------------------ unconditional diffusion model ------------------------------ #
class Diffusion(nn.Module):
    def __init__(
            self,
            denoiser,
            args,
            device:torch.device,
    ):
        super(Diffusion, self).__init__()
        self.denoiser = denoiser
        self.device = device
        self.n_times = args.timesteps

        self.define_noise_schedule(args)

        
    def scale_to_minus_one_to_one(self, x):
        # according to the DDPMs paper, normalization seems to be crucial to train reverse process network
        return x * 2 - 1
    
    def reverse_scale_to_zero_to_one(self, x):
        return (x + 1) * 0.5
    
    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
    def set_schedules_from_sigma(self):
        self.one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars ** 2
        self.alpha_bars = 1 - self.one_minus_alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        alphas = torch.ones_like(self.alpha_bars)
        for i in range(1, len(alphas)):
            alphas[i] = self.alpha_bars[i] / self.alpha_bars[i-1]
        alphas[0] = alphas[1] - (alphas[2] - alphas[1])  # linearly extrapolate
        self.alphas = alphas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.betas = 1 - self.alphas
        self.sqrt_betas = torch.sqrt(self.betas)

    def define_noise_schedule(self, args):
        '''define beta schedule depending on the distribution of the noise (as defined by the std))'''
        sigma_min, sigma_max = args.sigma_minmax if 'sigma_minmax' in args else [0.0001, .9999]
        
        timesteps = torch.linspace(0, 1, steps=self.n_times, device=self.device)
        
        # using the schedule outlined in the Nichol and Dhariwal (2021) paper 
        s = 0.01
        f_t = torch.cos((timesteps + s)/(1 + s) * torch.pi/2) ** 2
        f_0 = np.cos(s/(1 + s) * np.pi/2) ** 2
        alpha_bars = sigma_min + (sigma_max - sigma_min) * (f_t / f_0)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)
        self.set_schedules_from_sigma()
    
    def make_noisy(self, x_zeros, t):
        '''perturb x_0 into x_t (i.e., take x_0 samples into forward diffusion kernels)'''
        epsilon = torch.randn_like(x_zeros)
        
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, t, x_zeros.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, t, x_zeros.shape)
        
        noisy_sample = x_zeros * sqrt_alpha_bar + epsilon * sqrt_one_minus_alpha_bar
    
        return noisy_sample.detach(), epsilon

    def forward(self, x_zeros:torch.Tensor):
        '''forward pass of the model'''

        B, *_ = x_zeros.shape  # batch size
        
        # (1) randomly choose diffusion time-step
        t = torch.randint(low=0, high=self.n_times, size=(B,), device=self.device, dtype=torch.long)
        
        # (2) forward diffusion process: perturb x_zeros with fixed variance schedule
        x_zeros = x_zeros.detach().requires_grad_(True)
        noisy_x, epsilon = self.make_noisy(x_zeros, t)
        noisy_x = noisy_x.detach().requires_grad_(True)
    
        # (7) estimate the noise epsilon: predict epsilon (noise) given perturbed data at diffusion-timestep t.
        pred_epsilon = self.denoiser(noisy_x, t)  # = score of p(x_t|x_t+1)
        
        # (8) set the target and prediction for the model
        target = epsilon
        prediction = pred_epsilon
        
        mse_weights = torch.ones_like(x_zeros)
                
        return target, prediction, mse_weights





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





class BF_CNN_RF(nn.Module):
    def __init__(self, args):
        super(BF_CNN_RF, self).__init__()
        self.time_emb = TimeEmbedding(args.time_channels)
        
        if args.num_layers != 21:
            raise ValueError('number of layers must be 21 ')

        if args.RF not in [5,8,9,13,23,43]:
            raise ValueError('choose a receptive field in [5,8,9,13,23,43]')

        #this creates RF=9x9, because of the way interspersing 3x3 layers in my code work. Improve code later 
        if args.RF == 9:
            args.RF = 8

        self.num_layers = args.num_layers #21
        self.conv_layers = nn.ModuleList([])
        self.BN_layers = nn.ModuleList([])
        self.time_embeddings = nn.ModuleList([])

        self.conv_layers.append(nn.Conv2d(args.num_channels,args.num_kernels, args.kernel_size, padding=args.padding , bias=False))

        for l in range(1,self.num_layers-1):
            if l%((args.num_layers - 1)/ (((args.RF-1)/2)-1)) != 0: ### set some of kernel sizes to 1x1
                kernel_size = 1
                padding = 0
            else:
                kernel_size = args.kernel_size
                padding = args.padding
            self.conv_layers.append(nn.Conv2d(args.num_kernels ,args.num_kernels, kernel_size, padding=padding , bias=False))
            self.BN_layers.append(BF_batchNorm(args.num_kernels ))
            self.time_embeddings.append(nn.Linear(args.time_channels, args.num_kernels))

        self.conv_layers.append(nn.Conv2d(args.num_kernels, args.num_channels, args.kernel_size, padding=args.padding , bias=False))



    def forward(self, x, t):
        t = self.time_emb(t)
        
        # activations = []
        relu = nn.ReLU(inplace=True)
        x = self.conv_layers[0](x) #first layer linear

        for l in range(1,self.num_layers-1):
            x = self.conv_layers[l](x)
            x += self.time_embeddings[l-1](t).unsqueeze(-1).unsqueeze(-1)
            x = self.BN_layers[l-1](x)
            # activations.append((x>0))
            x = relu(x)

        x = self.conv_layers[-1](x)
        # return x, activations
        return x















class BF_CNN(nn.Module):

    def __init__(self, args): 
        super(BF_CNN, self).__init__()


        self.num_layers = args.num_layers
        self.first_layer_linear = args.first_layer_linear
        
        self.conv_layers = nn.ModuleList([])
        self.BN_layers = nn.ModuleList([])


        self.conv_layers.append(nn.Conv2d(args.num_channels,args.num_kernels, args.kernel_size, padding=args.padding , bias=False))

        for l in range(1,self.num_layers-1):
            self.conv_layers.append(nn.Conv2d(args.num_kernels ,args.num_kernels, args.kernel_size, padding=args.padding , bias=False))
            self.BN_layers.append(BF_batchNorm(args.num_kernels ))

        self.conv_layers.append(nn.Conv2d(args.num_kernels,args.num_channels, args.kernel_size, padding=args.padding , bias=False))



    def forward(self, x):
        relu = nn.ReLU(inplace=True)

        x = self.conv_layers[0](x) #first layer linear (different from orginal/old implementation)
        if self.first_layer_linear is False: 
            x = relu(x)

        for l in range(1,self.num_layers-1):
            x = self.conv_layers[l](x)
            x = self.BN_layers[l-1](x)
            x = relu(x)
        x = self.conv_layers[-1](x)

        return x 
    