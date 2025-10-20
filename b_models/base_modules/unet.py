import math
import numpy as np
import torch.nn as nn
import torch


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
        
        ########## Time Embedding ##########
        self.time_emb = TimeEmbedding(args.time_channels)
        self.time_projection = LinearTimeProjection(args)
        
        ########## First Block ##########
        self.first = FirstBlock(args)
        
        ########## Down Blocks ##########
        # self.down = nn.ModuleDict([])
        self.down = nn.ModuleList([])
        for b in range(args.num_blocks):
            self.init_encoder_block(b, args)
            self.down.append(DownSample(args))

        ########## Mid-layers ##########
        self.mid = nn.ModuleList([])
        for l in range(args.num_mid_conv):
            self.mid.append(MiddleBlock(args, l))
                                    
        ########## Up Blocks ##########
        self.up = nn.ModuleList([])
        for b in range(args.num_blocks-1,-1,-1):
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

