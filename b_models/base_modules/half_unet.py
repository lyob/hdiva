import torch
from torch import nn

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
        self.out_channels = args.num_kernels_rec*(2**(b+1)) if l!=args.num_mid_conv_rec-1 else args.num_kernels_rec*(2**(b+2))
        # self.out_channels = args.num_kernels_rec*(2**(b+1))
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


class CompressBlock(nn.Module):
    """Compress the feature map to a vector of latent dimensions, such that we can compute mean and variance for VAE.
    We pool over the spatial dimensions and use a linear layer to get the desired output dimensions.
    Input: (B, C, H, W)
    Output: (B, 2*z_dim)  (mean and variance for each latent dimension)
    """
    def __init__(self, c_in, z_dim):
        super().__init__()
        self.compress_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, C, 1, 1)
            nn.Flatten(),  # (B, C)
            nn.Linear(c_in, 2*z_dim, bias=False)  # (B, 2*z_dim)
        )
        
    def forward(self, x:torch.Tensor):
        return self.compress_block(x)


class Half_UNet(nn.Module): 
    def __init__(self, args): 
        super(Half_UNet, self).__init__()
        # args: num_channels, num_kernels, kernel_size, padding, bias, num_enc_conv, num_dec_conv, pool_window, num_blocks, num_mid_conv, time_channels
        self.latent_dim = args.latent_dim

        ########## First Block ##########
        self.first = FirstBlock(args)
        
        ########## Down Blocks ##########
        self.down = nn.ModuleList([])
        for b in range(args.num_blocks_rec):
            self.init_encoder_block(b, args)
            self.down.append(DownSample(args))

        ########## Mid-layers ##########
        self.mid = nn.ModuleList([])
        for l in range(args.num_mid_conv_rec):
            if args.downsample_in_mid_block_rec and l>0:
                self.mid.append(DownSample(args))
            self.mid.append(MiddleBlock(args, l))
        
        ########## Compress Block ##########
        final_num_channels = args.num_kernels_rec * (2**(args.num_blocks_rec+1))  # number of channels after last downsample
        self.compress = CompressBlock(final_num_channels, args.latent_dim)

    def init_encoder_block(self, b, args):
        if b==0:
            for l in range(1, args.num_enc_conv_rec):
                self.down.append(DownBlock(args,b,l))
        else:
            for l in range(args.num_enc_conv_rec):
                self.down.append(DownBlock(args,b,l))

    def forward(self, x:torch.Tensor):
        ########## Encoder ##########
        x = self.first(x)
        for d in self.down:
            x = d(x)
        ########## Mid-layers ##########
        for m in self.mid:
            x = m(x)
        ########## Decoder ##########
        x = self.compress(x)  # (B, 2*z_dim)
        
        mu, logvar = x.chunk(2, dim=1)  # each (B, z_dim)
        return mu, logvar
    
    def sample(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std  # reparameterization trick



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