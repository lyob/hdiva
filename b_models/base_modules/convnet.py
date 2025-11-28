import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class LinearInfNet(nn.Module):
    def __init__(self, input_dim, z_dim, bias):
        super(LinearInfNet, self).__init__()
        # self.base_unet = base_unet
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.bias = bias
        self.linear_mu = nn.Linear(input_dim, z_dim, bias=bias)
        self.linear_sig = nn.Linear(input_dim, z_dim, bias=bias)


    def forward(self, x, t):
        # x: [B, 3, 256, 256]
        x = torch.flatten(x, start_dim=1)
        mu = self.linear_mu(x)
        sigma = F.softplus(self.linear_sig(x))
        z = mu + sigma*torch.randn_like(mu)
        return z, mu, sigma



# --------------------------- convolutional encoder -------------------------- #
class ConvNet(nn.Module):
    def __init__(self, args):
        super(ConvNet, self).__init__()
        
        self.latent_dim:int = args.latent_dim
        self.num_layers:int = args.num_layers_rec
        kernel_size = args.kernel_size_rec
        stride = args.stride_rec
        padding = args.padding_rec
        nonlin = args.activation_rec
        bias = args.bias_rec

        assert self.num_layers > 0, 'num_layers must be greater than 0'
        self.kernels = []
        for i in range(self.num_layers):
            self.kernels.append(kernel_size)
        
        if nonlin == 'relu':
            self.nonlin = nn.ReLU() 
        elif nonlin == 'gelu':
            self.nonlin = nn.GELU()
        else:
            raise ValueError('nonlin must be specified, e.g. "relu" or "gelu"')
        
        '''first convolutional layer'''
        self.d1_input_channels = args.num_channels
        self.d1_input_dims = args.image_dim
        # input: 1 x 32 x 32
        # output: 4 x 16 x 16
        self.conv_d1 = nn.Conv2d(self.d1_input_channels, self.kernels[0], kernel_size, stride, padding, bias=bias)  # in_channels, out_channels, kernel_size, stride, padding
        
        if self.num_layers > 1:
            for i in range(1, self.num_layers):
                setattr(self, f'conv_d{i+1}', nn.Conv2d(self.kernels[i-1], self.kernels[i], kernel_size, stride, padding, bias=bias))
                setattr(self, f'bn_d{i+1}', BF_batchNorm(self.kernels[i]))
                setattr(self, f'd{i+1}_input_dims', self.compute_output_dims(getattr(self, f'd{i}_input_dims'), self.kernels[i-1], kernel_size, stride, padding))
                # if i%2 == 0:
                    # setattr(self, f'pool_d{i//2}', nn.AvgPool2d(kernel_size=2, stride=2, padding=1 ))
        
        '''linear layers'''
        linear_input_dims = self.compute_output_dims(getattr(self, f'd{self.num_layers}_input_dims'), self.kernels[-1], kernel_size, stride, padding)
        linear_input_dims = linear_input_dims**2 * self.kernels[-1]
        self.linear_mu = nn.Linear(linear_input_dims, self.latent_dim, bias=bias)
        self.linear_logvar = nn.Linear(linear_input_dims, self.latent_dim, bias=bias)
        
    def compute_output_dims(self, input_dim, output_channels, kernel_size, stride, padding):
        '''calculate the output dimensions of a convolutional layer, for a given input dimension and convolutional settings'''
        return ((input_dim - kernel_size + 2*padding)//stride + 1)
        
    def forward(self, x):
        for i in range(1, self.num_layers+1):
            x = getattr(self, f'conv_d{i}')(x)
            if i!=1:
                x = getattr(self, f'bn_d{i}')(x)
            x = self.nonlin(x)
        
        x = torch.flatten(x, start_dim=1)
        mu = self.linear_mu(x)
        logvar = self.linear_logvar(x)
        
        return mu, logvar

    def sample(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std
    


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