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