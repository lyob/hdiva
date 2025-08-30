import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderMLPBlock(nn.Module):
    def __init__(self, input_channels, hidden_dims, z_dim, bias=True):
        super(EncoderMLPBlock, self).__init__()
        self.fc = nn.Linear(input_channels, hidden_dims, bias=bias)
        self.bn = nn.BatchNorm1d(hidden_dims)

        self._mu = nn.Linear(hidden_dims, z_dim, bias=bias)
        self._var = nn.Linear(hidden_dims, z_dim, bias=bias)

    def forward(self, x):
        x = F.relu(self.bn(self.fc(x)))
        mu = self._mu(x)
        var = F.softplus(self._var(x))
        return mu, var

class DecoderMLPBlock(nn.Module):
    def __init__(self, z1_dim, hidden_dim, z2_dim):
        self.fc1 = nn.Linear(z1_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self._mu = nn.Linear(hidden_dim, z2_dim)
        self._var = nn.Linear(hidden_dim, z2_dim)

    def forward(self, x):
        x = F.relu(self.bn(self.fc1(x)))
        _mu = self._mu(x)
        _var = F.softplus(self._var(x))
        return _mu, _var

class FinalDecoder(nn.Module):
    def __init__(self, z_final, hidden_dim, input_dim):
        self.fc1 = nn.Linear(z_final, hidden_dim)
        self._mu = nn.Linear(hidden_dim, input_dim)
        self._var = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        _mu = self._mu(x)
        return _mu
