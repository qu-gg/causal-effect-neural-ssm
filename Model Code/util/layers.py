"""
@file layers.py

This script holds miscellaneous helper Torch layers
"""
import torch
import torch.nn as nn


class Gaussian(nn.Module):
    """
    Gaussian sample layer with 2 simple linear layers
    Code modified from: https://github.com/jariasf/GMVAE/blob/master/pytorch/networks/Layers.py
    """
    def __init__(self, in_dim, z_dim):
        super(Gaussian, self).__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.var = nn.Linear(in_dim, z_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z

    def forward(self, x):
        mu = self.mu(x)
        logvar = self.var(x)

        # Clamp the variance
        if (mu < -100).any() or (mu > 85).any() or (logvar < -100).any() or (logvar > 85).any():
            mu = torch.clamp(mu, min=-100, max=85)
            logvar = torch.clamp(logvar, min=-100, max=85)
            print("Explosion in mu/logvar of component")

        # Reparameterize and sample
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z


class Flatten(nn.Module):
    """
    Handles flattening a Tensor within a Sequential
    Code from: https://github.com/cagatayyildiz/ODE2VAE/blob/master/torch_ode2vae_minimal.py
    """
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    """
    Handles unflattening a vector into a 4D vector in a Sequential object
    Code from: https://github.com/cagatayyildiz/ODE2VAE/blob/master/torch_ode2vae_minimal.py
    """
    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, input):
        nc = input[0].numel() // (self.w ** 2)
        return input.view(input.size(0), nc, self.w, self.w)
