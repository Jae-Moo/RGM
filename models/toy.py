import torch
import torch.nn as nn
import numpy as np
import math


class ToyDiscriminator(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=32):
        super().__init__()
        self.x_module = FC(data_dim+1, hidden_dim, num_blocks=3)
        self.out_module = nn.Sequential(nn.Linear(hidden_dim, 1),)
    
    def forward(self, x, t):
        x = x.float()
        
        x_out = torch.cat([x, t[:,None]/4.], dim=1)
        x_out = self.x_module(x_out)
        out   = self.out_module(x_out)
        return out, out


class ToyGenerator(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=32, nz=2):
        super().__init__()
        self.x_module = FC(data_dim+1+nz, hidden_dim, num_blocks=3)
        self.out_module = nn.Sequential(nn.Linear(hidden_dim, data_dim),)
    
    def forward(self, x, t, latent_z):
        x = x.float()
        x_out = torch.cat([x, t[:,None]/4., latent_z], dim=1)

        x_out = self.x_module(x_out)
        out = self.out_module(x_out)
        return out


class FC(nn.Module):
    def __init__(self, data_dim=3, hidden_dim=512, num_blocks=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        modules = [nn.Linear(data_dim, hidden_dim), nn.Tanh()]
        for i in range(num_blocks-1):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.Tanh())
        self.fc = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.fc(x)