import numpy as np
import torch
import torch.nn.functional as F
from codebase import utils as ut
from torch import autograd, nn, optim
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, z_dim, x_dim, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, 2 * z_dim),
        )

    def encode(self, x, y=None):
        xy = x if y is None else torch.cat((x, y), dim=-1)
        h = self.net(xy)
        m, v = ut.gaussian_parameters(h, dim=-1)
        return m, v


class Decoder(nn.Module):
    def __init__(self, z_dim, x_dim, y_dim=0, c_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + y_dim + c_dim, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, 2*x_dim),
        )

    def decode(self, z, y=None, c=None):
        zy = z if y is None else torch.cat((z, y), dim=-1)
        zyc = zy if c is None else torch.cat((zy, c), dim=-1)
        h = self.net(zyc)
        m, v = ut.gaussian_parameters(h, dim=-1)
        return m, v
