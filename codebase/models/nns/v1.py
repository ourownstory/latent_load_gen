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
            # nn.Linear(64, 64),
            # nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, 2 * z_dim),
        )

    def encode(self, x, y=None):
        xy = x if y is None else torch.cat((x, y), dim=1)
        h = self.net(xy)
        m, v = ut.gaussian_parameters(h, dim=1)
        return m, v


class Decoder(nn.Module):
    def __init__(self, z_dim, x_dim, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + y_dim, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            # nn.Linear(64, 64),
            # nn.ELU(),
        )
        self.mu_linear = nn.Linear(32, x_dim)
        self.var_linear = nn.Linear(32, x_dim)
        self.softplus = torch.nn.Softplus()

    def decode(self, z, y=None):
        zy = z if y is None else torch.cat((z, y), dim=1)
        pre_logits = self.net(zy)
        mu = self.mu_linear(pre_logits)
        var = self.softplus(self.var_linear(pre_logits))
        return mu, var


class Classifier(nn.Module):
    def __init__(self, y_dim):
        super().__init__()
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(784, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, y_dim)
        )

    def classify(self, x):
        return self.net(x)
