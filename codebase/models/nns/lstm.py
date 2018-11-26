import torch
from codebase import utils as ut
from torch import nn

HIDDEN_DIM = 64
NUM_LAYERS = 1
BIDIRECTIONAL = False

class Encoder(nn.Module):
    def __init__(self, z_dim, x_dim, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.embedder = nn.Linear(1 + y_dim, HIDDEN_DIM)
        self.rnn = nn.LSTM(
            input_size=HIDDEN_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            bidirectional=BIDIRECTIONAL,
            batch_first=True,
            # bias=True,
            # dropout=0,
        )
        self.regressor = nn.Linear(NUM_LAYERS * (1+BIDIRECTIONAL) * HIDDEN_DIM, 2 * z_dim)

    def encode(self, x, y=None):
        x = x.unsqueeze(2)
        if y is not None:
            y = y.unsqueeze(1).expand(y.size()[0], self.x_dim, self.y_dim)
            x = torch.cat((x, y), dim=2)

        input = self.embedder(x)
        _, (h_n, _) = self.rnn(input=input)

        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        # reshape to (batch, last_hidden_outputs)
        h = h_n.permute(1, 0, 2).view(-1, NUM_LAYERS * (1+BIDIRECTIONAL) * HIDDEN_DIM)
        out = self.regressor(h)

        m, v = ut.gaussian_parameters(out, dim=-1)
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
