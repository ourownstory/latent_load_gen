import torch
from codebase import utils as ut
from torch import nn


class Encoder(nn.Module):
    def __init__(self, z_dim, x_dim, y_dim=0, c_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.c_dim = c_dim
        self.HIDDEN_DIM = 64
        self.NUM_LAYERS = 1
        self.BIDIRECTIONAL = False

        self.initializer = nn.Linear(x_dim + y_dim + c_dim,
                                     2 * self.NUM_LAYERS * (1 + self.BIDIRECTIONAL) * self.HIDDEN_DIM)
        self.elu_i = nn.ELU()
        self.embedder = nn.Linear(1 + 1 + y_dim + c_dim, self.HIDDEN_DIM)
        self.elu = nn.ELU()
        self.rnn = nn.LSTM(
            input_size=self.HIDDEN_DIM,
            hidden_size=self.HIDDEN_DIM,
            num_layers=self.NUM_LAYERS,
            bidirectional=self.BIDIRECTIONAL,
            batch_first=False,
            # bias=True,
            # dropout=0,
        )
        self.regressor = nn.Linear(self.NUM_LAYERS * (1+self.BIDIRECTIONAL) * self.HIDDEN_DIM, 2 * z_dim)

    def encode(self, x, y=None, c=None):
        xy = x if y is None else torch.cat((x, y), dim=-1)
        xyc = xy if c is None else torch.cat((xy, c), dim=-1)
        states = self.elu(self.initializer(xyc)).view(
            x.size()[0], self.NUM_LAYERS * (1 + self.BIDIRECTIONAL), 2 * self.HIDDEN_DIM)
        (h_0, c_0) = states.split(self.HIDDEN_DIM, dim=-1)

        x = x.unsqueeze(2)
        position = torch.arange(self.x_dim, dtype=torch.float) / (0.5 * (self.x_dim - 1)) - 1.0
        position = position.view(1, self.x_dim).expand(x.size()[0], self.x_dim).unsqueeze(2)
        x = torch.cat((x, position), dim=2)

        if y is not None:
            y = y.unsqueeze(1).expand(x.size()[0], self.x_dim, self.y_dim)
            x = torch.cat((x, y), dim=2)
        if c is not None:
            c = c.unsqueeze(1).expand(x.size()[0], self.x_dim, self.c_dim)
            x = torch.cat((x, c), dim=2)

        input = self.elu(self.embedder(x))
        # print(x.size())
        # print(input.size())
        _, (h_n, _) = self.rnn(input.transpose(1, 0), (h_0.transpose(1, 0), c_0.transpose(1, 0)))

        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        # reshape to (batch, last_hidden_outputs)
        h = h_n.transpose(1, 0).reshape(-1, self.NUM_LAYERS * (1+self.BIDIRECTIONAL) * self.HIDDEN_DIM)
        out = self.regressor(h)

        m, v = ut.gaussian_parameters(out, dim=-1)
        return m, v


class Decoder(nn.Module):
    def __init__(self, z_dim, x_dim, y_dim=0, c_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.HIDDEN_DIM = 64
        self.NUM_LAYERS = 1
        self.BIDIRECTIONAL = False

        self.initializer = nn.Linear(z_dim + y_dim + c_dim,
                                     2 * self.NUM_LAYERS * (1+self.BIDIRECTIONAL) * self.HIDDEN_DIM)
        self.elu = nn.ELU()
        self.rnn = nn.LSTM(
            input_size=z_dim + y_dim + c_dim + 1,
            hidden_size=self.HIDDEN_DIM,
            num_layers=self.NUM_LAYERS,
            bidirectional=self.BIDIRECTIONAL,
            batch_first=False,
            # bias=True,
            # dropout=0,
        )
        self.regressor = nn.Linear((1+self.BIDIRECTIONAL) * self.HIDDEN_DIM, 2)

    def decode(self, z, y=None, c=None):
        # Note: not designed for IW!!
        assert len(z.size()) <= 2

        zy = z if y is None else torch.cat((z, y), dim=-1)
        zyc = zy if c is None else torch.cat((zy, c), dim=-1)

        states = self.elu(self.initializer(zyc)).view(
            z.size()[0], self.NUM_LAYERS * (1+self.BIDIRECTIONAL), 2*self.HIDDEN_DIM)

        (h_0, c_0) = states.split(self.HIDDEN_DIM, dim=-1)

        input = zyc.unsqueeze(1).expand(*zyc.size()[0:-1], self.x_dim, zyc.size()[-1])

        position = torch.arange(self.x_dim, dtype=torch.float) / (0.5 * (self.x_dim - 1)) - 1.0
        position = position.view(1, self.x_dim).expand(z.size()[0], self.x_dim).unsqueeze(2)
        input = torch.cat((input, position), dim=2)

        output, (_, _) = self.rnn(input.transpose(1, 0), (h_0.transpose(1, 0), c_0.transpose(1, 0)))
        out = self.regressor(output.transpose(1, 0)).transpose(-2, -1).reshape(z.size()[0], -1)
        m, v = ut.gaussian_parameters(out, dim=-1)
        return m, v
