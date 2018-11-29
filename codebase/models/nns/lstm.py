import torch
from codebase import utils as ut
from torch import nn

HIDDEN_DIM = 64
NUM_LAYERS = 2
BIDIRECTIONAL = False


class Encoder(nn.Module):
    def __init__(self, z_dim, x_dim, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.initializer = nn.Linear(x_dim + y_dim,
                                     2 * NUM_LAYERS * (1 + BIDIRECTIONAL) * HIDDEN_DIM)
        self.elu_i = nn.ELU()
        self.embedder = nn.Linear(1 + y_dim, HIDDEN_DIM)
        self.elu = nn.ELU()
        self.rnn = nn.LSTM(
            input_size=HIDDEN_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            bidirectional=BIDIRECTIONAL,
            batch_first=False,
            # bias=True,
            # dropout=0,
        )
        self.regressor = nn.Linear(NUM_LAYERS * (1+BIDIRECTIONAL) * HIDDEN_DIM, 2 * z_dim)

    def encode(self, x, y=None):
        if y is not None:
            xy = torch.cat((x, y), dim=-1)
            y = y.unsqueeze(1).expand(y.size()[0], self.x_dim, self.y_dim)
            x = torch.cat((x.unsqueeze(2), y), dim=2)
        else:
            xy = x
            x = x.unsqueeze(2)

        states = self.elu(self.initializer(xy)).view(
            x.size()[0], NUM_LAYERS * (1 + BIDIRECTIONAL), 2 * HIDDEN_DIM)
        (h_0, c_0) = states.split(HIDDEN_DIM, dim=-1)

        input = self.elu(self.embedder(x))
        # print(x.size())
        # print(input.size())
        _, (h_n, _) = self.rnn(input.transpose(1, 0), (h_0.transpose(1, 0), c_0.transpose(1, 0)))

        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        # reshape to (batch, last_hidden_outputs)
        h = h_n.transpose(1, 0).reshape(-1, NUM_LAYERS * (1+BIDIRECTIONAL) * HIDDEN_DIM)
        out = self.regressor(h)

        m, v = ut.gaussian_parameters(out, dim=-1)
        return m, v


class Decoder(nn.Module):
    def __init__(self, z_dim, x_dim, y_dim=0, c_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.initializer = nn.Linear(z_dim + y_dim + c_dim,
                                     2 * NUM_LAYERS * (1+BIDIRECTIONAL) * HIDDEN_DIM)
        self.elu = nn.ELU()
        self.rnn = nn.LSTM(
            input_size=z_dim + y_dim + c_dim,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            bidirectional=BIDIRECTIONAL,
            batch_first=False,
            # bias=True,
            # dropout=0,
        )
        self.regressor = nn.Linear((1+BIDIRECTIONAL) * HIDDEN_DIM, 2)

    def decode(self, z, y=None, c=None):
        # Note: not designed for IW!!
        assert len(z.size()) <= 2

        zy = z if y is None else torch.cat((z, y), dim=-1)
        zyc = zy if c is None else torch.cat((zy, c), dim=-1)

        states = self.elu(self.initializer(zyc)).view(
            z.size()[0], NUM_LAYERS * (1+BIDIRECTIONAL), 2*HIDDEN_DIM)

        (h_0, c_0) = states.split(HIDDEN_DIM, dim=-1)

        input = zyc.unsqueeze(1).expand(*zyc.size()[0:-1], self.x_dim, zyc.size()[-1])

        output, (_, _) = self.rnn(input.transpose(1, 0), (h_0.transpose(1, 0), c_0.transpose(1, 0)))
        out = self.regressor(output.transpose(1, 0)).transpose(-2, -1).reshape(z.size()[0], -1)
        m, v = ut.gaussian_parameters(out, dim=-1)
        return m, v
