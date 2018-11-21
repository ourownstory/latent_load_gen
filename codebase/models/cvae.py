import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
import numpy as np


class CVAE(nn.Module):
    def __init__(self, nn='v1', name='cvae', z_dim=2, x_dim=24, warmup=False, var_pen=1):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.y_dim = 2
        self.c_dim = 2
        self.warmup = warmup
        self.var_pen = var_pen
        # Small note: unfortunate name clash with torch.nn
        # nn here refers to the specific architecture file found in
        # codebase/models/nns/*.py
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim, self.x_dim, self.y_dim)
        self.dec = nn.Decoder(self.z_dim, self.x_dim, self.y_dim, self.c_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def kl_elem(self, z, qm, qv):
        kl_elem = ut.kl_normal(qm, qv, self.z_prior_m, self.z_prior_v)
        return kl_elem

    def negative_elbo_bound_for(self, x, x_hat, y, c):
        qm, qv = self.enc.encode(x, y=y)
        # sample z(1) (for monte carlo estimate of p(x|z(1))
        z = ut.sample_gaussian(qm, qv)

        kl = self.kl_elem(z, qm, qv).mean(-1)

        # decode
        mu, var = self.dec.decode(z, y=y, c=c)
        nll, rec_mse, rec_var = ut.nlog_prob_normal(
            mu=mu, y=x_hat, var=var, fixed_var=self.warmup, var_pen=self.var_pen)
        rec, rec_mse, rec_var = nll.mean(-1), rec_mse.mean(-1), rec_var.mean(-1)
        nelbo = kl + rec
        # loss = {"nelbo": kl + rec, "kl": kl, "rec": rec, "rec_mse": rec_mse, "rec_var": rec_var}
        return [nelbo, kl, rec, rec_mse, rec_var]

    def negative_elbo_bound(self, x_0, x_1, y_real):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x_0: tensor: (batch, dim): Observations without EV
            x_1: tensor: (batch, dim): Observations with EV
            y_real: tensor: (batch,): Magnitude of EV for given (x_1 - x_0)

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        # y = np.int(y_real > 1)
        # y = np.eye(self.y_dim)[y]

        # identity mapping: x_0
        y_0 = torch.zeros((y_real.shape[0], self.y_dim))
        y_0[:, 0] = 1
        c_0 = torch.zeros((y_real.shape[0], self.c_dim))
        loss_0_0 = self.negative_elbo_bound_for(x=x_0, x_hat=x_0, y=y_0, c=c_0)

        # identity mapping: x_1
        y_1 = torch.zeros((y_real.shape[0], self.y_dim))
        y_1[:, 1] = 1
        loss_1_1 = self.negative_elbo_bound_for(x=x_1, x_hat=x_1, y=y_1, c=c_0)

        # add EV:
        c_add = torch.zeros((y_real.shape[0], self.c_dim))
        c_add[:, 0] = y_real
        loss_0_1 = self.negative_elbo_bound_for(x=x_0, x_hat=x_1, y=y_0, c=c_add)

        # subtract EV:
        c_sub = torch.zeros((y_real.shape[0], self.c_dim))
        c_sub[:, 1] = y_real
        loss_1_0 = self.negative_elbo_bound_for(x=x_1, x_hat=x_0, y=y_1, c=c_sub)

        # could also handle case of subtracting from x_0 and adding to x_1

        loss = []
        for i in range(len(loss_0_0)):
            loss.append((loss_0_0[i] + loss_1_1[i] + loss_0_1[i] + loss_1_0[i])/4.0)

        return tuple(loss)

    def loss(self, sample, iw=1):
        if iw > 1:
            raise NotImplementedError
            # niwae, kl, rec, rec_mse, rec_var = self.negative_iwae_bound(x, iw)
            # loss = niwae
            # summaries = dict((
            #     ('train/loss', niwae),
            #     ('gen/iwae', -niwae),
            #     ('gen/kl_z', kl),
            #     ('gen/rec', rec),
            #     ('gen/rec_mse', rec_mse),
            #     ('gen/rec_var', rec_var),
            # ))
        else:
            nelbo, kl, rec, rec_mse, rec_var = self.negative_elbo_bound(
                x_0=sample["other"], x_1=sample["use"], y_real=sample["y_real"]
            )
            loss = nelbo
            summaries = dict((
                ('train/loss', nelbo),
                ('gen/elbo', -nelbo),
                ('gen/kl_z', kl),
                ('gen/rec', rec),
                ('gen/rec_mse', rec_mse),
                ('gen/rec_var', rec_var),
            ))

        return loss, summaries

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return self.dec.decode(z)

    def set_to_eval(self):
        self.warmup = False
        self.var_pen = 1
