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

    def nelbo_niwae_for(self, x, x_hat, y, c, iw):
        # switch between nelbo, niwae
        if iw < 1:
            return self.negative_elbo_bound_for(x, x_hat, y, c)
        else:
            return self.negative_iwae_bound_for(x, x_hat, y, c, iw)

    def negative_elbo_bound_for(self, x, x_hat, y, c):
        qm, qv = self.enc.encode(x, y=y)
        # sample z(1) (for monte carlo estimate of p(x|z(1))
        z = ut.sample_gaussian(qm, qv)

        kl = self.kl_elem(z, qm, qv)

        # decode
        mu, var = self.dec.decode(z, y=y, c=c)
        rec, rec_mse, rec_var = ut.nlog_prob_normal(
            mu=mu, y=x_hat, var=var, fixed_var=self.warmup, var_pen=self.var_pen)

        # reduce
        kl = kl.mean(-1)
        rec, rec_mse, rec_var = rec.mean(-1), rec_mse.mean(-1), rec_var.mean(-1)
        nelbo = kl + rec
        return nelbo, kl, rec, rec_mse, rec_var

    def negative_iwae_bound_for(self, x, x_hat, y, c, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            x_hat: tensor: (batch, dim): Observations
            y: tensor: (batch, y_dim): whether observations contain EV
            c: tensor: (batch, c_dim): target mapping specification
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        # encode
        qm, qv = self.enc.encode(x, y=y)

        # replicate qm, qv
        q_shape = list(qm.shape)
        qm = qm.unsqueeze(1).expand(q_shape[0], iw, *q_shape[1:])
        qv = qv.unsqueeze(1).expand(q_shape[0], iw, *q_shape[1:])
        # replicate x, y, c
        x_shape = list(x_hat.shape)
        x_hat = x_hat.unsqueeze(1).expand(x_shape[0], iw, *x_shape[1:])
        y_shape = list(y.shape)
        y = y.unsqueeze(1).expand(y_shape[0], iw, *y_shape[1:])
        c_shape = list(c.shape)
        c = c.unsqueeze(1).expand(c_shape[0], iw, *c_shape[1:])

        # sample z(1)...z(iw) (for monte carlo estimate of p(x|z(1))
        z = ut.sample_gaussian(qm, qv)

        kl_elem = self.kl_elem(z, qm, qv)

        # decode
        mu, var = self.dec.decode(z, y=y, c=c)

        nll, rec_mse, rec_var = ut.nlog_prob_normal(
            mu=mu, y=x_hat, var=var, fixed_var=self.warmup, var_pen=self.var_pen)
        log_prob, rec_mse, rec_var = -nll, rec_mse.mean(), rec_var.mean()

        niwae = -ut.log_mean_exp(log_prob - kl_elem, dim=1).mean(-1)

        # reduce
        rec = -log_prob.mean(1).mean(-1)
        kl = kl_elem.mean(1).mean(-1)
        return niwae, kl, rec, rec_mse, rec_var

    def compute_nelbo_niwae(self, x_0, x_1, y_real, x_no_ev=None, iw=0):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x_no_ev: tensor: (batch, dim): Observations that do not have EV
            x_0: tensor: (batch, dim): Observations with EV removed
            x_1: tensor: (batch, dim): Observations including EV
            y_real: tensor: (batch,): Magnitude of EV for given (x_1 - x_0)
            iw: int: (): Number of importance weighted samples

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        # y = np.int(y_real > 1)
        # y = np.eye(self.y_dim)[y]

        # identity mapping: x_no_ev
        if x_no_ev is not None:
            y_no_ev = torch.zeros((x_no_ev.shape[0], self.y_dim))
            y_no_ev[:, 0] = 1
            c_no_ev = torch.zeros((x_no_ev.shape[0], self.c_dim))
            loss_no_ev = self.nelbo_niwae_for(x=x_no_ev, x_hat=x_no_ev, y=y_no_ev, c=c_no_ev, iw=iw)

        # identity mapping: x_0
        y_0 = torch.zeros((y_real.shape[0], self.y_dim))
        y_0[:, 0] = 1
        c_0 = torch.zeros((y_real.shape[0], self.c_dim))
        if x_no_ev is None:
            # note: is done already with x_no_ev
            loss_0_0 = self.nelbo_niwae_for(x=x_0, x_hat=x_0, y=y_0, c=c_0, iw=iw)

        # identity mapping: x_1
        y_1 = torch.zeros((y_real.shape[0], self.y_dim))
        y_1[:, 1] = 1
        loss_1_1 = self.nelbo_niwae_for(x=x_1, x_hat=x_1, y=y_1, c=c_0, iw=iw)

        # add EV:
        c_add = torch.zeros((y_real.shape[0], self.c_dim))
        c_add[:, 0] = y_real
        loss_0_1 = self.nelbo_niwae_for(x=x_0, x_hat=x_1, y=y_0, c=c_add, iw=iw)

        # subtract EV:
        c_sub = torch.zeros((y_real.shape[0], self.c_dim))
        c_sub[:, 1] = y_real
        loss_1_0 = self.nelbo_niwae_for(x=x_1, x_hat=x_0, y=y_1, c=c_sub, iw=iw)

        # could also handle case of subtracting from x_0 and adding to x_1

        loss = []
        for i in range(len(loss_1_1)):
            if x_no_ev is not None:
                loss.append((loss_no_ev[i] + loss_1_1[i] + loss_0_1[i] + loss_1_0[i]) / 4.0)
            else:
                loss.append((loss_0_0[i] + loss_1_1[i] + loss_0_1[i] + loss_1_0[i]) / 4.0)
        return tuple(loss)

    def loss(self, sample, iw=0):
        nelbo, kl, rec, rec_mse, rec_var = self.compute_nelbo_niwae(
            x_no_ev=sample["x_no_ev"],
            x_0=sample["x_0"],
            x_1=sample["x_1"],
            y_real=sample["y_real"],
            iw=iw,
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

    def sample_x(self, batch, y, c):
        z = self.sample_z(batch)
        return self.sample_x_given(z, y, c)

    def sample_x_given(self, z, y, c):
        return self.dec.decode(z, y, c)

    def set_to_eval(self):
        self.warmup = False
        self.var_pen = 1




class GMCVAE(CVAE):
    def __init__(self, nn='v1', name='gmvae', z_dim=2, x_dim=24, warmup=False, var_pen=1,
                 k=100):
        super().__init__(nn, name, z_dim, x_dim, warmup, var_pen=var_pen)
        # Mixture of Gaussians prior
        self.k = k
        self.z_pre = torch.nn.Parameter(
            torch.randn(1, 2 * self.k, self.z_dim) / np.sqrt(self.k * self.z_dim))

    def kl_elem(self, z, qm, qv):
        # Compute the mixture of Gaussian prior
        prior_m, prior_v = ut.gaussian_parameters(self.z_pre, dim=1)

        log_prob_net = ut.log_normal(z, qm, qv)
        log_prob_prior = ut.log_normal_mixture(z, prior_m, prior_v)

        # print("log_prob_net:", log_prob_net.mean(), "log_prob_prior:", log_prob_prior.mean())
        kl_elem = log_prob_net - log_prob_prior
        return kl_elem
