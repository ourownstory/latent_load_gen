import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
import numpy as np


class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2, x_dim=24, warmup=False, var_pen=1):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.warmup = warmup
        self.var_pen = var_pen
        # Small note: unfortunate name clash with torch.nn
        # nn here refers to the specific architecture file found in
        # codebase/models/nns/*.py
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim, self.x_dim)
        self.dec = nn.Decoder(self.z_dim, self.x_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def kl_elem(self, z, qm, qv):
        kl_elem = ut.kl_normal(qm, qv, self.z_prior_m, self.z_prior_v)
        return kl_elem

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        # encode
        qm, qv = self.enc.encode(x)

        # sample z(1) (for monte carlo estimate of p(x|z(1))
        z = ut.sample_gaussian(qm, qv)

        # decode
        mu, var = self.dec.decode(z)

        rec = ut.nlog_prob_normal(
            mu=mu, y=x, var=var, fixed_var=self.warmup, var_pen=self.var_pen
        ).mean(-1)
        kl = self.kl_elem(z, qm, qv).mean(-1)
        nelbo = kl + rec
        return nelbo, kl, rec

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        # encode
        qm, qv = self.enc.encode(x)

        # replicate qm, qv
        q_shape = list(qm.shape)
        qm = qm.unsqueeze(1).expand(q_shape[0], iw, *q_shape[1:])
        qv = qv.unsqueeze(1).expand(q_shape[0], iw, *q_shape[1:])
        # replicate x
        x_shape = list(x.shape)
        x = x.unsqueeze(1).expand(x_shape[0], iw, *x_shape[1:])

        # sample z(1)...z(iw) (for monte carlo estimate of p(x|z(1))
        z = ut.sample_gaussian(qm, qv)

        # decode
        mu, var = self.dec.decode(z)

        log_prob = -ut.nlog_prob_normal(mu=mu, y=x, var=var,
                                        fixed_var=self.warmup, var_pen=self.var_pen)

        kl_elem = self.kl_elem(z, qm, qv)

        niwae = -ut.log_mean_exp(log_prob - kl_elem, dim=1).mean(-1)
        rec = -log_prob.mean(1).mean(-1)
        kl = kl_elem.mean(1).mean(-1)
        return niwae, kl, rec

    def loss(self, x, iw=1):
        if iw > 1:
            niwae, kl, rec = self.negative_iwae_bound(x, iw)
            loss = niwae
            summaries = dict((
                ('train/loss', niwae),
                ('gen/iwae', -niwae),
                ('gen/kl_z', kl),
                ('gen/rec', rec),
            ))
        else:
            nelbo, kl, rec = self.negative_elbo_bound(x)
            loss = nelbo
            summaries = dict((
                ('train/loss', nelbo),
                ('gen/elbo', -nelbo),
                ('gen/kl_z', kl),
                ('gen/rec', rec),
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


class GMVAE(VAE):
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
