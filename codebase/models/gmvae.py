import numpy as np
import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F
from codebase.models.vae import VAE


class GMVAE(VAE):
    def __init__(self, nn='v1', name='gmvae', z_dim=2, x_dim=24, warmup=False, k=500, var_pen=1):
        super().__init__(nn, name, z_dim, x_dim, warmup, var_pen=var_pen)
        self.k = k

        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(
            torch.randn(1, 2 * self.k, self.z_dim) / np.sqrt(self.k * self.z_dim))
        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)

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

        # Compute the mixture of Gaussian prior
        prior_m, prior_v = ut.gaussian_parameters(self.z_pre, dim=1)

        # encode
        # print(x.size())
        qm, qv = self.enc.encode(x)
        # print(qm.size())

        # sample z(1) (for monte carlo estimate of p(x|z(1))
        z = ut.sample_gaussian(qm, qv)

        # decode
        mu, var = self.dec.decode(z)

        # KL Term
        log_prob_net = ut.log_normal(z, qm, qv)
        log_prob_prior = ut.log_normal_mixture(z, prior_m, prior_v)
        # print("log_prob_net:", log_prob_net.mean(), "log_prob_prior:", log_prob_prior.mean())
        kl_elem = log_prob_net - log_prob_prior
        # reduce
        kl = kl_elem.mean(-1)

        # Rec Term
        rec = ut.nlog_prob_normal(
            mu=mu, y=x, var=var, fixed_var=self.warmup, var_pen=self.var_pen
        ).mean(-1)

        # negative ELBO
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

        # Compute the mixture of Gaussian prior
        prior_m, prior_v = ut.gaussian_parameters(self.z_pre, dim=1)

        # encode
        qm, qv = self.enc.encode(x)

        # replicate
        q_shape = list(qm.shape)
        x_shape = list(x.shape)
        qm = qm.unsqueeze(1).expand(q_shape[0], iw, *q_shape[1:])
        qv = qv.unsqueeze(1).expand(q_shape[0], iw, *q_shape[1:])
        # sample z(1)...z(iw) (for monte carlo estimate of p(x|z(1))
        z = ut.sample_gaussian(qm, qv)
        # decode
        logits = self.dec.decode(z)

        # Rec Term
        x = x.unsqueeze(1).expand(x_shape[0], iw, *x_shape[1:])
        log_prob = ut.log_bernoulli_with_logits(x, logits)

        # KL Term
        # KL Term
        log_prob_net = ut.log_normal(z, qm, qv)
        log_prob_prior = ut.log_normal_mixture(z, prior_m, prior_v)
        kl_elem = log_prob_net - log_prob_prior

        niwae = -ut.log_mean_exp(log_prob - kl_elem, dim=1).mean(-1)

        # reduce
        rec = -log_prob.mean(1).mean(-1)
        kl = kl_elem.mean(1).mean(-1)

        return niwae, kl, rec
