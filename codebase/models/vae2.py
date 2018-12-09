import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
import numpy as np


class VAE2(nn.Module):
    def __init__(self, nn='v1', name='vae2', z_dim=10, x_dim=24, c_dim=0,
                 warmup=False, var_pen=1, y_dim=0):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.c_dim = c_dim
        self.warmup = warmup
        self.var_pen = var_pen
        # Small note: unfortunate name clash with torch.nn
        # nn here refers to the specific architecture file found in
        # codebase/models/nns/*.py
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim, self.x_dim, self.y_dim, self.c_dim)
        self.dec = nn.Decoder(self.z_dim, self.x_dim, self.y_dim, self.c_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def kl_elementwise(self, z, qm, qv):
        kl_elem = ut.kl_normal(qm, qv, self.z_prior_m, self.z_prior_v)
        return kl_elem

    def nelbo_niwae(self, x, y, c, iw):
        """
        switches between nelbo, niwae

        Args:
            x: tensor: (batch, dim): Observations
            y: tensor: (batch, y_dim): metadata
            c: tensor: (batch, c_dim): conditionals
            iw: int: (): Number of importance weighted samples
        """
        if iw < 1:
            return self.negative_elbo_bound_for(x, y, c)
        else:
            return self.negative_iwae_bound_for(x, y, c, iw)

    def negative_elbo_bound_for(self, x, y, c):
        qm, qv = self.enc.encode(x, y=y, c=c)
        # sample z(1) (for monte carlo estimate of p(x|z(1))
        z = ut.sample_gaussian(qm, qv)

        kl = self.kl_elementwise(z, qm, qv)

        # decode
        mu, var = self.dec.decode(z, y=y, c=c)
        rec, rec_mse, rec_var = ut.nlog_prob_normal(
            mu=mu, y=x, var=var, fixed_var=self.warmup, var_pen=self.var_pen)

        # reduce
        kl = kl.mean(-1)
        rec, rec_mse, rec_var = rec.mean(-1), rec_mse.mean(-1), rec_var.mean(-1)
        nelbo = kl + rec
        return nelbo, kl, rec, rec_mse, rec_var

    def negative_iwae_bound_for(self, x, y, c, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        # encode
        qm, qv = self.enc.encode(x, y=y, c=c)

        # replicate qm, qv
        q_shape = list(qm.shape)
        qm = qm.unsqueeze(1).expand(q_shape[0], iw, *q_shape[1:])
        qv = qv.unsqueeze(1).expand(q_shape[0], iw, *q_shape[1:])

        # replicate x, y, c
        x_shape = list(x.shape)
        x = x.unsqueeze(1).expand(x_shape[0], iw, *x_shape[1:])
        if y is not None:
            y_shape = list(y.shape)
            y = y.unsqueeze(1).expand(y_shape[0], iw, *y_shape[1:])
        if c is not None:
            c_shape = list(c.shape)
            c = c.unsqueeze(1).expand(c_shape[0], iw, *c_shape[1:])

        # sample z(1)...z(iw) (for monte carlo estimate of p(x|z(1))
        z = ut.sample_gaussian(qm, qv)

        kl_elem = self.kl_elementwise(z, qm, qv)

        # decode
        mu, var = self.dec.decode(z, y=y, c=c)

        nll, rec_mse, rec_var = ut.nlog_prob_normal(
            mu=mu, y=x, var=var, fixed_var=self.warmup, var_pen=self.var_pen)
        log_prob, rec_mse, rec_var = -nll, rec_mse.mean(), rec_var.mean()

        niwae = -ut.log_mean_exp(log_prob - kl_elem, dim=1).mean(-1)

        # reduce
        rec = -log_prob.mean(1).mean(-1)
        kl = kl_elem.mean(1).mean(-1)
        return niwae, kl, rec, rec_mse, rec_var

    def loss(self, x, meta=None, c=None, iw=0):
        nelbo, kl, rec, rec_mse, rec_var = self.nelbo_niwae(
            x=x,
            y=meta,
            c=c,
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

    def sample_x(self, batch, y=None, c=None):
        z = self.sample_z(batch)
        return self.sample_x_given(z, y, c)

    def sample_x_given(self, z, y=None, c=None):
        if y is not None:
            y = torch.FloatTensor(y)
        if c is not None:
            c = torch.FloatTensor(c)
        return self.dec.decode(torch.FloatTensor(z), y, c)

    def set_to_eval(self):
        self.warmup = False
        self.var_pen = 1


class GMVAE2(VAE2):
    def __init__(self, nn='v1', name='gmvae2', z_dim=2, x_dim=24, c_dim=0,
                 warmup=False, var_pen=1,
                 k=100):
        super().__init__(nn, name, z_dim, x_dim, c_dim, warmup, var_pen=var_pen)
        # Mixture of Gaussians prior
        self.k = k
        self.z_pre = torch.nn.Parameter(
            torch.randn(1, 2 * self.k, self.z_dim) / np.sqrt(self.k * self.z_dim))

    def kl_elementwise(self, z, qm, qv):
        # Compute the mixture of Gaussian prior
        prior_m, prior_v = ut.gaussian_parameters(self.z_pre, dim=1)

        log_prob_net = ut.log_normal(z, qm, qv)
        log_prob_prior = ut.log_normal_mixture(z, prior_m, prior_v)

        # print("log_prob_net:", log_prob_net.mean(), "log_prob_prior:", log_prob_prior.mean())
        kl_elem = log_prob_net - log_prob_prior
        return kl_elem


class VAE2CAR(VAE2):
    def __init__(self, nn='v1', name='vae2car', z_dim=2, x_dim=24, c_dim=0,
                 warmup=False, var_pen=1,
                 use_model=None):
        super().__init__(nn, name, z_dim, x_dim, c_dim, warmup, var_pen=var_pen)
        self.use_model = use_model

    def negative_elbo_bound_for(self, x, y, c):
        # get conditional by encoding c, where c is expected to be x_use
        z_use_mu, z_use_var = self.use_model.enc.encode(
            x=c,
            y=y,
        )

        # TODO: decide
        # z_use = ut.sample_gaussian(z_use_mu, z_use_var)
        z_use = z_use_mu

        return super().negative_elbo_bound_for(x, y=y, c=z_use)

    # def negative_iwae_bound_for(self, x, y, c, iw):
    #     raise NotImplementedError

    def negative_iwae_bound_for(self, x, y, c, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        # get conditionals
        z_use_mu, z_use_var = self.use_model.enc.encode(x=c, y=y)

        # replicate c
        z_use_shape = list(z_use_mu.shape)
        z_use_mu = z_use_mu.unsqueeze(1).expand(z_use_shape[0], iw, *z_use_shape[1:])
        z_use_var = z_use_var.unsqueeze(1).expand(z_use_shape[0], iw, *z_use_shape[1:])
        # sample multiple c
        c = ut.sample_gaussian(z_use_mu, z_use_var)
        c = c.view(z_use_shape[0]*iw, *z_use_shape[1:])

        # replicate x, y
        x_shape = list(x.shape)
        x = x.unsqueeze(1).expand(x_shape[0], iw, *x_shape[1:])
        x = x.reshape(x_shape[0]*iw, *x_shape[1:])
        if y is not None:
            y_shape = list(y.shape)
            y = y.unsqueeze(1).expand(y_shape[0], iw, *y_shape[1:])
            y = y.reshape(y_shape[0]*iw, *y_shape[1:])

        # encode
        qm, qv = self.enc.encode(x, y=y, c=c)

        # replicate qm, qv
        q_shape = list(qm.shape)
        qm = qm.unsqueeze(1).expand(q_shape[0], iw, *q_shape[1:])
        # print(qm.shape, q_shape)
        qm = qm.reshape(q_shape[0]*iw, *q_shape[1:])
        qv = qv.unsqueeze(1).expand(q_shape[0], iw, *q_shape[1:])
        qv = qv.reshape(q_shape[0]*iw, *q_shape[1:])

        # replicate x, y, c
        x_shape = list(x.shape)
        x = x.unsqueeze(1).expand(x_shape[0], iw, *x_shape[1:])
        x = x.reshape(x_shape[0]*iw, *x_shape[1:])
        if y is not None:
            y_shape = list(y.shape)
            y = y.unsqueeze(1).expand(y_shape[0], iw, *y_shape[1:])
            y = y.reshape(y_shape[0]*iw, *y_shape[1:])
        c_shape = list(c.shape)
        c = c.unsqueeze(1).expand(c_shape[0], iw, *c_shape[1:])
        c = c.reshape(c_shape[0]*iw, *c_shape[1:])

        # sample z(1)...z(iw) (for monte carlo estimate of p(x|z(1))
        z = ut.sample_gaussian(qm, qv)

        kl_elem = self.kl_elementwise(z, qm, qv)

        # decode
        mu, var = self.dec.decode(z, y=y, c=c)

        nll, rec_mse, rec_var = ut.nlog_prob_normal(
            mu=mu, y=x, var=var, fixed_var=self.warmup, var_pen=self.var_pen)
        log_prob, rec_mse, rec_var = -nll, rec_mse.mean(), rec_var.mean()
        elem_shape = list(kl_elem.shape)
        kl_elem = kl_elem.view(z_use_shape[0], iw*iw)
        log_prob = log_prob.view(z_use_shape[0], iw*iw)
        niwae = -ut.log_mean_exp(log_prob - kl_elem, dim=1).mean(-1)

        # reduce
        rec = -log_prob.mean(1).mean(-1)
        kl = kl_elem.mean(1).mean(-1)
        return niwae, kl, rec, rec_mse, rec_var

    def sample_x(self, batch, y=None, c=None):
        if c is None:
            c = self.use_model.sample_z(batch)
        z = self.sample_z(batch)
        return self.sample_x_given(z, y, c)

    def sample_x_given(self, z, y=None, c=None):
        if y is not None:
            y = torch.FloatTensor(y)
        if c is not None:
            c = torch.FloatTensor(c)
        else:
            c = torch.zeros((z.size()[0], self.use_model.z_dim))
        return self.dec.decode(torch.FloatTensor(z), y, c)


class GMVAE2CAR(VAE2CAR):
    def __init__(self, nn='v1', name='gmvae2car', z_dim=2, x_dim=24, c_dim=0,
                 warmup=False, var_pen=1,
                 use_model=None,
                 k=100
                 ):
        super().__init__(nn, name, z_dim, x_dim, c_dim, warmup, var_pen=var_pen, use_model=use_model)
        # Mixture of Gaussians prior
        self.k = k
        self.z_pre = torch.nn.Parameter(
            torch.randn(1, 2 * self.k, self.z_dim) / np.sqrt(self.k * self.z_dim))

    def kl_elementwise(self, z, qm, qv):
        # Compute the mixture of Gaussian prior
        prior_m, prior_v = ut.gaussian_parameters(self.z_pre, dim=1)

        log_prob_net = ut.log_normal(z, qm, qv)
        log_prob_prior = ut.log_normal_mixture(z, prior_m, prior_v)

        # print("log_prob_net:", log_prob_net.mean(), "log_prob_prior:", log_prob_prior.mean())
        kl_elem = log_prob_net - log_prob_prior
        return kl_elem


