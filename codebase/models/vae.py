import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2, x_dim=24, warmup=False):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.warmup = warmup
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

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations
            fixed_var: tensor: bool: whether to simplify loss to a MSE

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        # encode
        qm, qv = self.enc.encode(x)
        # sample z(1) (for monte carlo estimate of p(x|z(1))
        z = ut.sample_gaussian(qm, qv)
        # decode
        mu, var = self.dec.decode(z)

        # KL Term
        kl_elem = ut.kl_normal(qm, qv, self.z_prior_m, self.z_prior_v)
        # reduce
        kl = kl_elem.mean(-1)

        # Rec Term
        # rec = ut.mse_loss(x, logits).mean(-1)
        rec = ut.nlog_prob_normal(mu=mu, y=x, var=var, fixed_var=self.warmup).mean(-1)

        # negative ELBO
        nelbo = kl + rec
        ################################################################################
        # End of code modification
        ################################################################################
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
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be scalar
        ################################################################################
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
        kl_elem = ut.kl_normal(qm, qv, self.z_prior_m, self.z_prior_v)

        niwae = -ut.log_mean_exp(log_prob - kl_elem, dim=1).mean(-1)

        # reduce
        rec = -log_prob.mean(1).mean(-1)
        kl = kl_elem.mean(1).mean(-1)

        ################################################################################
        # End of code modification
        ################################################################################
        return niwae, kl, rec

    def loss(self, x):
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
