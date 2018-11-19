import numpy as np
import os
import shutil
import torch
from codebase.models.vae import VAE, GMVAE
from torch.nn import functional as F
from HourlyLoadDataset import HourlyLoad2017Dataset


def nlog_prob_normal(mu, y, var=None, fixed_var=False, var_pen=1):
    diff = y - mu
    # makes it just MSE
    mse = torch.mul(diff, diff)
    if fixed_var:
        var_cost = torch.zeros(*y.shape)
    else:
        # return actual log-likelihood
        mse = torch.div(mse, var)
        var_cost = var_pen*torch.log(var)
        # these two last terms would make it a correct log(p), but are not required for MLE
        # cost += log_2pi
        # cost *= 0.5
    mse = mse.sum(-1)
    var_cost = var_cost.sum(-1)
    cost = mse + var_cost
    return cost, mse, var_cost


def sample_gaussian(m, v):
    """
    Element-wise application reparameterization trick to sample from Gaussian

    Args:
        m: tensor: (batch, ...): Mean
        v: tensor: (batch, ...): Variance

    Return:
        z: tensor: (batch, ...): Samples
    """
    # reparm
    norm = torch.distributions.normal.Normal(loc=0, scale=1)
    e = norm.sample(sample_shape=m.size())
    # shift scale
    z = m + torch.sqrt(v)*e
    return z


def log_normal(x, m, v):
    """
    Computes the elem-wise log probability of a Gaussian and then sum over the
    last dim. Basically we're assuming all dims are batch dims except for the
    last dim.

    Args:
        x: tensor: (batch_1, batch_2, ..., batch_k, dim): Observation
        m: tensor: (batch_1, batch_2, ..., batch_k, dim): Mean
        v: tensor: (batch_1, batch_2, ..., batch_k, dim): Variance

    Return:
        log_prob: tensor: (batch_1, batch_2, ..., batch_k): log probability of
            each sample. Note that the summation dimension is not kept
    """
    norm = torch.distributions.normal.Normal(loc=m, scale=v.sqrt())
    log_prob = norm.log_prob(x).sum(-1)
    return log_prob


def log_normal_mixture(z, m, v):
    """
    Computes log probability of a uniformly-weighted Gaussian mixture.

    Args:
        z: tensor: (batch, dim): Observations
        m: tensor: (batch, mix, dim): Mixture means
        v: tensor: (batch, mix, dim): Mixture variances

    Return:
        log_prob: tensor: (batch,): log probability of each sample
    """
    x = z.unsqueeze(-2).expand(*z.shape[:-1], m.size()[1], z.shape[-1])
    log_prob = log_mean_exp(log_normal(x, m, v), -1)
    return log_prob


def gaussian_parameters(h, dim=-1):
    """
    Converts generic real-valued representations into mean and variance
    parameters of a Gaussian distribution

    Args:
        h: tensor: (batch, ..., dim, ...): Arbitrary tensor
        dim: int: (): Dimension along which to split the tensor for mean and
            variance

    Returns:
        m: tensor: (batch, ..., dim / 2, ...): Mean
        v: tensor: (batch, ..., dim / 2, ...): Variance
    """
    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v


def kl_normal(qm, qv, pm, pv):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
    sum over the last dimension

    Args:
        qm: tensor: (batch, dim): q mean
        qv: tensor: (batch, dim): q variance
        pm: tensor: (batch, dim): p mean
        pv: tensor: (batch, dim): p variance

    Return:
        kl: tensor: (batch,): kl between each sample
    """
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return kl


def log_mean_exp(x, dim):
    """
    Compute the log(mean(exp(x), dim)) in a numerically stable manner

    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which mean is computed

    Return:
        _: tensor: (...): log(mean(exp(x), dim))
    """
    return log_sum_exp(x, dim) - np.log(x.size(dim))


def log_sum_exp(x, dim=0):
    """
    Compute the log(sum(exp(x), dim)) in a numerically stable manner

    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which sum is computed

    Return:
        _: tensor: (...): log(sum(exp(x), dim))
    """
    max_x = torch.max(x, dim)[0]
    new_x = x - max_x.unsqueeze(dim).expand_as(x)
    return max_x + (new_x.exp().sum(dim)).log()


def load_model_by_name(model, global_step):
    """
    Load a model based on its name model.name and the checkpoint iteration step

    Args:
        model: Model: (): A model
        global_step: int: (): Checkpoint iteration
    """
    file_path = os.path.join('checkpoints',
                             model.name,
                             'model-{:05d}.pt'.format(global_step))
    state = torch.load(file_path)
    model.load_state_dict(state)
    print("Loaded from {}".format(file_path))


def evaluate_lower_bound(model, eval_set, run_iwae=False):
    check_model = isinstance(model, VAE) or isinstance(model, GMVAE)
    assert check_model, "This function is only intended for VAE and GMVAE"

    print('*' * 80)
    print("LOG-LIKELIHOOD LOWER BOUNDS ON TEST SUBSET")
    print('*' * 80)

    x = eval_set
    torch.manual_seed(0)

    def detach_torch_tuple(args):
        return (v.detach() for v in args)

    def compute_metrics(fn, repeat):
        metrics = [0, 0, 0, 0, 0]
        for _ in range(repeat):
            niwae, kl, rec, rec_mse, rec_var = detach_torch_tuple(fn(x))
            metrics[0] += niwae / repeat
            metrics[1] += kl / repeat
            metrics[2] += rec / repeat
            metrics[3] += rec_mse / repeat
            metrics[4] += rec_var / repeat
        return metrics

    # Run multiple times to get low-var estimate
    nelbo, kl, rec, rec_mse, rec_var = compute_metrics(model.negative_elbo_bound, 100)
    print("NELBO: {}. KL: {}. Rec: {}. Rec_mse: {}. Rec_var: {}".format(nelbo, kl, rec, rec_mse, rec_var))

    if run_iwae:
        for iw in [1, 10, 100]:
            repeat = max(100 // iw, 1)  # Do at least 100 iterations
            fn = lambda x: model.negative_iwae_bound(x, iw)
            niwae, kl, rec, rec_mse, rec_var = compute_metrics(fn, repeat)
            print("Negative IWAE-{}: {}".format(iw, niwae))


def save_model_by_name(model, global_step):
    save_dir = os.path.join('checkpoints', model.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
    state = model.state_dict()
    torch.save(state, file_path)
    print('Saved to {}'.format(file_path))


def prepare_writer(model_name, overwrite_existing=False):
    log_dir = os.path.join('logs', model_name)
    save_dir = os.path.join('checkpoints', model_name)
    if overwrite_existing:
        delete_existing(log_dir)
        delete_existing(save_dir)
    # Sadly, I've been told *not* to use tensorflow :<
    # writer = tf.summary.FileWriter(log_dir)
    writer = None
    return writer


def log_summaries(writer, summaries, global_step):
    pass # Sad :<
    # for tag in summaries:
    #     val = summaries[tag]
    #     tf_summary = tf.Summary.Value(tag=tag, simple_value=val)
    #     writer.add_summary(tf.Summary(value=[tf_summary]), global_step)
    # writer.flush()


def delete_existing(path):
    if os.path.exists(path):
        print("Deleting existing path: {}".format(path))
        shutil.rmtree(path)


def reset_weights(m):
    try:
        m.reset_parameters()
    except AttributeError:
        pass


def get_load_data(device, batch_size, in_memory=False):
    train_loader = torch.utils.data.DataLoader(
        HourlyLoad2017Dataset(root_dir="data/split", mode='train', in_memory=in_memory),
        batch_size=batch_size,
        shuffle=True)

    val = HourlyLoad2017Dataset(root_dir="data/split", mode='val', in_memory=in_memory)
    test = HourlyLoad2017Dataset(root_dir="data/split", mode='test', in_memory=in_memory)

    if in_memory:
        val_set = val.use
        test_set = test.use
    else:
        val_set = []
        test_set = []
        for i in range(len(val)//10):
            val_set.append(val[i]["use"])
        for i in range(len(test)//10):
            test_set.append(test[i]["use"])

    val_set = torch.tensor(val_set, dtype=torch.float).to(device)
    test_set = torch.tensor(test_set, dtype=torch.float).to(device)

    return train_loader, val_set, test_set

class FixedSeed:
    def __init__(self, seed):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        np.random.set_state(self.state)
