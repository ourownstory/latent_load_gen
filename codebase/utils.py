import numpy as np
import os
import shutil
import torch
from codebase.models.vae import VAE, GMVAE
from codebase.models.cvae import CVAE
from codebase.models.vae2 import VAE2
from torch.nn import functional as F
from HourlyLoadDataset import HourlyLoad2017Dataset
from ConditionalLoadDataset import ConditionalLoadDataset
# from LoadDataset2 import LoadDataset2
import pandas as pd


log_2pi = np.log(2*np.pi)


def nlog_prob_normal(mu, y, var=None, fixed_var=False, var_pen=1):
    # print(mu.shape, y.shape, var.shape)
    diff = y - mu
    # makes it just MSE
    mse = torch.mul(diff, diff)
    if fixed_var:
        var_cost = torch.zeros(*y.shape)
    else:
        # return actual log-likelihood
        mse = torch.div(mse, (var + 1e-5))  #* (1.0/var_pen)
        var_cost = torch.log(var*var_pen + 1)  #* var_pen
        # var_cost = torch.mul(var, var)
    mse = mse.sum(-1)
    var_cost = var_cost.sum(-1)
    cost = mse + var_cost
    # these two last terms make it a correct log(p), but are not required for MLE
    # cost += log_2pi * y.size()[-1]
    # cost *= 0.5
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


def evaluate_lower_bound_conditional(model, eval_set, run_iwae=False):
    check_model = isinstance(model, CVAE)
    assert check_model, "This function is only intended for CVAE and GMCVAE"

    print('*' * 80)
    print("LOG-LIKELIHOOD LOWER BOUNDS ON TEST SUBSET")
    print('*' * 80)

    x_inputs = eval_set
    torch.manual_seed(0)

    def detach_torch_tuple(args):
        return (v.detach() for v in args)

    def compute_metrics(repeat):
        metrics = [0, 0, 0, 0, 0]
        for _ in range(repeat):
            niwae, kl, rec, rec_mse, rec_var = detach_torch_tuple(
                model.compute_nelbo_niwae(**x_inputs)
            )
            metrics[0] += niwae / repeat
            metrics[1] += kl / repeat
            metrics[2] += rec / repeat
            metrics[3] += rec_mse / repeat
            metrics[4] += rec_var / repeat
        return metrics

    # Run multiple times to get low-var estimate
    nelbo, kl, rec, rec_mse, rec_var = compute_metrics(100)
    print("NELBO: {}. KL: {}. Rec: {}. Rec_mse: {}. Rec_var: {}".format(
        nelbo, kl, rec, rec_mse, rec_var))

    if run_iwae:
        for iw in [1, 10, 100]:
            repeat = max(100 // iw, 1)  # Do at least 100 iterations
            # fn = lambda x: model.compute_nelbo_niwae(x_inputs, iw=iw)
            x_inputs['iw'] = iw
            niwae, kl, rec, rec_mse, rec_var = compute_metrics(repeat)
            print("Negative IWAE-{}: {}. KL: {}. Rec: {}. Rec_mse: {}. Rec_var: {}".format(
                iw, niwae, kl, rec, rec_mse, rec_var))


def evaluate_lower_bound2(model, eval_set, run_iwae=False, mode='val', verbose=True, repeats=10, summaries=None):
    check_model = isinstance(model, VAE2)
    assert check_model, "This function is only intended for VAE2 and GMVAE2"

    if verbose:
        print('*' * 80)
        print("LOG-LIKELIHOOD LOWER BOUNDS ON {} SUBSET".format(mode))
        print('*' * 80)

    x_inputs = eval_set
    x_inputs['iw'] = 0
    torch.manual_seed(0)

    def detach_torch_tuple(args):
        return (v.detach() for v in args)

    def compute_metrics(repeat):
        metrics = [0, 0, 0, 0, 0]
        for _ in range(repeat):
            niwae, kl, rec, rec_mse, rec_var = detach_torch_tuple(
                model.nelbo_niwae(**x_inputs)
            )
            metrics[0] += niwae / repeat
            metrics[1] += kl / repeat
            metrics[2] += rec / repeat
            metrics[3] += rec_mse / repeat
            metrics[4] += rec_var / repeat
        return metrics

    # Run multiple times to get low-var estimate
    nelbo, kl, rec, rec_mse, rec_var = compute_metrics(repeats)
    if summaries is not None:
        summaries["loss_type"] = 0
        summaries["loss"], summaries["kl_z"], summaries["rec_mse"], summaries["rec_var"] = nelbo, kl, rec_mse, rec_var
        log_summaries(summaries, mode, model.name, verbose=False)
    print("{}-NELBO: {}. KL: {}. Rec: {}. Rec_mse: {}. Rec_var: {}".format(
        mode, nelbo, kl, rec, rec_mse, rec_var))

    if run_iwae:
        for iw in [1, 4, 10]:
            repeat = max(repeats // (iw*iw), 1)  # Do at least 10 iterations
            x_inputs['iw'] = iw
            niwae, kl, rec, rec_mse, rec_var = compute_metrics(repeat)
            if summaries is not None:
                summaries["loss_type"] = iw
                summaries["loss"], summaries["kl_z"], summaries["rec_mse"], summaries["rec_var"] \
                    = niwae, kl, rec_mse, rec_var
                log_summaries(summaries, mode, model.name, verbose=False)
            print("{}-Negative IWAE-{}: {}. KL: {}. Rec: {}. Rec_mse: {}. Rec_var: {}".format(
                mode, iw, niwae, kl, rec, rec_mse, rec_var))


def save_model_by_name(model, global_step):
    save_dir = os.path.join('checkpoints', model.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
    state = model.state_dict()
    torch.save(state, file_path)
    print('Saved to {}'.format(file_path))


def prepare_writer(model_name, overwrite_existing=False):
    # log_dir = os.path.join('logs', model_name)
    save_dir = os.path.join('checkpoints', model_name)
    if overwrite_existing:
        # delete_existing(log_dir)
        delete_existing(save_dir)
    # Sadly, I've been told *not* to use tensorflow :<
    # writer = tf.summary.FileWriter(log_dir)
    writer = None
    return writer


def log_summaries(summaries, mode, model_name, verbose=False):
    if verbose:
        print(", ".join(["{}: {:.4f}".format(key, v) for key, v in summaries.items()]))
    save_dir = os.path.join('checkpoints', model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'summaries_{}.csv'.format(mode))
    if os.path.isfile(file_path):
        with open(file_path, 'a') as file:
            file.write(",".join(["{:.4f}".format(value) for key, value in summaries.items()]) + "\n")
    else:
        # if verbose: print("creating new summaries CSV")
        with open(file_path, 'w') as file:
            file.write(",".join([key for key, value in summaries.items()]) + "\n"
                       + ",".join(["{:.4f}".format(value) for key, value in summaries.items()]) + "\n")


def delete_existing(path):
    if os.path.exists(path):
        print("Deleting existing path: {}".format(path))
        shutil.rmtree(path)


def reset_weights(m):
    try:
        m.reset_parameters()
    except AttributeError:
        pass


def get_load_data(device, split, batch_size=128, in_memory=False, log_normal=False, shift_scale=None):
    if split == 'train':
        train_loader = torch.utils.data.DataLoader(
            HourlyLoad2017Dataset(root_dir="data/split", mode='train',
                                  in_memory=in_memory, log_normal=log_normal, shift_scale=shift_scale),
            batch_size=batch_size,
            shuffle=True)
        return train_loader
    else:
        split_set = HourlyLoad2017Dataset(root_dir="data/split", mode=split,
                                          in_memory=in_memory, log_normal=log_normal, shift_scale=shift_scale)
        if in_memory:
            split_list = split_set.use
        else:
            split_list = []
            for i in range(len(split_set)):
                split_list.append(split_set[i]["use"])
        split_set = torch.tensor(split_list, dtype=torch.float).to(device)
        return split_set


def get_load_data_conditional(device, split, batch_size=64, in_memory=False, log_normal=False, get_hourly=True):
    if get_hourly:
        root_dir = "data/split"
    else:
        root_dir = "../data/CS236"

    shift_scale = get_shift_scale(get_hourly, log_normal)

    if split == 'train':
        train_loader = torch.utils.data.DataLoader(
            ConditionalLoadDataset(
                root_dir=root_dir, mode='train',
                in_memory=in_memory, log_normal=log_normal, shift_scale=shift_scale,
                get_ev_subset=False,
            ),
            batch_size=batch_size,
            shuffle=True
        )
        train_loader_ev = torch.utils.data.DataLoader(
            ConditionalLoadDataset(
                root_dir=root_dir, mode='train',
                in_memory=in_memory, log_normal=log_normal, shift_scale=shift_scale,
                get_ev_subset=True,
            ),
            batch_size=batch_size,
            shuffle=True
        )
        return train_loader, train_loader_ev, shift_scale
    else:
        split_set_ev = ConditionalLoadDataset(
            root_dir=root_dir, mode=split, in_memory=in_memory, log_normal=log_normal, shift_scale=shift_scale,
            get_ev_subset=True,
        )
        if in_memory:
            split_set = {
                "x_0": torch.FloatTensor(split_set_ev.x_0).to(device),
                "x_1": torch.FloatTensor(split_set_ev.x_1).to(device),
                "y_real": torch.FloatTensor(split_set_ev.y_real).to(device),
            }
        else:
            raise NotImplementedError
        return split_set, shift_scale


def get_shift_scale(hourly, log_normal):
    if hourly:
        if log_normal:
            # computed on train subset (for all x), log-transformed
            shift_scale = (-0.5223688943269471, 2.6144099155163927)
        else:
            # computed on train subset (for all x), without log
            shift_scale = (1.5625197109379185, 1.751249676924145)
    else:
        if log_normal:
            # computed on train, log-transformed, for 96resolution
            shift_scale = (-1.75972004287765, 4.078388702012023)
        else:
            # computed on train, without log, for 96resolution
            shift_scale = (1.2456642410923986, 1.6478924381994144)
    return shift_scale


def save_latent(model, val_set, mode, verbose=False, is_car_model=True):
    if is_car_model:
        z_use_mu, z_use_var = model.use_model.enc.encode(
            x=val_set['c'],
            y=val_set['y'],
        )
        c = z_use_mu
    else:
        c = val_set['c']
    z_mu, z_var = model.enc.encode(x=val_set['x'], y=val_set['y'], c=c)

    df_mu = pd.DataFrame(
        data=z_mu.detach().numpy(),
        columns=["mu-{:02d}".format(i + 1)for i in range(model.z_dim)],
    )
    df_var = pd.DataFrame(
        data=z_var.detach().numpy(),
        columns=["var-{:02d}".format(i + 1)for i in range(model.z_dim)],
    )
    # for i in range(model.z_dim):
    #     print(i)
    #     df["mu-{:02d}".format(i+1)] = z_mu[:, i]
    # for i in range(model.z_dim):
    #     df["var-{:02d}".format(i+1)] = z_var[:, i]

    save_dir = os.path.join('checkpoints', model.name)
    file_path = os.path.join(save_dir, '{}_latent_mu.csv'.format(mode))
    if verbose: print("saving latent: ", file_path)
    df_mu.to_csv(file_path, index=False)
    file_path = os.path.join(save_dir, '{}_latent_var.csv'.format(mode))
    if verbose: print("saving latent: ", file_path)
    df_var.to_csv(file_path, index=False)
