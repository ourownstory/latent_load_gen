import torch
from matplotlib import pyplot as plt
import numpy as np


def reverse_log_norm(x, shift_scale, log_normal=False):
    z = shift_scale[1]*x + shift_scale[0]
    if log_normal:
        z = np.exp(z)
    return z


def make_image_load(model, log_normal=False, shift_scale=(0, 1)):
    num_sample = 4*4
    sampled_mu, sampled_var = model.eval().sample_x(batch=num_sample)
    for i in range(num_sample):
        mu = sampled_mu[i].detach().numpy()
        std = np.sqrt(sampled_var[i].detach().numpy())

        ax = plt.subplot(4, 4, i + 1)
        # ax.axis('off')
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        plt.ylim((0, 5))
        # ax.set_ylim(bottom=0)
        plt.fill_between(np.arange(model.x_dim),
                         reverse_log_norm(mu - 2*std, shift_scale, log_normal),
                         reverse_log_norm(mu + 2*std, shift_scale, log_normal),
                         alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF', linewidth=0)
        plt.plot(reverse_log_norm(mu, shift_scale, log_normal))

    plt.show()


def make_image_load_z(model, log_normal=False, shift_scale=(0, 1)):
    z_dim = model.z_dim
    z_values = [-2, -0.5, 0.5, 2]
    num_sample = z_dim*len(z_values)
    z = torch.zeros(num_sample, z_dim)
    for v_i, value in enumerate(z_values):
        for z_i in range(z_dim):
            z[v_i*z_dim + z_i, z_i] = value
    # print(z)

    sampled_mu, sampled_var = model.eval().sample_x_given(z=z)
    for i in range(num_sample):
        mu = sampled_mu[i].detach().numpy()
        std = np.sqrt(sampled_var[i].detach().numpy())

        ax = plt.subplot(len(z_values), z_dim, i + 1)
        # ax.axis('off')
        plt.tight_layout()
        ax.set_title('z = {}'.format(z[i, :].numpy()))
        # plt.ylim((0, 5))
        plt.fill_between(np.arange(model.x_dim),
                         reverse_log_norm(mu - 2*std, shift_scale, log_normal),
                         reverse_log_norm(mu + 2*std, shift_scale, log_normal),
                         alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF', linewidth=0)
        plt.plot(reverse_log_norm(mu, shift_scale, log_normal))

    plt.show()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='gmvae', help="run VAE or GMVAE")
    parser.add_argument('--z', type=int, default=4, help="Number of latent dimensions")
    parser.add_argument('--iter_max', type=int, default=10000, help="Number of training iterations")
    # parser.add_argument('--iter_save', type=int, default=1000,  help="Save model every n iterations")
    parser.add_argument('--run', type=int, default=0, help="Run ID. In case you want to run replicates")
    parser.add_argument('--train', type=int, default=0, help="Flag for training")
    # parser.add_argument('--batch',     type=int, default=100,    help="Batch size")
    parser.add_argument('--k', type=int, default=16, help="Number mixture components in MoG prior")
    parser.add_argument('--warmup', type=int, default=1, help="Fix variance during first 1/4 of training")
    args = parser.parse_args()
    layout = [
        ('model={:s}', args.model),
        ('z={:02d}', args.z),
        ('run={:04d}', args.run)
    ]
    model_name = '_'.join([t.format(v) for (t, v) in layout])
    pprint(vars(args))
    print('Model name:', model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == 'gmvae':
        model = GMVAE(z_dim=args.z, name=model_name, x_dim=24, warmup=args.warmup, k=args.k).to(device)
    else:
        model = VAE(z_dim=args.z, name=model_name, x_dim=24, warmup=args.warmup).to(device)

    ut.load_model_by_name(model, global_step=args.iter_max)

    # make_image_load(model)
    make_image_load_z(model)


if __name__ == '__main__':
    from codebase import utils as ut
    from codebase.models.vae import VAE, GMVAE

    from pprint import pprint
    import argparse

    main()


