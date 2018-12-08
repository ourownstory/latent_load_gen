import torch
from matplotlib import pyplot as plt
import numpy as np
from plot_vae import reverse_log_norm
from codebase.utils import get_shift_scale


def make_image_load(model, shift_scale):
    model.eval()
    num_sample = 2*4
    # sample_z = model.sample_z(batch=num_sample)
    # sampled_mu, sampled_var = model.sample_x_given(z=sample_z)
    sampled_mu, sampled_var = model.sample_x(batch=num_sample)

    for i in range(num_sample):
        mu = sampled_mu[i].detach().numpy()
        std = np.sqrt(sampled_var[i].detach().numpy())

        ax = plt.subplot(2, 4, i + 1)
        # ax.axis('off')
        plt.tight_layout()
        # ax.set_title('Sample #{}'.format(i+1))
        # plt.ylim((0, 5))
        # ax.set_ylim(bottom=0)
        plt.fill_between(np.arange(model.x_dim),
                         reverse_log_norm(mu - 2*std, shift_scale),
                         reverse_log_norm(mu + 2*std, shift_scale),
                         alpha=0.2, facecolor='b', linewidth=0)
        plt.plot(reverse_log_norm(mu, shift_scale), 'b')
    # plt.savefig("checkpoints/random_samples_{}.png".format(model.name), dpi=300)
    plt.suptitle("Random samples from Model ({})".format(model.name))
    plt.show()


def make_image_load_z(model, shift_scale):
    model.eval()
    z_dim = model.z_dim
    z_values = [-2, 2]
    num_sample = z_dim*len(z_values)
    z = torch.zeros(num_sample, z_dim)
    for z_i in range(z_dim):
        for v_i, value in enumerate(z_values):
            z[z_i*len(z_values) + v_i, z_i] = value

    sampled_mu, sampled_var = model.sample_x_given(z=z)

    for i in range(num_sample):
        mu = sampled_mu[i].detach().numpy()
        std = np.sqrt(sampled_var[i].detach().numpy())

        ax = plt.subplot(z_dim, len(z_values), i + 1)
        # ax.axis('off')
        # plt.tight_layout()
        # ax.set_title('z = {}'.format(z[i, :].numpy()))
        # plt.ylim((0, 5))
        plt.fill_between(np.arange(model.x_dim),
                         reverse_log_norm(mu - 2 * std, shift_scale),
                         reverse_log_norm(mu + 2 * std, shift_scale),
                         alpha=0.2, facecolor='r', linewidth=0)
        plt.plot(reverse_log_norm(mu, shift_scale), 'r')

    # plt.savefig("checkpoints/latent_{}.png".format(model.name), dpi=300)
    plt.suptitle("Latent space samples from model {}; blue: y=0, red: y=1".format(model.name))
    plt.show()


def make_image_load_z_use(model, shift_scale):
    model.eval()
    c_dim = model.use_model.z_dim
    z_values = [-2, 2]
    num_sample = c_dim*len(z_values)
    z = torch.ones(num_sample, model.z_dim)

    c = torch.zeros(num_sample, c_dim)
    for z_i in range(c_dim):
        for v_i, value in enumerate(z_values):
            c[z_i*len(z_values) + v_i, z_i] = value

    sampled_mu, sampled_var = model.sample_x_given(z=z, c=c)

    for i in range(num_sample):
        mu = sampled_mu[i].detach().numpy()
        std = np.sqrt(sampled_var[i].detach().numpy())

        ax = plt.subplot(c_dim, len(z_values), i + 1)
        # ax.axis('off')
        # plt.tight_layout()
        # ax.set_title('z = {}'.format(z[i, :].numpy()))
        # plt.ylim((0, 5))
        plt.fill_between(np.arange(model.x_dim),
                         reverse_log_norm(mu - 2 * std, shift_scale),
                         reverse_log_norm(mu + 2 * std, shift_scale),
                         alpha=0.2, facecolor='r', linewidth=0)
        plt.plot(reverse_log_norm(mu, shift_scale), 'r')

    # plt.savefig("checkpoints/latent_{}.png".format(model.name), dpi=300)
    plt.suptitle("Latent space samples from model {}; blue: y=0, red: y=1".format(model.name))
    plt.show()




def detach_mu_var(mu, var):
    mu = mu.detach().numpy()
    std = np.sqrt(var.detach().numpy())
    return mu, std


def main():
    raise NotImplementedError


if __name__ == '__main__':
    main()


