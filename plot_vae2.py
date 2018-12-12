import torch
from matplotlib import pyplot as plt
import numpy as np
# from plot_vae import reverse_log_norm



def reverse_log_norm(x, shift_scale, log_normal):
    z = shift_scale[1]*x + shift_scale[0]
    if log_normal:
        z = np.exp(z) - 1e-5
    return z


def make_image_load(model, shift_scale, log_car):
    # print(log_car)
    model.eval()
    num_sample = 2*4
    # meta = torch.tensor([[61.7, 96.9, 0.00360, 3.00]]).repeat(num_sample, 1)
    meta_weather = np.random.normal(size=(num_sample, 3))
    meta_day = np.zeros((num_sample, 7))
    random_day = np.random.randint(0, high=7, size=(num_sample))
    meta_day[np.arange(num_sample), random_day] = 1
    meta = np.concatenate((meta_weather, meta_day), axis=1)
    # print('meta shape', meta.shape)

    # sample_z = model.sample_z(batch=num_sample)
    # sampled_mu, sampled_var = model.sample_x_given(z=sample_z)
    sampled_mu, sampled_var = model.sample_x(batch=num_sample, y = meta)

    for i in range(num_sample):
        mu = sampled_mu[i].detach().numpy()
        std = np.sqrt(sampled_var[i].detach().numpy())

        ax = plt.subplot(2, 4, i + 1)
        # ax.axis('off')
        plt.tight_layout()
        # ax.set_title('Sample #{}'.format(i+1))
        plt.ylim((0, 5))
        # ax.set_ylim(bottom=0)
        plt.fill_between(np.arange(model.x_dim),
                         reverse_log_norm(mu - 2*std, shift_scale, log_car),
                         reverse_log_norm(mu + 2*std, shift_scale, log_car),
                         alpha=0.2, facecolor='b', linewidth=0)
        plt.fill_between(np.arange(model.x_dim),
                         reverse_log_norm(mu - 1*std, shift_scale, log_car),
                         reverse_log_norm(mu + 1*std, shift_scale, log_car),
                         alpha=0.2, facecolor='b', linewidth=0)
        plt.plot(reverse_log_norm(mu, shift_scale, log_car), 'b')
    plt.savefig("checkpoints/random_samples_{}.png".format(model.name), dpi=300)
    # plt.suptitle("Random samples from Model ({})".format(model.name))
    # plt.show()


def make_image_load_day(model, shift_scale, log_car):
    # print(log_car)
    model.eval()
    num_sample = 4
    sample_z = model.sample_z(batch=num_sample)
    meta_weather = np.random.normal(size=(num_sample, 3))
    # sample_z = torch.zeros(num_sample, model.z_dim)
    # meta_weather = torch.zeros((num_sample, 3))
    for day in range(7):
        print(day)
        meta_day = np.zeros((num_sample, 7))
        meta_day[:, day] = 1
        meta = np.concatenate((meta_weather, meta_day), axis=1)
        # print(meta)

        sampled_mu, sampled_var = model.sample_x_given(z=sample_z, y=meta)

        for i in range(num_sample):
            mu = sampled_mu[i].detach().numpy()
            std = np.sqrt(sampled_var[i].detach().numpy())

            ax = plt.subplot(7, num_sample, day*num_sample + i + 1)
            # ax.axis('off')
            plt.tight_layout()
            # ax.set_title('Sample #{}'.format(i+1))
            plt.ylim((0, 5))
            # ax.set_ylim(bottom=0)
            plt.fill_between(np.arange(model.x_dim),
                             reverse_log_norm(mu - 2*std, shift_scale, log_car),
                             reverse_log_norm(mu + 2*std, shift_scale, log_car),
                             alpha=0.2, facecolor='b', linewidth=0)
            plt.fill_between(np.arange(model.x_dim),
                             reverse_log_norm(mu - 1*std, shift_scale, log_car),
                             reverse_log_norm(mu + 1*std, shift_scale, log_car),
                             alpha=0.2, facecolor='b', linewidth=0)
            plt.plot(reverse_log_norm(mu, shift_scale, log_car), 'b', alpha=1)
    plt.savefig("checkpoints/samples_day_{}.png".format(model.name), dpi=300)
    # plt.suptitle("Random samples for days 0-6 ({})".format(model.name))
    # plt.show()

# Write up section on data/data cleaning
# Generate plots from before/after cleaning
# Plots conditioned on metadata
#   


def make_image_load_z(model, shift_scale, log_car, meta=None):
    model.eval()
    z_dim = model.z_dim
    z_values = [-3, -1, 1, 3]
    num_sample = z_dim*len(z_values)

    # meta = torch.tensor([[61.7, 96.9, 0.00360, 3.00]]).repeat(num_sample, 1)

    meta_weather = np.zeros((num_sample, 3))
    meta_day = np.zeros((num_sample, 7))
    meta_day[np.arange(num_sample), 0] = 1
    meta = np.concatenate((meta_weather, meta_day), axis=1)

    # print(meta)
    z = torch.zeros(num_sample, z_dim)
    for z_i in range(z_dim):
        for v_i, value in enumerate(z_values):
            z[z_i*len(z_values) + v_i, z_i] = value

    sampled_mu, sampled_var = model.sample_x_given(z=z, y=meta)

    #print('smu, svar', sampled_mu, sampled_var)
    for i in range(num_sample):
        mu = sampled_mu[i].detach().numpy()
        std = np.sqrt(sampled_var[i].detach().numpy())

        ax = plt.subplot(z_dim, len(z_values), i + 1)
        # ax.axis('off')
        # plt.tight_layout()
        # ax.set_title('z = {}'.format(z[i, :].numpy()))
        plt.ylim((0, 5))
        plt.fill_between(np.arange(model.x_dim),
                         reverse_log_norm(mu - 2 * std, shift_scale, log_car),
                         reverse_log_norm(mu + 2 * std, shift_scale, log_car),
                         alpha=0.2, facecolor='r', linewidth=0)
        plt.fill_between(np.arange(model.x_dim),
                         reverse_log_norm(mu - 1 * std, shift_scale, log_car),
                         reverse_log_norm(mu + 1 * std, shift_scale, log_car),
                         alpha=0.2, facecolor='r', linewidth=0)
        plt.plot(reverse_log_norm(mu, shift_scale, log_car), 'r')

    plt.savefig("checkpoints/latent_{}.png".format(model.name), dpi=300)
    # plt.suptitle("Latent space samples from model {}; blue: y=0, red: y=1".format(model.name))
    # plt.show()


def make_image_load_z_use(model, shift_scale, log_car):
    model.eval()
    c_dim = model.use_model.z_dim
    z_values = [-5, 5]
    num_sample = c_dim*len(z_values)
    z = torch.zeros(num_sample, model.z_dim)

    meta_weather = np.zeros((num_sample, 3))
    meta_day = np.zeros((num_sample, 7))
    meta_day[np.arange(num_sample), 0] = 1
    meta = np.concatenate((meta_weather, meta_day), axis=1)

    c = torch.zeros(num_sample, c_dim)
    for z_i in range(c_dim):
        for v_i, value in enumerate(z_values):
            c[z_i*len(z_values) + v_i, z_i] = value

    sampled_mu, sampled_var = model.sample_x_given(z=z, y=meta, c=c)

    for i in range(num_sample):
        mu = sampled_mu[i].detach().numpy()
        std = np.sqrt(sampled_var[i].detach().numpy())

        ax = plt.subplot(c_dim, len(z_values), i + 1)
        # ax.axis('off')
        # plt.tight_layout()
        # ax.set_title('z = {}'.format(z[i, :].numpy()))
        plt.ylim((0, 0.5))
        plt.fill_between(np.arange(model.x_dim),
                         reverse_log_norm(mu - 2 * std, shift_scale, log_car),
                         reverse_log_norm(mu + 2 * std, shift_scale, log_car),
                         alpha=0.2, facecolor='r', linewidth=0)
        plt.fill_between(np.arange(model.x_dim),
                         reverse_log_norm(mu - 1 * std, shift_scale, log_car),
                         reverse_log_norm(mu + 1 * std, shift_scale, log_car),
                         alpha=0.2, facecolor='r', linewidth=0)
        plt.plot(reverse_log_norm(mu, shift_scale, log_car), 'r')

    plt.savefig("checkpoints/conditional_use_{}.png".format(model.name), dpi=300)
    # plt.suptitle("conditional use samples from model {};".format(model.name))
    # plt.show()


def detach_mu_var(mu, var):
    mu = mu.detach().numpy()
    std = np.sqrt(var.detach().numpy())
    return mu, std






def main():
    raise NotImplementedError


if __name__ == '__main__':
    main()


