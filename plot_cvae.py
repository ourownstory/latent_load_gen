import torch
from matplotlib import pyplot as plt
import numpy as np
from plot_vae import reverse_log_norm


def make_image_load(model, log_normal=False, shift_scale=(0, 1)):
    model.eval()
    num_sample = 2*4
    sample_z = model.sample_z(batch=num_sample)
    y_0 = torch.zeros((num_sample, model.y_dim))
    y_0[:, 0] = 1
    c_0 = torch.zeros((num_sample, model.c_dim))
    sampled_mu_0, sampled_var_0 = model.sample_x_given(z=sample_z, y=y_0, c=c_0)
    y_1 = torch.zeros((num_sample,  model.y_dim))
    y_1[:, 1] = 1
    sampled_mu_1, sampled_var_1 = model.sample_x_given(z=sample_z, y=y_1, c=c_0)

    for i in range(num_sample):
        mu_0 = sampled_mu_0[i].detach().numpy()
        std_0 = np.sqrt(sampled_var_0[i].detach().numpy())
        mu_1 = sampled_mu_1[i].detach().numpy()
        std_1 = np.sqrt(sampled_var_1[i].detach().numpy())

        ax = plt.subplot(2, 4, i + 1)
        # ax.axis('off')
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        plt.ylim((0, 5))
        # ax.set_ylim(bottom=0)
        plt.fill_between(np.arange(model.x_dim),
                         reverse_log_norm(mu_0 - 2*std_0, shift_scale, log_normal),
                         reverse_log_norm(mu_0 + 2*std_0, shift_scale, log_normal),
                         alpha=0.2, facecolor='b', linewidth=0)
        plt.fill_between(np.arange(model.x_dim),
                         reverse_log_norm(mu_1 - 2*std_1, shift_scale, log_normal),
                         reverse_log_norm(mu_1 + 2*std_1, shift_scale, log_normal),
                         alpha=0.2, facecolor='r', linewidth=0)
        plt.plot(reverse_log_norm(mu_0, shift_scale, log_normal), 'b')
        plt.plot(reverse_log_norm(mu_1, shift_scale, log_normal), 'r')

    plt.show()


def make_image_load_z(model, log_normal=False, shift_scale=(0, 1)):
    model.eval()
    z_dim = model.z_dim
    z_values = [-2, -0.5, 0.5, 2]
    num_sample = z_dim*len(z_values)
    z = torch.zeros(num_sample, z_dim)
    for v_i, value in enumerate(z_values):
        for z_i in range(z_dim):
            z[v_i*z_dim + z_i, z_i] = value

    y_0 = torch.zeros((num_sample, model.y_dim))
    y_0[:, 0] = 1
    c_0 = torch.zeros((num_sample, model.c_dim))
    sampled_mu_0, sampled_var_0 = model.sample_x_given(z=z, y=y_0, c=c_0)
    y_1 = torch.zeros((num_sample, model.y_dim))
    y_1[:, 1] = 1
    sampled_mu_1, sampled_var_1 = model.sample_x_given(z=z, y=y_1, c=c_0)

    for i in range(num_sample):
        mu_0 = sampled_mu_0[i].detach().numpy()
        std_0 = np.sqrt(sampled_var_0[i].detach().numpy())

        ax = plt.subplot(len(z_values), z_dim, i + 1)
        # ax.axis('off')
        plt.tight_layout()
        ax.set_title('z = {}'.format(z[i, :].numpy()))
        # plt.ylim((0, 5))
        plt.fill_between(np.arange(model.x_dim),
                         reverse_log_norm(mu_0 - 2 * std_0, shift_scale, log_normal),
                         reverse_log_norm(mu_0 + 2 * std_0, shift_scale, log_normal),
                         alpha=0.2, facecolor='b', linewidth=0)
        plt.plot(reverse_log_norm(mu_0, shift_scale, log_normal), 'b')
    # plt.suptitle("y=0")
    # plt.show()

    for i in range(num_sample):
        mu_1 = sampled_mu_1[i].detach().numpy()
        std_1 = np.sqrt(sampled_var_1[i].detach().numpy())

        ax = plt.subplot(len(z_values), z_dim, i + 1)
        # ax.axis('off')
        plt.tight_layout()
        ax.set_title('z = {}'.format(z[i, :].numpy()))
        # plt.ylim((0, 5))
        plt.fill_between(np.arange(model.x_dim),
                         reverse_log_norm(mu_1 - 2 * std_1, shift_scale, log_normal),
                         reverse_log_norm(mu_1 + 2 * std_1, shift_scale, log_normal),
                         alpha=0.2, facecolor='r', linewidth=0)
        plt.plot(reverse_log_norm(mu_1, shift_scale, log_normal), 'r')
    plt.suptitle("blue: y=0, red: y=1")
    plt.show()


def make_image_load_c(model, log_normal=False, shift_scale=(0, 1)):
    model.eval()
    num_sample = 4
    sample_z = model.sample_z(batch=num_sample)

    y_0 = torch.zeros((num_sample, model.y_dim))
    y_0[:, 0] = 1
    c_0 = torch.zeros((num_sample, model.c_dim))
    sampled_mu_0, sampled_var_0 = model.sample_x_given(z=sample_z, y=y_0, c=c_0)

    c_add = torch.zeros((num_sample, model.c_dim))
    c_add[:, 0] = 3
    qm_0, qv_0 = model.enc.encode(sampled_mu_0, y=y_0)
    sampled_mu_add_enc, sampled_var_add_enc = model.sample_x_given(z=qm_0, y=y_0, c=c_add)
    sampled_mu_add, sampled_var_add = model.sample_x_given(z=sample_z, y=y_0, c=c_add)

    y_1 = torch.zeros((num_sample,  model.y_dim))
    y_1[:, 1] = 1
    sampled_mu_1, sampled_var_1 = model.sample_x_given(z=sample_z, y=y_1, c=c_0)

    c_sub = torch.zeros((num_sample, model.c_dim))
    c_sub[:, 1] = 3
    qm_1, qv_1 = model.enc.encode(sampled_mu_1, y=y_1)
    sampled_mu_sub_enc, sampled_var_sub_enc = model.sample_x_given(z=qm_1, y=y_1, c=c_sub)
    sampled_mu_sub, sampled_var_sub = model.sample_x_given(z=sample_z, y=y_1, c=c_sub)

    for i in range(num_sample):
        # mu_0 = sampled_mu_0[i].detach().numpy()
        # std_0 = np.sqrt(sampled_var_0[i].detach().numpy())
        # mu_1 = sampled_mu_1[i].detach().numpy()
        # std_1 = np.sqrt(sampled_var_1[i].detach().numpy())

        mu_0, std_0 = detach_mu_var(sampled_mu_0[i], sampled_var_0[i])
        mu_1, std_1 = detach_mu_var(sampled_mu_1[i], sampled_var_1[i])
        mu_add, std_add = detach_mu_var(sampled_mu_add[i], sampled_var_add[i])
        mu_sub, std_sub = detach_mu_var(sampled_mu_sub[i], sampled_var_sub[i])
        mu_add_enc, std_add_enc = detach_mu_var(sampled_mu_add_enc[i], sampled_var_add_enc[i])
        mu_sub_enc, std_sub_enc = detach_mu_var(sampled_mu_sub_enc[i], sampled_var_sub_enc[i])

        ax = plt.subplot(4, 4, i + 1)
        plt.tight_layout()
        ax.set_title('#{} Add'.format(i))
        # plt.ylim((0, 5))
        # ax.axis('off')
        plt.fill_between(np.arange(model.x_dim),
                         reverse_log_norm(mu_0 - 2*std_0, shift_scale, log_normal),
                         reverse_log_norm(mu_0 + 2*std_0, shift_scale, log_normal),
                         alpha=0.2, facecolor='b', linewidth=0)
        plt.plot(reverse_log_norm(mu_0, shift_scale, log_normal), 'b')
        plt.fill_between(np.arange(model.x_dim),
                         reverse_log_norm(mu_add - 2*std_add, shift_scale, log_normal),
                         reverse_log_norm(mu_add + 2*std_add, shift_scale, log_normal),
                         alpha=0.2, facecolor='r', linewidth=0)
        plt.plot(reverse_log_norm(mu_add, shift_scale, log_normal), 'r')

        ax = plt.subplot(4, 4, i + 1 + num_sample)
        plt.tight_layout()
        ax.set_title('#{} Sub'.format(i))
        # plt.ylim((0, 5))
        # ax.axis('off')
        plt.fill_between(np.arange(model.x_dim),
                         reverse_log_norm(mu_1 - 2*std_1, shift_scale, log_normal),
                         reverse_log_norm(mu_1 + 2*std_1, shift_scale, log_normal),
                         alpha=0.2, facecolor='r', linewidth=0)
        plt.plot(reverse_log_norm(mu_1, shift_scale, log_normal), 'r')
        plt.fill_between(np.arange(model.x_dim),
                         reverse_log_norm(mu_sub - 2*std_sub, shift_scale, log_normal),
                         reverse_log_norm(mu_sub + 2*std_sub, shift_scale, log_normal),
                         alpha=0.2, facecolor='b', linewidth=0)
        plt.plot(reverse_log_norm(mu_sub, shift_scale, log_normal), 'b')

        ax = plt.subplot(4, 4, i + 1 + 2*num_sample)
        plt.tight_layout()
        ax.set_title('#{} Add (Enc)'.format(i))
        # plt.ylim((0, 5))
        # ax.axis('off')
        plt.fill_between(np.arange(model.x_dim),
                         reverse_log_norm(mu_0 - 2*std_0, shift_scale, log_normal),
                         reverse_log_norm(mu_0 + 2*std_0, shift_scale, log_normal),
                         alpha=0.2, facecolor='b', linewidth=0)
        plt.plot(reverse_log_norm(mu_0, shift_scale, log_normal), 'b')
        plt.fill_between(np.arange(model.x_dim),
                         reverse_log_norm(mu_add_enc - 2*std_add_enc, shift_scale, log_normal),
                         reverse_log_norm(mu_add_enc + 2*std_add_enc, shift_scale, log_normal),
                         alpha=0.2, facecolor='r', linewidth=0)
        plt.plot(reverse_log_norm(mu_add_enc, shift_scale, log_normal), 'r')

        ax = plt.subplot(4, 4, i + 1 + 3*num_sample)
        plt.tight_layout()
        ax.set_title('#{} Sub (Enc)'.format(i))
        # plt.ylim((0, 5))
        # ax.axis('off')
        plt.fill_between(np.arange(model.x_dim),
                         reverse_log_norm(mu_1 - 2*std_1, shift_scale, log_normal),
                         reverse_log_norm(mu_1 + 2*std_1, shift_scale, log_normal),
                         alpha=0.2, facecolor='r', linewidth=0)
        plt.plot(reverse_log_norm(mu_1, shift_scale, log_normal), 'r')
        plt.fill_between(np.arange(model.x_dim),
                         reverse_log_norm(mu_sub_enc - 2*std_sub_enc, shift_scale, log_normal),
                         reverse_log_norm(mu_sub_enc + 2*std_sub_enc, shift_scale, log_normal),
                         alpha=0.2, facecolor='b', linewidth=0)
        plt.plot(reverse_log_norm(mu_sub_enc, shift_scale, log_normal), 'b')

    plt.show()


def detach_mu_var(mu, var):
    mu = mu.detach().numpy()
    std = np.sqrt(var.detach().numpy())
    return mu, std


def main():
    raise NotImplementedError


if __name__ == '__main__':
    main()


