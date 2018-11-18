from codebase import utils as ut
from codebase.models.vae import VAE
from codebase.models.gmvae import GMVAE

from pprint import pprint
import argparse
import torch
from matplotlib import pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model',     type=str, default='gmvae', help="run VAE or GMVAE")
parser.add_argument('--z',         type=int, default=5,     help="Number of latent dimensions")
parser.add_argument('--iter_max',  type=int, default=10000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=1000,  help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',     type=int, default=1,     help="Flag for training")
parser.add_argument('--batch',     type=int, default=100,    help="Batch size")
parser.add_argument('--k',         type=int, default=100,    help="Number mixture components in MoG prior")
args = parser.parse_args()
layout = [
    ('model={:s}',  args.model),
    ('z={:02d}',  args.z),
    ('run={:04d}', args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.model == 'gmvae':
    model = GMVAE(z_dim=args.z, name=model_name, x_dim=24, warmup=True, k=args.k).to(device)
else:
    model = VAE(z_dim=args.z, name=model_name, x_dim=24, warmup=True).to(device)

ut.load_model_by_name(model, global_step=args.iter_max)


def make_image_load(model):
    num_sample = 4*4
    sampled_mu, sampled_var = model.eval().sample_x(batch=num_sample)
    for i in range(num_sample):
        mu = sampled_mu[i].detach().numpy()
        std = np.sqrt(sampled_var[i].detach().numpy())

        ax = plt.subplot(4, 4, i + 1)
        # ax.axis('off')
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        # plt.ylim((0, 6))
        plt.fill_between(np.arange(24), mu - std, mu + std,
                         alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF', linewidth=0)
        plt.plot(mu)

    plt.show()


make_image_load(model)
