import argparse
import torch
import tqdm
from codebase import utils as ut
from codebase.models.vae import VAE, GMVAE
from codebase.train import train
from pprint import pprint
from plot_vae import make_image_load, make_image_load_z

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model',     type=str, default='vae', help="run VAE or GMVAE")
parser.add_argument('--z',         type=int, default=4,     help="Number of latent dimensions")
parser.add_argument('--iter_max',  type=int, default=5000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=1000, help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--mode',      type=str, default='train',help="Flag for train, val, test, plot")
parser.add_argument('--batch',     type=int, default=128,   help="Batch size")
parser.add_argument('--k',         type=int, default=16,    help="Number mixture components in MoG prior")
parser.add_argument('--lr',        type=float, default=8e-4,help="Learning Rate(initial)")
parser.add_argument('--warmup',    type=int, default=1,     help="Fix variance during first 1/4 of training")
parser.add_argument('--var_pen',   type=int, default=10,     help="Penalty for variance - multiplied with var loss term")
parser.add_argument('--lr_gamma',  type=float, default=0.5, help="Anneling factor of lr")
parser.add_argument('--lr_m_num',  type=int, default=4,     help="Number of lr anneling milestones")
parser.add_argument('--iw',        type=int, default=0,    help="Number of IWAE samples for training")
args = parser.parse_args()
lr_milestones = [int(args.iter_max*((i+1)/(args.lr_m_num+1))) for i in range(args.lr_m_num)]
print("lr_milestones", lr_milestones, "lr", [args.lr*args.lr_gamma**i for i in range(args.lr_m_num)])
layout = [
    ('model={:s}',  args.model),
    ('z={:02d}',  args.z),
    ('run={:04d}', args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
shift_scale = {
    'use': (-0.32598444650515, 2.2728914107033136),
    # use use-values for all
    'car': (-0.32598444650515, 2.2728914107033136),
    'other': (-0.32598444650515, 2.2728914107033136),
    # use instead the real values:
    # 'car': (-8.742585689402953, 3.867340374856824),
    # 'other': (-0.5925285502290069, 2.732820150602578),
}  # computed on train subset, log-transformed

# choose model
if args.model == 'gmvae':
    model = GMVAE(
        z_dim=args.z, name=model_name, x_dim=24, warmup=(args.warmup==1), var_pen=args.var_pen,
        k=args.k
    ).to(device)
else:
    model = VAE(
        z_dim=args.z, name=model_name, x_dim=24, warmup=(args.warmup==1), var_pen=args.var_pen
    ).to(device)

if args.mode == 'train':
    writer = ut.prepare_writer(model_name, overwrite_existing=True)
    train_loader = ut.get_load_data(device, split='train', batch_size=args.batch,
                                    in_memory=True, log_normal=True, shift_scale=shift_scale)
    train(model=model,
          train_loader=train_loader,
          device=device,
          tqdm=tqdm.tqdm,
          writer=writer,
          lr=args.lr, lr_gamma=args.lr_gamma, lr_milestones=lr_milestones,
          iw=args.iw,
          iter_max=args.iter_max, iter_save=args.iter_save)
    model.set_to_eval()
    val_set = ut.get_load_data(device, split='val', in_memory=True, log_normal=True, shift_scale=shift_scale)
    ut.evaluate_lower_bound(model, val_set, run_iwae=(args.iw>1))
else:
    ut.load_model_by_name(model, global_step=args.iter_max)

if args.mode in ['val', 'test']:
    model.set_to_eval()
    val_set = ut.get_load_data(device, split=args.mode, in_memory=True, log_normal=True, shift_scale=shift_scale)
    ut.evaluate_lower_bound(model, val_set, run_iwae=(args.iw>1))

make_image_load(model, log_normal=True, shift_scale=shift_scale["use"])
make_image_load_z(model, log_normal=True, shift_scale=shift_scale["use"])
