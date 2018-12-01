import argparse
import torch
import tqdm
from codebase import utils as ut
from codebase.models.vae2 import GMVAE2, VAE2
from codebase.train import train2
from pprint import pprint
from plot_vae2 import make_image_load, make_image_load_z, make_image_load_c
from LoadDataset2 import LoadDataset2


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model',     type=str, default='v1', help="model_architecture: v1, lstm")
parser.add_argument('--z',         type=int, default=5,     help="Number of latent dimensions")
parser.add_argument('--iter_max',  type=int, default=10000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=1000, help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--mode',      type=str, default='train',help="Flag for train, val, test, plot")
parser.add_argument('--batch',     type=int, default=64,   help="Batch size")
parser.add_argument('--lr',        type=float, default=1e-3,help="Learning Rate(initial)")
parser.add_argument('--warmup',    type=int, default=0,     help="Fix variance during first 1/4 of training")
parser.add_argument('--var_pen',   type=int, default=3,     help="Penalty for variance - multiplied with var loss term")
parser.add_argument('--lr_gamma',  type=float, default=0.5, help="Anneling factor of lr")
parser.add_argument('--lr_m_num',  type=int, default=3,     help="Number of lr anneling milestones")
parser.add_argument('--k',         type=int, default=1,    help="Number mixture components in MoG prior")
parser.add_argument('--iw',        type=int, default=0,    help="Number of IWAE samples for training")
parser.add_argument('--filter_ev',type=int, default=1,    help="remove car values where days-total is less than 0.1kWh")
parser.add_argument('--hourly',    type=int, default=0,    help="hourly data instead of 15min resolution data")
# parser.add_argument('--run_car',    type=int, default=0,    help="whether to run the second model or first")
args = parser.parse_args()

lr_milestones = [int(args.iter_max*((i+1)/(args.lr_m_num+1))) for i in range(args.lr_m_num)]
print("lr_milestones", lr_milestones, "lr", [args.lr*args.lr_gamma**i for i in range(args.lr_m_num)])

layout = [
    ('{:s}',  "gmcvae" if args.k > 1 else "cvae"),
    ('{:s}',  args.model),
    ('x{:02d}',  24 if args.hourly==1 else 96),
    ('z{:02d}',  args.z),
    ('k{:02d}',  args.k),
    ('iw{:02d}',  args.iw),
    ('vp{:02d}',  args.var_pen),
    ('lr{:.4f}',  args.lr),
    ('it{:05d}', args.iter_max),
    ('run{:02d}', args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# choose model
if args.k > 1:
    model = GMVAE2(
        nn=args.model, z_dim=args.z, name=model_name, x_dim=24 if args.hourly==1 else 96,
        warmup=(args.warmup==1), var_pen=args.var_pen, k=args.k
    ).to(device)
else:
    model = VAE2(nn=args.model, z_dim=args.z, name=model_name, x_dim=24 if args.hourly==1 else 96,
                 warmup=(args.warmup==1), var_pen=args.var_pen).to(device)

root_dir = "data/split" if args.hourly else "../data/CS236"
# load train loader anyways - to get correct shift_scale values.
train_loader = torch.utils.data.DataLoader(
    LoadDataset2(root_dir=root_dir, mode='train', shift_scale=None, filter_ev=args.filter_ev),
    batch_size=args.batch, shuffle=True
)
shift_scale = train_loader.dataset.shift_scale

if args.mode == 'train':
    writer = ut.prepare_writer(model_name, overwrite_existing=True)
    train2(
        model=model,
        train_loader=train_loader,
        device=device,
        tqdm=tqdm.tqdm,
        writer=writer,
        lr=args.lr, lr_gamma=args.lr_gamma, lr_milestones=lr_milestones,
        iw=args.iw,
        iter_max=args.iter_max, iter_save=args.iter_save
    )

else:
    ut.load_model_by_name(model, global_step=args.iter_max)

if args.mode in ['val', 'test']:
    model.set_to_eval()
    split_set = LoadDataset2(
        root_dir=root_dir, mode=args.mode, shift_scale=shift_scale, filter_ev=args.filter_ev,
    )
    val_set = {
        "car": torch.FloatTensor(split_set.car).to(device),
        "other": torch.FloatTensor(split_set.other).to(device),
        # TODO: add metadata
        "meta": None,
    }
    # TODO: re-write
    ut.evaluate_lower_bound2(model, val_set, run_iwae=(args.iw>=1))

if args.mode == 'plot':
    # make_image_load(model, log_normal=args.log_normal)
    make_image_load_z(model, log_normal=args.log_normal)
    make_image_load_c(model, log_normal=args.log_normal)
