import argparse
import torch
import tqdm
from codebase import utils as ut
from codebase.models.vae2 import GMVAE2, VAE2
from codebase.train import train2
from pprint import pprint
from plot_vae2 import make_image_load, make_image_load_z, make_image_load_day, make_image_load_z_use
from LoadDataset2 import LoadDataset2
from collections import OrderedDict


def run(args, verbose=False):
    layout = [
        ('{:s}',  "gmvae2" if args.k > 1 else "vae2"),
        ('{:s}',  args.model),
        ('x{:02d}',  24 if args.hourly==1 else 96),
        ('z{:02d}',  args.z),
        ('k{:02d}',  args.k),
        ('iw{:02d}',  args.iw),
        ('vp{:02d}',  args.var_pen),
        ('lr{:.4f}',  args.lr),
        ('epo{:03d}', args.num_epochs),
        ('run{:02d}', args.run),
    ]
    model_name = '_'.join([t.format(v) for (t, v) in layout])
    if verbose: pprint(vars(args))
    print('Model name:', model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # cloud
    root_dir = "../data/data15_final"
    # Oskar
    # root_dir = "../data/CS236/data60/split" if (args.hourly == 1) else "../data/CS236/data15_final"
    # Will
    #root_dir = '/Users/willlauer/Desktop/latent_load_gen/data/split'

    # load train loader anyways - to get correct shift_scale values.
    train_loader = torch.utils.data.DataLoader(
        LoadDataset2(root_dir=root_dir, mode='train', shift_scale=None, filter_ev=False, smooth=args.smooth),
        batch_size=args.batch, shuffle=True,
    )
    shift_scale = train_loader.dataset.shift_scale

    if args.k > 1:
        model = GMVAE2(
            nn=args.model, z_dim=args.z, name=model_name, x_dim=24 if args.hourly==1 else 96,
            warmup=(args.warmup==1), var_pen=args.var_pen,
            k=args.k,
            y_dim=train_loader.dataset.dim_meta,
        ).to(device)
    else:
        model = VAE2(
            nn=args.model, z_dim=args.z, name=model_name, x_dim=24 if args.hourly==1 else 96,
            y_dim=train_loader.dataset.dim_meta,
            warmup=(args.warmup==1), var_pen=args.var_pen
        ).to(device)

    if args.mode == 'train':
        split_set = LoadDataset2(
            root_dir=root_dir,
            mode='val',
            shift_scale=shift_scale,
            filter_ev=False,
            smooth=None,
        )
        val_set = {
            "x": torch.FloatTensor(split_set.other).to(device),
            "y": torch.FloatTensor(split_set.meta).to(device),
            "c": None,
        }
        # maybe in future use Tensorboard?
        _ = ut.prepare_writer(model_name, overwrite_existing=True)
        train2(
            model=model,
            train_loader=train_loader,
            val_set=val_set,
            tqdm=tqdm.tqdm,
            # writer=writer,
            lr=args.lr, lr_gamma=args.lr_gamma, lr_milestone_every=args.lr_every,
            iw=args.iw,
            num_epochs=args.num_epochs
        )

    else:
        ut.load_model_by_name(model, global_step=args.num_epochs)

    if args.mode in ['val', 'test']:
        model.set_to_eval()
        split_set = LoadDataset2(
            root_dir=root_dir,
            mode=args.mode,
            shift_scale=shift_scale,
            filter_ev=False,
            smooth=None,
        )
        val_set = {
            "x": torch.FloatTensor(split_set.other).to(device),
            "y": torch.FloatTensor(split_set.meta).to(device),
            "c": None,
        }
        summaries = OrderedDict({
            'epoch': args.num_epochs,
            'loss': 0,
            'kl_z': 0,
            'rec_mse': 0,
            'rec_var': 0,
            'loss_type': 0,
            'lr': args.lr,
            'var_pen': model.var_pen,
        })

        ut.save_latent(model, val_set, mode=args.mode, is_car_model=False)

        ut.evaluate_lower_bound2(model, val_set, run_iwae=True, mode=args.mode, repeats=100, summaries=summaries)

    if args.mode == 'plot':

        # print(shift_scale["other"])
        # print(shift_scale)

        make_image_load(model, shift_scale["other"], (args.log_ev==1))
        make_image_load_day(model, shift_scale["other"], (args.log_ev==1))
        make_image_load_z(model, shift_scale["other"], (args.log_ev==1))

    if args.mode == 'load':
        if verbose: print(model)
    return model


def main(call_args=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, default='train', help="Flag for train, val, test, plot")
    parser.add_argument('--model', type=str, default='ff', help="model_architecture: ff, lstm")
    parser.add_argument('--z', type=int, default=5, help="Number of latent dimensions")
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of training iterations")
    parser.add_argument('--run', type=int, default=0, help="Run ID. In case you want to run replicates")
    parser.add_argument('--batch', type=int, default=128, help="Batch size")
    parser.add_argument('--lr', type=float, default=9e-3, help="Learning Rate(initial)")
    parser.add_argument('--warmup', type=int, default=0, help="Fix variance during first epoch of training")
    parser.add_argument('--var_pen', type=int, default=10, help="Penalty for variance - multiplied with var loss term")
    parser.add_argument('--lr_gamma', type=float, default=0.335, help="Anneling factor of lr")
    parser.add_argument('--lr_every', type=int, default=5, help="Number of lr anneling milestones")
    parser.add_argument('--k', type=int, default=1, help="Number mixture components in MoG prior")
    parser.add_argument('--iw', type=int, default=0, help="Number of IWAE samples for training")
    parser.add_argument('--log_ev', type=int, default=0, help="log-normalize car values")
    parser.add_argument('--hourly', type=int, default=0, help="hourly data instead of 15min resolution data")
    parser.add_argument('--smooth', type=int, default=2, help="0: original data, 1: loess data, 2: random mix")
    args = parser.parse_args()
    
    if call_args is not None:
        for k, v in call_args.items():
            print("Overriding default arg with call arg: ", k, v)
            setattr(args, k, v)

    # RUN
    return run(args)


if __name__ == '__main__':
    print('main called externally')
    model = main({"mode": 'train'})
    model = main({"mode": 'plot'})
    model = main({"mode": 'val'})
    # model = main({"mode": 'test'})

