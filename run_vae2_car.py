import run_vae2

import argparse
import torch
import tqdm
from codebase import utils as ut
from codebase.models.vae2 import VAE2CAR, GMVAE2CAR
from codebase.train import train2
from pprint import pprint
from plot_vae2 import make_image_load, make_image_load_z, make_image_load_z_use
from LoadDataset2 import LoadDataset2


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
        ('run{:02d}', args.run)
    ]
    model_name = 'car' + '_'.join([t.format(v) for (t, v) in layout])
    if verbose: pprint(vars(args))
    print('Model name:', model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load use-model
    use_model = run_vae2.main({"mode": 'load', })

    if args.k > 1:
        model = GMVAE2CAR(
            nn=args.model, name=model_name,
            z_dim=args.z, x_dim=24 if args.hourly==1 else 96, c_dim=use_model.z_dim,
            warmup=(args.warmup==1), var_pen=args.var_pen, use_model=use_model, k=args.k
        ).to(device)
    else:
        model = VAE2CAR(
            nn=args.model, name=model_name,
            z_dim=args.z, x_dim=24 if args.hourly==1 else 96, c_dim=use_model.z_dim,
            warmup=(args.warmup==1), var_pen=args.var_pen, use_model=use_model
        ).to(device)

    root_dir = "data/split" if args.hourly else "../data/CS236"
    # load train loader anyways - to get correct shift_scale values.
    train_loader = torch.utils.data.DataLoader(
        LoadDataset2(root_dir=root_dir, mode='train', shift_scale=None, filter_ev=True, log_car=(args.log_ev==1)),
        batch_size=args.batch, shuffle=True
    )
    shift_scale = train_loader.dataset.shift_scale

    if args.mode == 'train':
        split_set = LoadDataset2(
            root_dir=root_dir,
            mode='val',
            shift_scale=shift_scale,
            filter_ev=True,
            log_car=(args.log_ev==1),
        )
        val_set = {
            "x": torch.FloatTensor(split_set.car).to(device),
            # TODO: add metadata
            "y": None,
            "c": torch.FloatTensor(split_set.other).to(device),
        }
        writer = ut.prepare_writer(model_name, overwrite_existing=True)

        # make sure not to train the first VAE
        for p in model.use_model.parameters():
            p.requires_grad = False

        train2(
            model=model,
            train_loader=train_loader,
            val_set=val_set,
            tqdm=tqdm.tqdm,
            writer=writer,
            lr=args.lr, lr_gamma=args.lr_gamma, lr_milestone_every=args.lr_every,
            iw=args.iw,
            num_epochs=args.num_epochs,
            is_car_model=True,
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
            log_car=(args.log_ev==1),
        )
        val_set = {
            "x": torch.FloatTensor(split_set.other).to(device),
            # TODO: add metadata
            "y": None,
            "c": torch.FloatTensor(split_set.other).to(device),
        }
        ut.evaluate_lower_bound2(model, val_set, run_iwae=(args.iw>=1), mode=args.mode)

    if args.mode == 'plot':
        make_image_load(model, shift_scale["car"], (args.log_ev==1))
        make_image_load_z(model, shift_scale["car"], (args.log_ev==1))
        make_image_load_z_use(model, shift_scale["car"], (args.log_ev==1))

    if args.mode == 'load':
        if verbose: print(model)
    return model


def main(call_args=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, default='load', help="Flag for train, val, test, plot")
    parser.add_argument('--model', type=str, default='v1', help="model_architecture: v1, lstm")
    parser.add_argument('--z', type=int, default=5, help="Number of latent dimensions")
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of training iterations")
    parser.add_argument('--run', type=int, default=0, help="Run ID. In case you want to run replicates")
    parser.add_argument('--batch', type=int, default=64, help="Batch size")
    parser.add_argument('--lr', type=float, default=8e-3, help="Learning Rate(initial)")
    parser.add_argument('--warmup', type=int, default=1, help="Fix variance during first epoch of training")
    parser.add_argument('--var_pen', type=int, default=10, help="Penalty for variance - multiplied with var loss term")
    parser.add_argument('--lr_gamma', type=float, default=0.5, help="Anneling factor of lr")
    parser.add_argument('--lr_every', type=int, default=5, help="lr anneling every x epochs")
    parser.add_argument('--k', type=int, default=10, help="Number mixture components in MoG prior")
    parser.add_argument('--iw', type=int, default=4, help="Number of IWAE samples for training, will be SQARED!")
    parser.add_argument('--log_ev', type=int, default=0, help="log-normalize car values")
    parser.add_argument('--hourly', type=int, default=1, help="hourly data instead of 15min resolution data")
    # parser.add_argument('--run_car',    type=int, default=0,    help="whether to run the second model or first")
    args = parser.parse_args()

    if call_args is not None:
        for k, v in call_args.items():
            print("Overriding default arg with call arg: ", k, v)
            setattr(args, k, v)

    # RUN
    return run(args)


if __name__ == '__main__':
    model = main({"mode": 'train', "model": 'lstm'})

