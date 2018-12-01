import numpy as np
import torch
from codebase import utils as ut
from torch import optim
from codebase.models.cvae import CVAE


def train(model, train_loader, device, tqdm, writer, lr, lr_gamma, lr_milestones, iw,
          iter_max=np.inf, iter_save=np.inf,
          model_name='model', reinitialize=False):
    # Optimization
    if reinitialize:
        model.apply(ut.reset_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma)
    i = 0
    # model.warmup = True
    # print("warmup", model.warmup)
    with tqdm(total=iter_max) as pbar:
        while True:
            for batch_idx, sample in enumerate(train_loader):
                i += 1  # i is num of gradient steps taken by end of loop iteration
                if i == (iter_max//4):
                    # start learning variance
                    model.warmup = False
                optimizer.zero_grad()
                x = torch.tensor(sample).float().to(device)
                loss, summaries = model.loss(x, iw)

                loss.backward()
                optimizer.step()
                scheduler.step()

                # Feel free to modify the progress bar
                pbar.set_postfix(loss='{:.2e}'.format(loss))
                pbar.update(1)

                # Log summaries
                if i % 50 == 0:
                    ut.log_summaries(writer, summaries, i)

                # Save model
                if i % iter_save == 0:
                    ut.save_model_by_name(model, i)
                    # print(optimizer.param_groups[0]['lr'])
                    # print("warmup", model.warmup)
                    print("\n", [(key, v.item()) for key, v in summaries.items()])

                if i == iter_max:
                    return


def train_c(model, train_loader,  train_loader_ev, device, tqdm, writer, lr, lr_gamma, lr_milestones, iw,
          iter_max=np.inf, iter_save=np.inf,
          model_name='model', reinitialize=False):
    assert isinstance(model, CVAE)

    # Optimization
    if reinitialize:
        model.apply(ut.reset_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma)

    # model.warmup = True
    # print("warmup", model.warmup)

    iterator = iter(train_loader)

    iterator_ev = iter(train_loader_ev)
    i = 0
    with tqdm(total=iter_max) as pbar:
        while True:
            i += 1  # i is num of gradient steps taken by end of loop iteration
            if i == (iter_max // 4):
                # start learning variance
                model.warmup = False
            optimizer.zero_grad()

            # must handle two data-loader queues...
            try:
                sample = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                sample = next(iterator)
            try:
                sample_ev = next(iterator_ev)
            except StopIteration:
                iterator_ev = iter(train_loader_ev)
                sample_ev = next(iterator_ev)

            # combine the batches
            for k, v in sample.items():
                sample[k] = torch.tensor(v).float().to(device)
            for k, v in sample_ev.items():
                sample[k] = torch.tensor(v).float().to(device)
            sample_ev = None

            # run model
            loss, summaries = model.loss(sample, iw)

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Feel free to modify the progress bar
            pbar.set_postfix(loss='{:.2e}'.format(loss))
            pbar.update(1)

            # Log summaries
            if i % 50 == 0:
                ut.log_summaries(writer, summaries, i)

            # Save model
            if i % iter_save == 0:
                ut.save_model_by_name(model, i)
                # print(optimizer.param_groups[0]['lr'])
                # print("warmup", model.warmup)
                print("\n", [(key, v.item()) for key, v in summaries.items()])

            if i == iter_max:
                return


def train2(model, train_loader, device, tqdm, writer, lr, lr_gamma, lr_milestones, iw,
          iter_max=np.inf, iter_save=np.inf,
          model_name='model', reinitialize=False):
    assert isinstance(model, CVAE)

    # Optimization
    if reinitialize:
        model.apply(ut.reset_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma)

    # model.warmup = True
    # print("warmup", model.warmup)

    iterator = iter(train_loader)

    i = 0
    with tqdm(total=iter_max) as pbar:
        while True:
            i += 1  # i is num of gradient steps taken by end of loop iteration
            if i == (iter_max // 4):
                # start learning variance
                model.warmup = False
            optimizer.zero_grad()

            # must handle two data-loader queues...
            try:
                sample = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                sample = next(iterator)

            # run model
            loss, summaries = model.loss(
                x=sample["other"], meta=sample["meta"], c=None, iw=0
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Feel free to modify the progress bar
            pbar.set_postfix(loss='{:.2e}'.format(loss))
            pbar.update(1)

            # Log summaries
            if i % 50 == 0:
                ut.log_summaries(writer, summaries, i)

            # Save model
            if i % iter_save == 0:
                ut.save_model_by_name(model, i)
                # print(optimizer.param_groups[0]['lr'])
                # print("warmup", model.warmup)
                print("\n", [(key, v.item()) for key, v in summaries.items()])

            if i == iter_max:
                return
