import numpy as np
import torch
from codebase import utils as ut
from torch import optim


def train(model, train_loader, device, tqdm, writer,
          iter_max=np.inf, iter_save=np.inf,
          model_name='model', reinitialize=False):
    # Optimization
    if reinitialize:
        model.apply(ut.reset_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    i = 0
    # model.warmup = True
    with tqdm(total=iter_max) as pbar:
        while True:
            for batch_idx, sample in enumerate(train_loader):
                i += 1  # i is num of gradient steps taken by end of loop iteration
                if i == (iter_max//4):
                    # start learning variance
                    model.warmup = False
                optimizer.zero_grad()
                x = torch.tensor(sample).float().to(device)
                # TODO: why use bernoulli here (from hw code)??
                # x = torch.bernoulli(x.to(device).reshape(x.size(0), -1))
                loss, summaries = model.loss(x)

                loss.backward()
                optimizer.step()

                # Feel free to modify the progress bar
                pbar.set_postfix(loss='{:.2e}'.format(loss))
                pbar.update(1)

                # Log summaries
                if i % 50 == 0:
                    ut.log_summaries(writer, summaries, i)

                # Save model
                if i % iter_save == 0:
                    ut.save_model_by_name(model, i)

                if i == iter_max:
                    return
