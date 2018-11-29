from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np


class ConditionalLoadDataset(Dataset):
    """hourly aggregate load of a day dataset, with conditionals"""

    def __init__(self, root_dir, mode='train', in_memory=True, log_normal=False, shift_scale=None,
                 get_ev_subset=True):
        """
        Args:
            root_dir (string): Directory.
            mode (str): Subdir of root_dir (e.g. 'train', 'val', 'test'),
                        contains subdirs (data_ids) with files (day_nr.csv)
        """
        self.root_dir = root_dir
        self.root_dir = os.path.join(self.root_dir, mode)
        self.in_memory = in_memory
        self.eps = 1e-5
        self.shift_scale = shift_scale
        self.ev_subset = get_ev_subset
        if in_memory:
            self.use = pd.read_csv(os.path.join(self.root_dir, "use.csv")).values
            self.car = pd.read_csv(os.path.join(self.root_dir, "car.csv")).values
            # add:
            self.other = self.use - self.car
            # ensure consistency
            self.car = np.maximum(self.car, 0)
            self.use = np.maximum(self.use, self.car)
            self.other = np.maximum(self.other, 0)
            # the conditional, representing the log ratio of EV vs other appliances.
            # self.y_real = np.log(self.car.sum(-1) + self.eps) - np.log(self.other.sum(-1) + self.eps)
            # self.y_real = self.car.sum(-1) / (self.use.sum(-1) + self.eps)
            # the conditional, as EV magnitude
            self.y_real = self.car.sum(-1)

            if get_ev_subset:
                self.x_0 = self.other[self.y_real > 0.1]
                self.x_1 = self.use[self.y_real > 0.1]
                self.y_real = self.y_real[self.y_real > 0.1]
                self.y_real = self.y_real * 0.1
                # self.y_real = np.log(1 + self.y_real)
            else:
                self.x_no_ev = self.other
                self.y_real = None

            # remove
            self.car = None
            self.use = None
            self.other = None

            if log_normal:
                if get_ev_subset:
                    self.x_0 = np.log(self.x_0 + self.eps)
                    self.x_1 = np.log(self.x_1 + self.eps)
                    self.y_real = np.log(1.0 + 10*self.y_real)
                else:
                    self.x_no_ev = np.log(self.x_no_ev + self.eps)

            if shift_scale is not None:
                if get_ev_subset:
                    self.x_0 = (self.x_0 - self.shift_scale[0]) / self.shift_scale[1]
                    self.x_1 = (self.x_1 - self.shift_scale[0]) / self.shift_scale[1]
                else:
                    self.x_no_ev = (self.x_no_ev - self.shift_scale[0]) / self.shift_scale[1]
        else:
            raise NotImplementedError

    def __len__(self):
        if self.in_memory:
            if self.ev_subset:
                return self.y_real.shape[0]
            else:
                return self.x_no_ev.shape[0]
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        if self.in_memory:
            if self.ev_subset:
                sample = {
                    "x_0": self.x_0[idx],
                    "x_1": self.x_1[idx],
                    "y_real": self.y_real[idx],
                }
            else:
                sample = {
                    "x_no_ev": self.x_no_ev[idx],
                }
        else:
            raise NotImplementedError
        return sample


def run_test(split):
    print("\n")
    print(split)
    # shift_scale = (-0.5223688943269471, 2.6144099155163927)
    shift_scale = None  # to compute new
    # root_dir = "data/split"
    root_dir = "../data/CS236"
    split_set = ConditionalLoadDataset(
        root_dir=root_dir, mode=split, in_memory=True, log_normal=False,
        shift_scale=shift_scale,
        get_ev_subset=False
    )
    split_set_ev = ConditionalLoadDataset(
        root_dir=root_dir, mode=split, in_memory=True, log_normal=False,
        shift_scale=shift_scale,
        get_ev_subset=True
    )
    print("No EV", len(split_set))
    print("EV", len(split_set_ev))
    print("x_0", np.percentile(split_set_ev.x_0.sum(-1), [0, 5, 50, 95, 100]))
    print("ev", np.percentile((split_set_ev.x_1 - split_set_ev.x_0).sum(-1), [0, 5, 50, 95, 100]))
    print("x_no_ev", np.percentile(split_set.x_no_ev.sum(-1), [0, 5, 50, 95, 100]))
    print("y_real", np.percentile(split_set_ev.y_real, [0, 5, 10, 25, 50, 75, 90, 95, 100]))
    # print("y_real == 0", np.mean(split_set_ev.y_real == 0))
    # print("y_real < log(1+1)", np.mean(split_set_ev.y_real < np.log(1+1)))

    if shift_scale is None:
        shift_scale_new = (
            np.concatenate((split_set.x_no_ev, split_set_ev.x_0, split_set_ev.x_1), axis=0).mean(),
            np.std(np.concatenate((split_set.x_no_ev, split_set_ev.x_0, split_set_ev.x_1), axis=0))
        )
        print("shift_scale_new: ", shift_scale_new)


if __name__ == "__main__":
    for spl in ['test', 'val', 'train']:
        run_test(spl)
