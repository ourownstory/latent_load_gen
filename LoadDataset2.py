from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
import torch


class LoadDataset2(Dataset):
    """hourly aggregate load of a day dataset, with conditionals"""

    def __init__(self, root_dir, mode='train', shift_scale=None, filter_ev=True):
        """
        Args:
            root_dir (string): Directory.
            mode (str): Subdir of root_dir (e.g. 'train', 'val', 'test'),
                        contains subdirs (data_ids) with files (day_nr.csv)
        """
        self.root_dir = root_dir
        self.root_dir = os.path.join(self.root_dir, mode)
        self.eps = 1e-5
        self.shift_scale = shift_scale
        self.filter_ev = filter_ev

        # read
        self.use = pd.read_csv(os.path.join(self.root_dir, "use.csv")).values
        self.car = pd.read_csv(os.path.join(self.root_dir, "car.csv")).values

        # TODO: add reading of metadata
        self.meta = None

        # ensure consistency
        self.car = np.maximum(self.car, 0)
        # add:
        self.other = np.maximum(self.use, 0) - self.car
        # ensure consistency
        self.other = np.maximum(self.other, 0)
        self.use = None

        if shift_scale is None:
            # we want to get the other shift scale for all entries, as that model is trained such.
            self.shift_scale = {
                "other": (np.mean(self.other), np.std(self.other)),
            }

        if self.filter_ev:
            # the ev-magnitude
            self.y_real = self.car.sum(-1)
            self.car = self.car[self.y_real > 0.1]
            self.other = self.other[self.y_real > 0.1]
            self.y_real = None

        if shift_scale is None:
            # add car scaling factors only after potentially filtering out.
            self.shift_scale["car"] = (np.mean(self.car), np.std(self.car))

        # always shift and scale
        self.car = (self.car - self.shift_scale["car"][0]) / self.shift_scale["car"][1]
        self.other = (self.other - self.shift_scale["other"][0]) / self.shift_scale["other"][1]
        self.car = torch.FloatTensor(self.car)
        self.other = torch.FloatTensor(self.other)

    def __len__(self):
        return self.other.size()[0]

    def __getitem__(self, idx):
        # TODO: add metadata
        sample = {
            "car": self.car[idx],
            "other": self.other[idx],
            # "meta": None,
        }
        return sample


def run_test(split):
    print("\n")
    print(split)
    # shift_scale = (-0.5223688943269471, 2.6144099155163927)
    shift_scale = None  # to compute new
    root_dir = "data/split"
    # root_dir = "../data/CS236"
    split_set = LoadDataset2(
        root_dir=root_dir,
        mode=split,
        filter_ev=False,
        shift_scale=shift_scale,
    )
    split_set_ev = LoadDataset2(
        root_dir=root_dir,
        mode=split,
        filter_ev=True,
        shift_scale=shift_scale,
    )
    print("No EV", len(split_set))
    print("EV", len(split_set_ev))
    print("car", np.percentile(split_set.car.sum(-1), [0, 5, 50, 95, 99, 100]))
    print("other", np.percentile(split_set.other.sum(-1), [0, 5, 50, 95, 99, 100]))

    print("car-filter", np.percentile(split_set_ev.car.sum(-1), [0, 5, 50, 95, 99, 100]))
    print("other-filter", np.percentile(split_set_ev.other.sum(-1), [0, 5, 50, 95, 99, 100]))

    if shift_scale is None:
        print("no filter", split_set.shift_scale)
        print("with filter", split_set_ev.shift_scale)


if __name__ == "__main__":
    for spl in ['test', 'val', 'train']:
        run_test(spl)
