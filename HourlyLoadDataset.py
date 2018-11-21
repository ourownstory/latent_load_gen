from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np


class HourlyLoad2017Dataset(Dataset):
    """hourly aggregate load of a day dataset"""

    def __init__(self, root_dir, mode='train', in_memory=False, log_normal=False, shift_scale=None):
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
        if in_memory:
            self.use = pd.read_csv(os.path.join(self.root_dir, "use.csv")).values
            self.car = pd.read_csv(os.path.join(self.root_dir, "car.csv")).values

            if log_normal:
                self.use = np.log(self.use + self.eps)
                self.car = np.log(self.car + self.eps)
                self.other = np.log(self.other + self.eps)
                if shift_scale is None:
                    self.shift_scale = {
                        "use": (self.use.mean(), np.std(self.use)),
                        "car": (self.car.mean(), np.std(self.car)),
                    }
                self.use = (self.use - self.shift_scale["use"][0]) / self.shift_scale["use"][1]
                self.car = (self.car - self.shift_scale["car"][0]) / self.shift_scale["car"][1]

        else:
            self.data_ids = sorted([int(x) for x in next(os.walk(self.root_dir))[1]])
            self.csv_list = []
            # print(self.root_dir, self.data_ids)
            for subdir in [str(x) for x in self.data_ids]:
                # print(next(os.walk(os.path.join(self.root_dir, subdir)))[2])
                for file_name in next(os.walk(os.path.join(self.root_dir, subdir)))[2]:
                    self.csv_list.append((subdir, file_name))

    def __len__(self):
        if self.in_memory:
            return self.use.shape[0]
        else:
            return len(self.csv_list)

    def __getitem__(self, idx):
        if self.in_memory:
            sample = self.use[idx]
        else:
            csv = pd.read_csv(os.path.join(self.root_dir, *self.csv_list[idx]))
            # sample = {
            #     "use": csv["use"].values,
            #     "car": csv["car"].values
            # }
            sample = csv["use"].values
        return sample


def run_test(split):
    print(split)
    split_set = HourlyLoad2017Dataset(root_dir="data/split", mode=split, in_memory=True, log_normal=False)
    print(len(split_set))
    # print(split_set.use)
    print(split_set.shift_scale)
    print("use", np.percentile(split_set.use.sum(-1), [0, 5, 50, 95, 100]))
    print("car", np.percentile(split_set.car.sum(-1), [0, 5, 50, 95, 100]))


if __name__ == "__main__":
    for spl in ['test', 'val', 'train']:
        run_test(spl)
