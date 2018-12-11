from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
import torch
from visualize import visualize_data2, visualize_data2_smooth

class LoadDataset2(Dataset):
    """hourly aggregate load of a day dataset, with conditionals"""

    def __init__(self, root_dir, mode,
                 shift_scale=None, filter_ev=False, log_car=False,
                 smooth=None, as_numpy=False):
        """
        Args:
            root_dir (string): Directory.
            mode (str): Subdir of root_dir (e.g. 'train', 'val', 'test'),
                        contains subdirs (data_ids) with files (day_nr.csv)
        """
        self.root_dir = os.path.join(root_dir, mode)
        self.eps = 1e-5
        self.shift_scale = shift_scale
        self.filter_ev = filter_ev
        self.log_car = log_car
        self.smooth = None if smooth == 0 else smooth
        # read

        '''
        # Will's version
        filepath = '/Users/willlauer/Desktop/latent_load_gen/data/split'
        self.use = pd.read_csv(filepath + '/' + mode + '/use.csv').values
        self.car = pd.read_csv(filepath + '/' + mode + '/car.csv').values
        self.loessUse = pd.read_csv(filepath + '/' + mode + '/loess_use.csv').values
        self.loessCar = pd.read_csv(filepath + '/' + mode + '/loess_car.csv').values
        self.meta = pd.read_csv(filepath + '/' + mode + '/meta.csv').astype('float32').values

        '''
        # Oskar's version
        #self.use = pd.read_csv(os.path.join(self.root_dir, "use.csv")).values
        self.car = pd.read_csv(os.path.join(self.root_dir, "car.csv")).values
        self.other = pd.read_csv(os.path.join(self.root_dir, "other.csv")).values
        if self.smooth is not None:
            #self.loessUse = pd.read_csv(os.path.join(self.root_dir, "loess_use.csv")).values
            self.loessCar = pd.read_csv(os.path.join(self.root_dir, "loess_car.csv")).values
            self.loessOther = pd.read_csv(os.path.join(self.root_dir, "loess_other.csv")).values

        # store in array like car, other
        # rows correspond to car , use,
        # metadata in vector format
        # one-hot for day of week, 2 for temp, 1 for rain
        self.meta = pd.read_csv(os.path.join(self.root_dir, "meta.csv")).astype('float32').values
        # print('meta shape', self.meta.shape)

        if self.smooth is not None:
            # Apply same consistency to the loess smoothed curves
            self.loessCar = np.maximum(self.loessCar, 0)
            #self.loessUse = np.maximum(self.loessUse, 0)
            #self.loessOther = self.loessUse - self.loessCar
            self.loessOther = np.maximum(self.loessOther, 0)

        # ensure consistency
        self.car = np.maximum(self.car, 0)
        # add:
        #self.other = self.use - self.car
        # ensure consistency
        self.other = np.maximum(self.other, 0)
        #self.use = None

        if shift_scale is None:
            # we want to get the other shift scale for all entries, as that model is trained such.
            self.shift_scale = {
                "other": (np.mean(self.other.reshape(-1)), np.std(self.other.reshape(-1))),
            }
            if self.smooth is not None:
                self.shift_scale["loessOther"] = (np.mean(self.loessOther.reshape(-1)), np.std(self.loessOther.reshape(-1)))

        if self.filter_ev:
            # the ev-magnitude
            # self.y_real = self.car.sum(-1)
            has_ev = self.car.sum(-1) > 1
            self.car = self.car[has_ev]
            self.other = self.other[has_ev]
            self.meta = self.meta[has_ev]
            if self.smooth is not None:
                self.loessOther = self.loessOther[has_ev]
                self.loessCar = self.loessCar[has_ev]

        if self.log_car:
            # self.car = np.log(self.car + 1)
            self.car = np.log(self.car + self.eps)
            if self.smooth is not None and smooth != 1:
                raise NotImplementedError
            else:
                self.loessCar = np.log(self.loessCar + self.eps)

        if shift_scale is None:
            # add car scaling factors only after potentially filtering out.
            self.shift_scale["car"] = (np.mean(self.car), np.std(self.car))
            if self.smooth is not None:
                self.shift_scale["loessOther"] = (np.mean(self.loessOther.reshape(-1)), np.std(self.loessOther.reshape(-1)))
                self.shift_scale["loessCar"] = (np.mean(self.loessCar.reshape(-1)), np.std(self.loessCar.reshape(-1)))

        # always shift and scale
        self.car = (self.car - self.shift_scale["car"][0]) / self.shift_scale["car"][1]
        self.other = (self.other - self.shift_scale["other"][0]) / self.shift_scale["other"][1]
        if self.smooth is not None:
            self.loessOther = (self.loessOther - self.shift_scale["loessOther"][0]) / self.shift_scale["loessOther"][1]
            self.loessCar = (self.loessCar - self.shift_scale["loessCar"][0]) / self.shift_scale["loessCar"][1]


        # manipulate metadata
        # min temp: standardize
        self.meta[:, 0] = (self.meta[:, 0] - 40) / 15
        # max temp: standardize
        self.meta[:, 1] = (self.meta[:, 1] - 80) / 10
        # precipitation: standardize
        # self.meta[:, 2] = (self.meta[:, 2] - 0.2) / 0.5
        self.meta[:, 2] = np.log(10*self.meta[:, 2] + 0.1)

        # convert to one-hot
        day = np.zeros((self.meta.shape[0], 7))
        day[np.arange(self.meta.shape[0]), self.meta[:, 3].astype(int)] = 1
        self.meta = np.concatenate((self.meta[:, :3], day), axis=1)
        self.dim_meta = self.meta.shape[1]

        if not as_numpy:
            self.meta = torch.FloatTensor(self.meta)
            self.car = torch.FloatTensor(self.car)
            self.other = torch.FloatTensor(self.other)
            if self.smooth is not None:
                self.loessOther = torch.FloatTensor(self.loessOther)
                self.loessCar = torch.FloatTensor(self.loessCar)

    def __len__(self):
        # return self.other.size()[0]
        return self.other.shape[0]

    def __getitem__(self, idx):
        if self.smooth is None:
            sample = {
                "car": self.car[idx],
                "other": self.other[idx],
                "meta": self.meta[idx],

            }
        elif self.smooth == 1:
            sample = {
                "car": self.loessCar[idx],
                "other": self.loessOther[idx],
                "meta": self.meta[idx],

            }
        elif self.smooth == 2:
            r = np.random.rand()
            sample = {
                "car": (r * self.car[idx] + (1 - r) * self.loessCar[idx]),
                "other": (r * self.other[idx] + (1 - r) * self.loessOther[idx]),
                "meta": self.meta[idx],
            }
        else:
            raise NotImplementedError

        # Apply loess smoothing to the x entry
        #x = sample["other"] # batch_size * 96
        #lx = sample["loessOther"] # batch_size * 96
        #r = np.expand_dims(np.random.rand(x.shape[0]), 1) # batch_size * 1
        #smoothedX = (r * x + (1 - r) * lx).float()
        #sample["other"] = smoothedX

        return sample


def run_test(split):
    print("\n")
    print(split)
    shift_scale = {"other": (0, 1), "car": (0, 1), "loessOther": (0, 1), "loessCar": (0, 1)}
    # shift_scale = None  # to compute new
    # root_dir = "../data/CS236/data60/split"
    root_dir =  "../data/CS236/data15"
    split_set = LoadDataset2(
        root_dir=root_dir,
        mode=split,
        filter_ev=False,
        shift_scale=shift_scale,
        # log_car=True,
        as_numpy=True,
    )
    split_set_ev = LoadDataset2(
        root_dir=root_dir,
        mode=split,
        filter_ev=True,
        shift_scale=shift_scale,
        # log_car=True,
        as_numpy=True,
    )
    print("No EV", len(split_set))
    print("EV", len(split_set_ev))

    # print("car", np.percentile(split_set.car.sum(-1), [0, 5, 50, 95, 99, 100]))
    # print("other", np.percentile(split_set.other.sum(-1), [0, 5, 50, 95, 99, 100]))
    #
    # print("car-filter", np.percentile(split_set_ev.car.sum(-1), [0, 5, 50, 95, 99, 100]))
    # print("other-filter", np.percentile(split_set_ev.other.sum(-1), [0, 5, 50, 95, 99, 100]))

    print("car", np.percentile(split_set.car.reshape(-1), [0, 5, 50, 95, 99, 100]))
    print(min(list(split_set.car.reshape(-1))), max(list(split_set.car.reshape(-1))))
    print("other", np.percentile(split_set.other.reshape(-1), [0, 5, 50, 95, 99, 100]))
    for i in range(split_set.meta.shape[1]):
        print("meta", i, np.percentile(split_set.meta[:, i], [0, 1, 5, 50, 75, 95, 99, 100]))
        print("mean, std", (np.mean(split_set.meta[:, i]), np.std(split_set.meta[:, i])))

    print("car-filter", np.percentile(split_set_ev.car.reshape(-1), [0, 5, 50, 95, 99, 100]))
    print("other-filter", np.percentile(split_set_ev.other.reshape(-1), [0, 5, 50, 95, 99, 100]))

    if shift_scale is None:
        print("no filter", split_set.shift_scale)
        print("with filter", split_set_ev.shift_scale)


def run_plot(split, smooth=False):
    print("\n")
    print(split)
    shift_scale = {"other": (0, 1), "car": (0, 1), "loessOther": (0, 1), "loessCar": (0, 1)}
    # shift_scale = None  # to compute new
    # root_dir = "../data/CS236/data60/split"
    root_dir =  "../data/CS236/data15_final"

    split_set_ev = LoadDataset2(
        root_dir=root_dir,
        mode=split,
        filter_ev=True,
        shift_scale=shift_scale,
        log_car=False,
        smooth=0,
        as_numpy=True,
    )
    # print("EV", len(split_set_ev))

    if smooth:
        split_set_ev_smooth = LoadDataset2(
            root_dir=root_dir,
            mode=split,
            filter_ev=True,
            shift_scale=shift_scale,
            log_car=False,
            smooth=1,
            as_numpy=True,
        )
        visualize_data2_smooth(split_set_ev, split_set_ev_smooth)
    else:
        visualize_data2(split_set_ev)


if __name__ == "__main__":
    # for spl in ['test', 'val', 'train']:
    for spl in ['test']:
        # run_test(spl)
        run_plot(spl, smooth=True)
