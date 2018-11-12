from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd


class HourlyLoad2017Dataset(Dataset):
    """hourly aggregate load of a day dataset"""

    def __init__(self, root_dir, mode='train'):
        """
        Args:
            root_dir (string): Directory.
            mode (str): Subdir of root_dir (e.g. 'train', 'val', 'test'),
                        contains subdirs (data_ids) with files (day_nr.csv)
        """
        self.root_dir = root_dir
        self.root_dir = os.path.join(self.root_dir, mode)
        self.data_ids = sorted([int(x) for x in next(os.walk(self.root_dir))[1]])
        self.csv_list = []
        # print(self.root_dir, self.data_ids)
        for subdir in [str(x) for x in self.data_ids]:
            # print(next(os.walk(os.path.join(self.root_dir, subdir)))[2])
            for file_name in next(os.walk(os.path.join(self.root_dir, subdir)))[2]:
                self.csv_list.append((subdir, file_name))

    def __len__(self):
        return len(self.csv_list)

    def __getitem__(self, idx):
        csv = pd.read_csv(os.path.join(self.root_dir, *self.csv_list[idx]))
        sample = {
            "use": csv["use"].values,
            "car": csv["car"].values
        }
        return sample
