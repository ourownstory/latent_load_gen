from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd


class HourlyLoad2017Dataset(Dataset):
    """hourly aggregate load of a day dataset"""

    def __init__(self, root_dir, data_ids):
        """
        Args:
            root_dir (string): Directory with all the dataids-subdirs.
            data_ids (list): Optional transform to be applied
                on a sample.
        """
        self.data_ids = data_ids
        self.root_dir = root_dir
        self.csv_list = []
        for path, subdirs, files in os.walk(root_dir):
            # print(path, subdirs, files)
            for subdir in subdirs:
                for name in files:
                    self.csv_list.append(os.path.join(subdir, name))

    def __len__(self):
        return len(self.csv_list)

    def __getitem__(self, idx):
        csv_name = os.path.join(self.root_dir,
                                self.csv_list[idx])
        image = pd.read_csv(csv_name)

        # old
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

