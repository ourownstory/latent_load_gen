import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from HourlyLoadDataset import HourlyLoad2017Dataset


def visualize_data(use_direct):
    if use_direct:
        path_raw = "data/raw"
        path = "data/processed"
        dataids = pd.read_csv(os.path.join(path_raw, "dataids.csv"), header=None).values[0]
        dataids = [str(x) for x in dataids]
        print(dataids)

        root_dir = path
        csv_list = []
        subdirs = next(os.walk(root_dir))[1]
        print(sorted(subdirs))
        for subdir in subdirs:
            # only add the ones given by data_ids
            if subdir in dataids:
                for file_name in next(os.walk(os.path.join(root_dir, subdir)))[2]:
                    csv_list.append((subdir, file_name))

        print(csv_list)

        # print(dataids)
        csv = pd.read_csv(os.path.join(path, *csv_list[2]))
        # use = csv.use.values
        # car = csv.car.values
        # print(car)
        plt.plot(csv.use.values, 'b')
        plt.plot(csv.car.values, 'ro-')
        plt.show()

    else:
        load_dataset = HourlyLoad2017Dataset(root_dir="data/split", mode='test')
        print(load_dataset)
        print(len(load_dataset))

        for i in range(len(load_dataset)):
            sample = load_dataset[i]

            # print(i, sample)
            # print(i, sample['use'].shape, sample['car'].shape)

            ax = plt.subplot(2, 2, i + 1)
            plt.tight_layout()
            ax.set_title('Sample #{}'.format(i))
            # ax.axis('off')
            plt.plot(sample['use'], 'b')
            plt.plot(sample['car'], 'ro-')

            if i == 3:
                plt.show()
                break


if __name__ == '__main__':
    visualize_data(use_direct=False)
