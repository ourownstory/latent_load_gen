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


def visualize_data2(dataset):
    print(dataset)
    print(len(dataset))
    num_samples = 4*4

    for i, idx in enumerate(np.random.randint(0, high=len(dataset), size=num_samples)):
        sample = dataset[idx]

        # print(i, sample)
        # print(i, sample['use'].shape, sample['car'].shape)

        ax = plt.subplot(4, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(idx))
        # ax.axis('off')
        plt.plot(sample['other'], 'b')
        plt.plot(sample['car'], 'r')

    plt.show()


def visualize_data2_smooth(dataset, dataset_smooth):
    print(len(dataset))
    num_samples = 4*4

    for i, idx in enumerate(np.random.randint(0, high=len(dataset), size=num_samples)):
        sample = dataset[idx]
        sample_smooth = dataset_smooth[idx]

        # print(i, sample)
        # print(i, sample['use'].shape, sample['car'].shape)

        ax = plt.subplot(4, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(idx))
        # ax.axis('off')
        plt.plot(sample['other'], 'b')
        plt.plot(sample['car'], 'r')
        plt.plot(sample_smooth['other'], 'k')
        plt.plot(sample_smooth['car'], 'y')

    plt.show()



if __name__ == '__main__':
    visualize_data(use_direct=False)
    # visualize_data2(root_dir="../data/CS236/data15_final")
