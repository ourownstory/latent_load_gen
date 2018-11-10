import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

path = "data/processed/"
dataids = np.array(pd.read_csv(path + "dataids.csv", header=None))[0]
# print(dataids)
use = pd.read_csv(path + "use_{}.csv".format(dataids[0])).values
# print(use)
car = pd.read_csv(path + "car_{}.csv".format(dataids[0])).values
# print(car)
for i in range(3):
    plt.plot(use[i], 'b')
    plt.plot(car[i], 'ro-')
# plt.show()

root_dir = path
csv_list = []
for path, subdirs, files in os.walk(root_dir):
    # print(path, subdirs, files)
    for subdir in subdirs:
        for name in files:
            csv_list.append(os.path.join(subdir, name))

print(csv_list)
