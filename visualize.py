import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
plt.show()
