import numpy as np
import pandas as pd
from datetime import datetime
import os


rerun_raw_processing = True
num_days = 365
daily_resolution = 24
remove_missing_threshold = 3

if rerun_raw_processing:
    data_in = "data/raw/"
    data_out = "data/processed/"
    raw = pd.read_csv(data_in + "dataport-export_hourly_use_car_EV-houses.csv")
    print(raw.head())

    num_ids = len(set(raw["dataid"]))
    print("num_ids", num_ids)
    missing_values = num_days*num_ids*daily_resolution - len(raw)
    print("missing_values", missing_values, "/", num_days*num_ids*daily_resolution)

    print("negative values:", raw["use"][raw["use"] < 0])
    print("negative values:", raw["car1"][raw["car1"] < 0])

    dataid2idx = {}
    dataids = sorted(list(set(raw["dataid"])))
    for idx, dataid in enumerate(dataids):
        dataid2idx[str(int(dataid))] = idx

    use = -1*np.ones((num_ids, num_days, daily_resolution))
    car = -1*np.ones((num_ids, num_days, daily_resolution))
    for i, row in raw.iterrows():
        if (i+1)%5000 == 0:
            print(i+1)
            # break
        idx = dataid2idx[str(int(row.dataid))]
        # print(row.localminute[:-3])
        date = datetime.strptime(row.localminute[:-3], '%Y-%m-%d %H:%M:%S')
        # print(date.timetuple().tm_yday)
        # print(date.hour)
        use[idx, date.timetuple().tm_yday - 1, date.hour] = row["use"]
        car[idx, date.timetuple().tm_yday - 1, date.hour] = row["car1"]

    use_missing = use < 0
    car_missing = car < 0
    use += use_missing
    car += car_missing

    use_missing = np.sum(use_missing, axis =-1)
    car_missing = np.sum(car_missing, axis =-1)
    print(np.sum(use_missing > 6, axis =-1))
    print(np.sum(car_missing, axis =-1))

    for i, dataid in enumerate(dataids):
        for day in range(365):
            df = pd.DataFrame()
            df["use"] = use[i, day, :]
            df["car"] = car[i, day, :]
            directory = data_out + "/" + str(dataid) + "/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            df.to_csv(directory + "{}.csv".format(day), index=False)
            # df = pd.DataFrame(car[i, i, :])
            # df.to_csv(data_out + "car_{}.csv".format(dataid), index=False)

    with open(data_in + "dataids.csv", 'w') as f:
        f.write(",".join([str(i) for i in dataids]))

    df = pd.DataFrame(use_missing)
    df.to_csv(data_in + "use_missing.csv", index=False)
    df = pd.DataFrame(car_missing)
    df.to_csv(data_in + "car_missing.csv", index=False)
else:
    data_path = "data/raw/"
    dataids = pd.read_csv(data_path + "dataids.csv", header=None).values[0]
    print(dataids)
    use_missing = pd.read_csv(data_path + "use_missing.csv").values
    use_missing_average = use_missing.mean(-1)
    # print(use_missing_average.astype('int'))
    # remove all data_ids with more than x missing hourly measurements (on average)
    dataids_clean = dataids[use_missing_average < remove_missing_threshold]
    print(dataids_clean)
    with open(data_path + "dataids_clean.csv", 'w') as f:
        f.write(",".join([str(i) for i in dataids_clean]))








