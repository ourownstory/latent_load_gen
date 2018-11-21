import numpy as np
import pandas as pd
from datetime import datetime
import os
from shutil import copyfile


run_raw_processing = True
run_split = True
save_in_one = True
num_days = 365
daily_resolution = 24
remove_missing_threshold = 3
split = {"train": 0.8, "val": 0.1, "test": 0.1}
path_raw = "data/raw/"
path_processed = "data/processed/"
path_split = "data/split"
np.random.seed(0)

if run_raw_processing:
    raw = pd.read_csv(path_raw + "dataport-export_hourly_use_car_EV-houses.csv")
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
            directory = path_processed + "/" + str(dataid) + "/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            df.to_csv(directory + "{}.csv".format(day), index=False)
            # df = pd.DataFrame(car[i, i, :])
            # df.to_csv(data_out + "car_{}.csv".format(dataid), index=False)

    with open(path_raw + "dataids.csv", 'w') as f:
        f.write(",".join([str(i) for i in dataids]))

    df = pd.DataFrame(use_missing)
    df.to_csv(path_raw + "use_missing.csv", index=False)
    df = pd.DataFrame(car_missing)
    df.to_csv(path_raw + "car_missing.csv", index=False)

    # clean data
    dataids = pd.read_csv(path_raw + "dataids.csv", header=None).values[0]
    print(dataids)
    use_missing = pd.read_csv(path_raw + "use_missing.csv").values
    use_missing_average = use_missing.mean(-1)
    # print(use_missing_average.astype('int'))
    # remove all data_ids with more than x missing hourly measurements (on average)
    dataids_clean = dataids[use_missing_average < remove_missing_threshold]
    print(dataids_clean)
    with open(path_raw + "dataids_clean.csv", 'w') as f:
        f.write(",".join([str(i) for i in dataids_clean]))

if run_split:
    dataids_clean = pd.read_csv(os.path.join(path_raw, "dataids_clean.csv"), header=None).values[0]
    np.random.shuffle(dataids_clean)
    val_idx = int(len(dataids_clean) * split["train"])
    test_idx = int(len(dataids_clean) * (split["train"] + split["val"]))
    # print(val_idx)
    # print(test_idx)
    split_dataids = {
        'train': dataids_clean[:val_idx],
        'val': dataids_clean[val_idx:test_idx],
        'test': dataids_clean[test_idx:]
        }
    print(split_dataids)
    for mode, dataids in split_dataids.items():
        mode_dir = os.path.join(path_split, mode)
        # print(mode_dir)
        if not os.path.exists(mode_dir):
            os.makedirs(mode_dir)
        for dataid in [str(x) for x in dataids]:
            subdir = os.path.join(mode_dir, dataid)
            # print(subdir)
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # print(os.path.join(path_processed, dataid))
            # print(next(os.walk(os.path.join(path_processed, dataid)))[2])
            for file_name in next(os.walk(os.path.join(path_processed, dataid)))[2]:
                src = os.path.join(path_processed, dataid, file_name)
                dst = os.path.join(subdir, file_name)
                # print(src, dst)
                copyfile(src, dst)

if save_in_one:
    root_dir = "data/split"
    for mode in ["test", "val", "train"]:
        split_dir = os.path.join(root_dir, mode)
        data_ids = sorted([int(x) for x in next(os.walk(split_dir))[1]])
        csv_list = []
        # print(root_dir, data_ids)
        for subdir in [str(x) for x in data_ids]:
            # print(next(os.walk(os.path.join(root_dir, subdir)))[2])
            for file_name in next(os.walk(os.path.join(split_dir, subdir)))[2]:
                csv_list.append((subdir, file_name))

        columns = {"use": [], "car": []}
        for idx in range(len(csv_list)):
            csv = pd.read_csv(os.path.join(split_dir, *csv_list[idx]))
            for col in columns.keys():
                columns[col].append(csv[col].values)
        for col in columns.keys():
            file_name = os.path.join(split_dir, "{}.csv".format(col))
            pd.DataFrame(columns[col]).to_csv(file_name, index=False)












