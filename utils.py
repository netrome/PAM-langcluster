import numpy as np
import csv

def get_slices(data_file):
    # Get the features
    np_data = np.load(data_file)
    meta_reader = csv.reader(open(data_file.replace(".npy", ".csv"), "r"))
    n = len(np_data)

    slices = []
    meta = [next(meta_reader)]

    # Preprocess the features
    for k in range(n):
        tmp = np_data[k]
        next_csv = next(meta_reader)

        start = range(0, len(tmp) + 1, 50)
        stop = start[2:]
        L = [tmp[i : j] for i, j in zip(start, stop)]
        meta += [next_csv for i, j in zip(start, stop)]
        slices += L

    slices = np.array(slices)
    n = len(slices)

    #slices = slices.reshape(list(slices.shape) + [1]) 
    return n, slices, meta
