import numpy as np
import pandas as pd
file_name_list = ["electricity.txt", "exchange_rate.txt", "solar_AL.txt", "traffic.txt"]

p_train = 0.6
p_val = 0.2
p_test = 0.2

for file_name in file_name_list:
    file_head = file_name.split('.')[0]
    train_file =  file_head + "_train" + ".txt"
    val_file = file_head + "_val" + ".txt"
    test_file = file_head + "_test" + ".txt"
    fin = open(file_name)
    rawdat = np.loadtxt(fin, delimiter=',')
    m, n = rawdat.shape
    end_train_idx = int(m * p_train)
    end_val_idx = int(m * (1 - p_test))
    train_data = rawdat[:end_train_idx, :]
    val_data = rawdat[end_train_idx:end_val_idx, :]
    test_data = rawdat[end_val_idx:, :]
    np.savetxt(train_file, train_data, '%0.6f')
    np.savetxt(val_file, val_data, '%0.6f')
    np.savetxt(test_file, test_data, '%0.6f')
    fin.close()