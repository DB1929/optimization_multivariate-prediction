import numpy as np
import torch


class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean


def get_scaler(file_path):
    fin = open(file_path)
    data = np.loadtxt(fin, delimiter=',')
    m, n = data.shape
    scale = np.zeros(n)
    for i in range(n):
        scale[i] = np.max(np.abs(data[:, i]))
    fin.close()
    return scale

def get_path(file_path):
    file_head = file_path.split('.')[0]
    train_file = file_head + "_train" + ".txt"
    val_file = file_head + "_val" + ".txt"
    test_file = file_head + "_test" + ".txt"
    return train_file, val_file, test_file