import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class LSTNet_dataSet(Dataset):
    def __init__(self, path, args, mode):
        self.path = path
        self.P = args.window
        self.h = args.horizon
        self.mode = mode
        self.scale = args.scale
        self.normalize = args.normalize
        self.__read_data__()
    def __read_data__(self):
        # fin = open(self.path)
        # rawdat = np.loadtxt(fin, delimiter=',')
        rawdat = np.array(pd.read_csv(self.path, sep=" ", header=None))
        self.n, self.m = rawdat.shape
        # TODO if mode == "test": 新增两个属性 rse rae
        if self.mode == "test":
            self.rse = rawdat.std() * np.sqrt((self.n - 1.) / self.n)  # 计算label的标准差 raw suqre error
            self.rae = np.mean(np.abs(rawdat - np.mean(rawdat)))  # raw absolute error
        self.dat = _normalized(rawdat, self.normalize, self.scale)
        # fin.close()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.P
        r_idx = s_end + self.h - 1
        x = self.dat[s_begin:s_end]
        y = self.dat[r_idx]
        return x, y

    def __len__(self):
        # 总共的数据个数
        return len(self.dat) - self.P - self.h + 1

def _normalized(dat, normalize, scale):
    # normalized by the maximum value of entire matrix.
    m = dat.shape[1]

    if (normalize == 0):
        dat = dat

    if (normalize == 1):
        dat = dat / np.max(scale)

    # normlized by the maximum value of each row(sensor).

    if (normalize == 2):
        for i in range(m):
            dat[:, i] = dat[:, i] / scale[i]
    return dat