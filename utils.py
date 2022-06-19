import torch
import numpy as np
from torch.autograd import Variable


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))

# TODO 优化数据读取
class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, device, args):
        self.device = device
        self.P = args.window
        self.h = args.horizon
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',')
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.scale = np.ones(self.m)
        self._normalized(args.normalize) # 根据指定规则，归一化输入数据
        self._split(int(train * self.n), int((train + valid) * self.n), self.n) # 获取划分后的数据，数据格式为 X：[n_train, window_size, horizon]，同样还有val和test的数据

        self.scale = torch.as_tensor(self.scale, device=device, dtype=torch.float)

        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)  # 将test数据转换为未标准化数据

        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp) # 计算label的标准差 raw suqre error
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp))) # raw absolute error
        fin.close()

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train, valid, test):
        """
        划分训练集，验证集，测试集
        :param train:训练集的最后一个idx
        :param valid:验证集的最后一个idx
        :param test:整个数据集的Length
        """
        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):
        """
        获取 batch_size 数据，这里一次性创造所有数据的张量,容易导致内存爆炸，因此建议改为dataload加载
        另外，很具其运算机制，可以看出来，这里的输入，是 windows大小的数据，label仅是， windows+horizon时刻的数值，并不是一个序列
        :param idx_set:
        :param horizon:
        :return:
        """
        n = len(idx_set)
        # print(n, self.P, self.m, self.device)
        X = torch.zeros((n, self.P, self.m), device=self.device) # 内存超标的原因，一次性把所有数据都加载进来了
        Y = torch.zeros((n, self.m), device=self.device)

        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.as_tensor(self.dat[start:end, :], device=self.device)
            Y[i, :] = torch.as_tensor(self.dat[idx_set[i], :], device=self.device)

        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length,device=self.device)
        else:
            index = torch.as_tensor(range(length),device=self.device,dtype=torch.long)
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]

            yield Variable(X), Variable(Y)
            start_idx += batch_size
