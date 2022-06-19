__author__ = "Guan Song Wang"

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.P = args.window
        self.m = data.m
        self.hidR = args.hidRNN
        self.hidC = args.hidCNN
        self.hidS = args.hidSkip
        self.Ck = args.CNN_kernel
        self.skip = args.skip
        self.device = args.device

        self.hw = args.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=args.dropout)
        if (self.skip > 0):  # skip连接额外的参数
            self.pt = (self.P - self.Ck) / self.skip # 这里计算的是卷积后的序列长度有多少个长度为self.skip的周期，也就是说这里计算包含了序列中所有历史周期
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        if (self.hw > 0):  # AR的参数
            self.highway = nn.Linear(self.hw, 1)
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh

    def forward(self, x):
        batch_size = x.size(0)

        # CNN 卷积必须针对四维张量，所以需要扩充维度
        c = x.view(-1, 1, self.P, self.m)

        c = c.type(torch.FloatTensor).to(self.device) # RuntimeError: Input type (torch.cuda.DoubleTensor) and weight type (torch.cuda.FloatTensor) should be the same

        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3) # [batch_size, feature, sequence_length]  卷积前后大小计算 w_new = (w-k_size+1)/s

        # RNN RNN必须处理三维数据
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r) # RNN的参数 batch_first default是false，因此输入数据的格式是：(seq, batch, feature)（输出同理） 由于这里只需要最后一个h_t，所以第一个输出不要

        r = self.dropout(torch.squeeze(r, 0))

        # skip-rnn

        if (self.skip > 0):
            self.pt=int(self.pt)
            s = c[:, :, int(-self.pt * self.skip):].contiguous() # 将所有以skip为周期的数据点都保留

            s = s.view(batch_size, self.hidC, self.pt, self.skip) # 这些周期数据的序列长度是周期个数（self.pt），特征长度是self.hidC
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS) # 还原回batch_size形式
            s = self.dropout(s)
            r = torch.cat((r, s), 1)

        res = self.linear1(r)

        # highway 模型线性AR AR处理的是一维数据，所以这里相当于把多维特征变成一个一个的特征去处理，最后再合并
        if (self.hw > 0):

            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)

            z = self.highway(z.to(torch.float32))

            z = z.view(-1, self.m)
            res = res + z

        if (self.output):
            res = self.output(res)
        return res
