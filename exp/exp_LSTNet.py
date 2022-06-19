import numpy as np
from data.data_loader import LSTNet_dataSet
from exp.exp_basic import Exp_Basic
from util.tool import get_scaler, get_path
from models import LSTNet,MHA_Net,CNN,RNN

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import math
import time

import warnings
warnings.filterwarnings('ignore')

class Exp_LSTNet(Exp_Basic):

    def __init__(self, args):
        super(Exp_LSTNet, self).__init__(args)

    def _build_model(self, data):
        args = self.args
        model = eval(args.model).Model(args, data)  # 通过这种方式来根据参数，确认模型
        nParams = sum([p.nelement() for p in model.parameters()])
        print('number of parameters: %d' % nParams)
        if args.cuda:
            model = nn.DataParallel(model)
        return model

    def _get_data(self, path, mode='train'):
        args = self.args
        data_set = LSTNet_dataSet(path, args, mode)
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=True)
        return data_set, data_loader

    def vali(self, model, dataset, dataloader):
        args = self.args
        model.eval()
        total_loss = 0
        total_loss_l1 = 0
        n_samples = 0
        predict = None
        test = None
        for i, (batch_x, batch_y) in enumerate(dataloader):

            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float().to(args.device)

            output = model(batch_x)
            if predict is None:
                predict = output.clone().detach()
                test = batch_y
            else:
                predict = torch.cat((predict, output.clone().detach()))
                test = torch.cat((test, batch_y))

            evaluateL1, evaluateL2 = self.get_evaluate()

            scale = torch.as_tensor(dataset.scale, device = args.device,dtype=torch.float)
            scale = scale.expand(output.size(0), dataset.m)
            total_loss += float(evaluateL2(output * scale, batch_y * scale).data.item())
            total_loss_l1 += float(evaluateL1(output * scale, batch_y * scale).data.item())

            n_samples += int((output.size(0) * dataset.m))

        rse = math.sqrt(total_loss / n_samples) / dataset.rse
        rae = (total_loss_l1 / n_samples) / dataset.rae

        predict = predict.data.cpu().numpy()
        Ytest = test.data.cpu().numpy()
        sigma_p = (predict).std(axis=0)
        sigma_g = (Ytest).std(axis=0)
        mean_p = predict.mean(axis=0)
        mean_g = Ytest.mean(axis=0)
        index = (sigma_g != 0)
        correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
        correlation = (correlation[index]).mean()
        return rse, rae, correlation


    def train(self):
        args = self.args
        args.scale = get_scaler(args.data)  # 计算整个数据的scale
        # 获取训练集，测试集和验证集的地址
        train_file, val_file, test_file = get_path(args.data)

        train_data, train_loader = self._get_data(train_file)
        vali_data, vali_loader = self._get_data(val_file, mode='test')
        test_data, test_loader = self._get_data(test_file, mode='test')

        model = self._build_model(train_data)
        optim = self.makeOptimizer(model.parameters())
        criterion = self._select_criterion()

        best_val = 10000000
        try:
            print('Training start')

            total_loss = 0
            n_samples = 0
            for epoch in range(1, args.epochs + 1):
                epoch_start_time = time.time()
                model.train()
                for i, (batch_x, batch_y) in enumerate(train_loader):

                    batch_x = batch_x.float().to(args.device)
                    batch_y = batch_y.float().to(args.device)

                    optim.zero_grad()
                    output = model(batch_x)

                    scale = torch.as_tensor(train_data.scale, device = args.device,dtype=torch.float)
                    scale = scale.expand(output.size(0), train_data.m)

                    loss = criterion(output * scale, batch_y * scale)  # 损失计算按照原scale进行
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    optim.step()
                    total_loss += loss.data.item()
                    n_samples += int((output.size(0) * train_data.m))
                train_loss = total_loss / n_samples
                val_loss, val_rae, val_corr = self.vali(model, vali_data, vali_loader)
                print('| end of epoch {:3d} | time used: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.
                    format(epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr))

                if val_loss < best_val:
                    with open(args.save, 'wb') as f:
                        torch.save(model, f)
                    best_val = val_loss

                if epoch % 10 == 0:
                    test_acc, test_rae, test_corr = self.vali(model, test_data, test_loader)
                    print("| test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}\n".format(test_acc, test_rae, test_corr))
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    def test(self):
        super().test()

    def makeOptimizer(self, params):
        args = self.args
        if args.optim == 'sgd':
            optimizer = optim.SGD(params, lr=args.lr, )
        elif args.optim == 'adagrad':
            optimizer = optim.Adagrad(params, lr=args.lr, )
        elif args.optim == 'adadelta':
            optimizer = optim.Adadelta(params, lr=args.lr, )
        elif args.optim == 'adam':
            optimizer = optim.Adam(params, lr=args.lr, )
        else:
            raise RuntimeError("Invalid optim method: " + args.method)
        return optimizer

    def _select_criterion(self):
        args = self.args
        if args.L1Loss:
            criterion = nn.L1Loss(size_average=False)
        else:
            criterion = nn.MSELoss(size_average=False)
        return criterion
    def get_evaluate(self):
        evaluateL2 = nn.MSELoss(size_average=False)
        evaluateL1 = nn.L1Loss(size_average=False)
        return evaluateL1, evaluateL2