"""
MIT License

WMSSDA: Weighted Multi-Source Supervised Domain Adaptation

Copyright (c) 2022 Thomas RANVIER

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Using code from https://github.com/easezyc/deep-transfer-learning/blob/master/MUDA/MFSAN
"""

import numpy as np
import torch
import os
import time
from torch.utils.data import TensorDataset, DataLoader

from torch.autograd import Variable


class Encoder(torch.nn.Module):
    def __init__(self, input_dim):
        super(Encoder, self).__init__()
        self._linears = torch.nn.Sequential(
            torch.nn.BatchNorm1d(input_dim),
            torch.nn.Linear(input_dim, 128),
            torch.nn.LeakyReLU(.2, inplace=True),
        )
        
    def forward(self, x):
        return self._linears(x)


class Neck(torch.nn.Module):
    def __init__(self):
        super(Neck, self).__init__()
        self._linears = torch.nn.Sequential(
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(.2),
            torch.nn.Linear(128, 32),
            torch.nn.LeakyReLU(.2, inplace=True),
        )
            
    def forward(self, x):
        return self._linears(x)


class Clf(torch.nn.Module):
    def __init__(self, output_dim):
        super(Clf, self).__init__()
        self._linears = torch.nn.Sequential(
            torch.nn.Linear(32, output_dim),
            torch.nn.Softmax(dim=-1),
        )
            
    def forward(self, x):
        return self._linears(x)


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = gaussian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss


class Method:
    def __init__(self, epochs, batch_size, lr, n_classes):
        super(Method, self).__init__()
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._n_classes = n_classes
        self._models = None
        self._domains_s = None
   

    def train(self, x_sources, y_sources, x_target, y_target, weighted_loss=True, domain_t=None):
        self._domains_s = list(x_sources.keys())
        self._models = {
            'enc': Encoder(x_target.shape[-1]),
        }
        for domain_s in self._domains_s:
            self._models[domain_s] = {
                'neck': Neck(),
                'clf': Clf(self._n_classes),
            }
        enc = self._models['enc']
        necks, clfs = {}, {}
        for domain_s in self._domains_s:
            necks[domain_s] = self._models[domain_s]['neck']
            clfs[domain_s] = self._models[domain_s]['clf']
        use_cuda = torch.cuda.is_available()
        loss_function_s, dl_s = {}, {}
        for domain_s in self._domains_s:
            ce_weights = None
            if weighted_loss:
                ce_weights = torch.Tensor(y_sources[domain_s].shape[0] / (self._n_classes * np.bincount(y_sources[domain_s])))
                if use_cuda:
                    ce_weights = ce_weights.cuda()
            loss_function_s[domain_s] = torch.nn.CrossEntropyLoss(weight=ce_weights)
            x, y = torch.Tensor(x_sources[domain_s]), torch.LongTensor(y_sources[domain_s])
            if use_cuda:
                x, y = x.cuda(), y.cuda()
            dl_s[domain_s] = DataLoader(TensorDataset(x, y), batch_size=self._batch_size, shuffle=True, drop_last=True)
        ce_weights = None
        if weighted_loss:
            ce_weights = torch.Tensor(y_target.shape[0] / (self._n_classes * np.bincount(y_target)))
            if use_cuda:
                ce_weights = ce_weights.cuda()
        loss_function_t = torch.nn.CrossEntropyLoss(weight=ce_weights)
        x_t, y_t = torch.Tensor(x_target), torch.LongTensor(y_target)
        if use_cuda:
            x_t, y_t = x_t.cuda(), y_t.cuda()
            enc = enc.cuda()
            for domain_s in self._domains_s:
                necks[domain_s] = necks[domain_s].cuda()
                clfs[domain_s] = clfs[domain_s].cuda()
        dl_t = DataLoader(TensorDataset(x_t, y_t), batch_size=self._batch_size, shuffle=True, drop_last=True)
        params_clfs, params_necks = [], []
        for domain_s in self._domains_s:
            params_necks += list(necks[domain_s].parameters())
            params_clfs += list(clfs[domain_s].parameters())
        optimizer = torch.optim.Adam(list(enc.parameters()) +
                                     params_necks + params_clfs, lr=self._lr)
        n_batches = min([len(dl) for dl in dl_s.values()] + [len(dl_t)])
        start_training = time.time()
        for e in range(self._epochs):
            iter_batches_t = iter(dl_t)
            iter_batches_s = {}
            for domain_s in self._domains_s:
                iter_batches_s[domain_s] = iter(dl_s[domain_s])
            for batch in range(n_batches):
                x_t_batch, y_t_batch = next(iter_batches_t)
                for domain_s in x_sources.keys():
                    optimizer.zero_grad()
                    ## Feed target batch through all sources necks and clfs
                    feat_t = enc(x_t_batch)
                    y_probas_t_mean = []
                    necks_t, y_probas_t = {}, {}
                    for _domain in x_sources.keys():
                        necks_t[_domain] = necks[_domain](feat_t)
                        y_probas_t[_domain] = clfs[_domain](necks_t[_domain])
                        y_probas_t_mean.append(y_probas_t[_domain])
                    y_probas_t_mean = torch.mean(torch.stack(y_probas_t_mean), 0)
                    ## Feed source batch
                    x_s_batch, y_s_batch = next(iter_batches_s[domain_s])
                    feat_s = enc(x_s_batch)
                    neck_s = necks[domain_s](feat_s)
                    y_probas_s = clfs[domain_s](neck_s)
                    ## Classifier loss on source + target
                    loss_cls = (loss_function_s[domain_s](y_probas_s, y_s_batch) +
                                loss_function_t(y_probas_t_mean, y_t_batch)) / 2
                    ## MMD regularization
                    gamma = 2 / (1 + np.exp(-10 * (e) / self._epochs)) - 1
                    mmd_reg = gamma * mmd(neck_s, necks_t[domain_s])
                    ## L1 regularization
                    l1_reg = 0
                    for _domain in x_sources.keys():
                        if _domain != domain_s:
                            l1_reg += torch.mean(torch.abs(y_probas_t[domain_s] - y_probas_t[_domain]))
                    l1_reg *= gamma
                    ## Compute total loss
                    loss = loss_cls + mmd_reg + l1_reg
                    loss.backward()
                    optimizer.step()
            if e == 0 or (e + 1) % 10 == 0:
                time_elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_training))
                print(f'Epoch {e+1}/{self._epochs} - Time elapsed {time_elapsed}', flush=True)


    def predict(self, x):
        enc = self._models['enc']
        necks, clfs = {}, {}
        for domain_s in self._domains_s:
            necks[domain_s] = self._models[domain_s]['neck']
            clfs[domain_s] = self._models[domain_s]['clf']
        use_cuda = torch.cuda.is_available()
        x = torch.Tensor(x)
        if use_cuda:
            x = x.cuda()
            enc = enc.cuda()
            for domain_s in necks.keys():
                necks[domain_s] = necks[domain_s].cuda()
                clfs[domain_s] = clfs[domain_s].cuda()
        enc.eval()
        for domain_s in necks.keys():
            necks[domain_s].eval()
            clfs[domain_s].eval()
        y_probas = None
        with torch.no_grad():
            feat_t = enc(x)
            y_probas_mean = []
            for domain_s in necks.keys():
                y_probas_mean.append(clfs[domain_s](necks[domain_s](feat_t)))
            y_probas = torch.mean(torch.stack(y_probas_mean), 0)
        enc.train()
        for domain_s in necks.keys():
            necks[domain_s].train()
            clfs[domain_s].train()
        return y_probas.cpu()