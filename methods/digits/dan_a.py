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

Using code from https://github.com/CuthbertCai/pytorch_DAN
"""

import numpy as np
import torch
import os
import time
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from functools import partial


class Encoder(torch.nn.Module):
    def __init__(self, in_c):
        super(Encoder, self).__init__()
        ## Convs
        self._convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_c, 32, kernel_size=3, padding=1, stride=2),# 32x32 => 16x16 # 28x28 => 14x14
            torch.nn.LeakyReLU(.2, inplace=True),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(.2, inplace=True),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),# 16x16 => 8x8 # 14x14 => 7x7
            torch.nn.LeakyReLU(.2, inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(.2, inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),# 8x8 => 4x4 # 7x7 => 4x4
            torch.nn.LeakyReLU(.2, inplace=True),
        ])
        ## Linear output
        in_f = 64 * 4 * 4
        self._features = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(in_f),
            torch.nn.Dropout(.2),
            torch.nn.Linear(in_f, 512),
            torch.nn.LeakyReLU(.2, inplace=True),
        ])
            
    def forward(self, x):
        for l in self._convs:
            x = l(x)
        f = torch.flatten(x, start_dim=1)
        for l in self._features:
            f = l(f)
        return f


class Clf(torch.nn.Module):
    def __init__(self, output_dim):
        super(Clf, self).__init__()
        self._l1 = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 128),
            torch.nn.LeakyReLU(.2, inplace=True),
        ])
        self._l2 = torch.nn.ModuleList([
            torch.nn.Linear(128, output_dim),
            torch.nn.Softmax(dim=-1),
        ])
    
    def forward(self, x):
        for l in self._l1:
            x = l(x)
        x1 = x
        for l in self._l2:
            x = l(x)
        return [x1, x]


def pairwise_distance(x, y):
    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')
    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)
    return output


def gaussian_kernel_matrix(x, y, sigmas):
    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1. / (2. * sigmas)
    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)
    s = torch.matmul(beta, dist_)
    return torch.sum(torch.exp(-s), 0).view_as(dist)


def mmd(x, y, kernel= gaussian_kernel_matrix):
    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))
    return cost


def mmd_loss(source_features, target_features):
    use_cuda = torch.cuda.is_available()
    sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
    if use_cuda:
        gaussian_kernel = partial(gaussian_kernel_matrix, sigmas = Variable(torch.cuda.FloatTensor(sigmas)))
    else:
        gaussian_kernel = partial(gaussian_kernel_matrix, sigmas = Variable(torch.FloatTensor(sigmas)))
    return mmd(source_features, target_features, kernel= gaussian_kernel)


class Method:
    def __init__(self, epochs, batch_size, lr, in_c, n_classes):
        super(Method, self).__init__()
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._in_c = in_c
        self._n_classes = n_classes
        self._models = None
   

    def train(self, x_source, y_source, x_target, y_target, weighted_loss=True):
        self._models = {
            'enc': Encoder(self._in_c),
            'clf_s': Clf(self._n_classes),
            'clf_t': Clf(self._n_classes),
        }
        encoder, clf_s, clf_t = self._models['enc'], self._models['clf_s'], self._models['clf_t']
        use_cuda = torch.cuda.is_available()
        ## Put model on GPU if used
        if use_cuda:
            encoder, clf_s, clf_t = encoder.cuda(), clf_s.cuda(), clf_t.cuda()
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(clf_s.parameters()) + list(clf_t.parameters()), lr=self._lr)
        ## Losses definition
        ce_weights = None
        if weighted_loss:
            ce_weights = torch.Tensor(y_source.shape[0] / (self._n_classes * np.bincount(y_source)))
            if use_cuda:
                ce_weights = ce_weights.cuda()
        loss_s_class_fn = torch.nn.CrossEntropyLoss(weight=ce_weights)
        ce_weights = None
        if weighted_loss:
            ce_weights = torch.Tensor(y_target.shape[0] / (self._n_classes * np.bincount(y_target)))
            if use_cuda:
                ce_weights = ce_weights.cuda()
        loss_t_class_fn = torch.nn.CrossEntropyLoss(weight=ce_weights)
        ## Create dataloaders
        x_s, y_s, x_t, y_t = (torch.Tensor(x_source), torch.LongTensor(y_source),
                              torch.Tensor(x_target), torch.LongTensor(y_target))
        if use_cuda:
            x_s, y_s, x_t, y_t = x_s.cuda(), y_s.cuda(), x_t.cuda(), y_t.cuda()
        dataloader_source = DataLoader(TensorDataset(x_s, y_s), batch_size=self._batch_size, shuffle=True, drop_last=True)
        dataloader_target = DataLoader(TensorDataset(x_t, y_t), batch_size=self._batch_size, shuffle=True, drop_last=True)
        ## Actual training
        n_batches = min(len(dataloader_source), len(dataloader_target))
        start_training = time.time()
        for e in range(self._epochs):
            iter_batches_s = iter(dataloader_source)
            iter_batches_t = iter(dataloader_target)
            for batch in range(n_batches):
                optimizer.zero_grad()
                ## Feed source batch
                x_s_batch, y_s_batch = next(iter_batches_s)
                feat_s = encoder(x_s_batch)
                outputs_s = clf_s(feat_s)
                clf_s_loss = loss_s_class_fn(outputs_s[-1], y_s_batch)
                ## Feed target batch
                x_t_batch, y_t_batch = next(iter_batches_t)
                feat_t = encoder(x_t_batch)
                outputs_t = clf_t(feat_t)
                clf_t_loss = loss_t_class_fn(outputs_t[-1], y_t_batch)
                ## MMD regularization
                mmd_reg = mmd_loss(feat_s, feat_t)
                for i in range(len(outputs_s) - 1):
                    mmd_reg += mmd_loss(outputs_s[i], outputs_t[i])
                mmd_reg /= i + 1
                ## Loss
                loss = mmd_reg + ((.5 * clf_s_loss + clf_t_loss) / 2)
                loss.backward()
                optimizer.step()
            if e == 0 or (e + 1) % 10 == 0:
                time_elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_training))
                print(f'Epoch {e+1}/{self._epochs} - Time elapsed {time_elapsed}', flush=True)


    def predict(self, x):
        encoder, clf = self._models['enc'], self._models['clf_t']
        use_cuda = torch.cuda.is_available()
        x = torch.Tensor(x)
        if use_cuda:
            x = x.cuda()
            encoder, clf = encoder.cuda(), clf.cuda()
        encoder.eval()
        clf.eval()
        with torch.no_grad():
            outputs = clf(encoder(x))
            y_probas = outputs[-1]
        encoder.train()
        clf.train()
        return y_probas.cpu()