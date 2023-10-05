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

Using code from https://github.com/VisionLearningGroup/VisionLearningGroup.github.io/blob/master/M3SDA
"""

import numpy as np
import torch
import os
import time
from torch.utils.data import TensorDataset, DataLoader


class M3SDA(torch.nn.Module):
    def __init__(self, in_c, output_dim):
        super(M3SDA, self).__init__()
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
            torch.nn.Linear(in_f, 256),
            torch.nn.LeakyReLU(.2, inplace=True),
        ])
        self._clf = torch.nn.ModuleList([
            torch.nn.Linear(256, output_dim),
            torch.nn.Softmax(dim=-1),
        ])
            
    def forward(self, x):
        for l in self._convs:
            x = l(x)
        f = torch.flatten(x, start_dim=1)
        for l in self._features:
            f = l(f)
        x = f
        for l in self._clf:
            x = l(x)
        return f, x


def euclidean(x1,x2):
    return ((x1-x2)**2).sum().sqrt()


def k_moment(feats, k):
    fs = {}
    for domain in feats.keys():
        fs[domain] = (feats[domain]**k).mean(0)
    moment = 0
    for domain_s1 in feats.keys():
        for domain_s2 in feats.keys():
            if domain_s1 != domain_s2:
                moment += euclidean(fs[domain_s1], fs[domain_s2])
    return moment


def msda_regulizer(feats, belta_moment):
    fs = {}
    for domain in feats.keys():
        fs[domain] = feats[domain] - feats[domain].mean(0)
    moment = 0
    for domain_s1 in feats.keys():
        for domain_s2 in feats.keys():
            if domain_s1 != domain_s2:
                moment += euclidean(fs[domain_s1], fs[domain_s2])
    for i in range(belta_moment-1):
        moment += k_moment(feats, i+2)
    return moment


class Method:
    def __init__(self, epochs, batch_size, lr, in_c, n_classes):
        super(Method, self).__init__()
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._in_c = in_c
        self._n_classes = n_classes
        self._models = None
   

    def train(self, x_sources, y_sources, x_target, y_target, weighted_loss=True, domain_t=None):
        self._models = {
            'model': M3SDA(self._in_c, self._n_classes),
        }
        model = self._models['model']
        use_cuda = torch.cuda.is_available()
        optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)
        domains_s = list(x_sources.keys())
        loss_function_s, dl_s = {}, {}
        for domain_s in domains_s:
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
            model = model.cuda()
        dl_t = DataLoader(TensorDataset(x_t, y_t), batch_size=self._batch_size, shuffle=True, drop_last=True)
        start_training = time.time()
        for e in range(self._epochs):
            total_loss = 0
            n_batches = min([len(dl) for dl in dl_s.values()] + [len(dl_t)])
            iter_batches_t = iter(dl_t)
            iter_batches = {}
            for domain_s in domains_s:
                iter_batches[domain_s] = iter(dl_s[domain_s])
            for batch in range(n_batches):
                optimizer.zero_grad()
                x_batch, y_batch = next(iter_batches_t)
                feats = {}
                feats[domain_t], output_t = model(x_batch)
                loss = loss_function_t(output_t, y_batch)
                for domain_s in domains_s:
                    x_batch, y_batch = next(iter_batches[domain_s])
                    feats[domain_s], output_s = model(x_batch)
                    loss_s = loss_function_s[domain_s](output_s, y_batch)
                    loss += loss_s
                loss /= len(domains_s) + 1
                moment_dist_regu = (1./12.) * msda_regulizer(feats, len(domains_s) + 1) / self._batch_size**2
                loss += moment_dist_regu
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if e == 0 or (e + 1) % 10 == 0:
                time_elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_training))
                print(f'Epoch {e+1}/{self._epochs} - Time elapsed {time_elapsed}', flush=True)


    def predict(self, x):
        model = self._models['model']
        use_cuda = torch.cuda.is_available()
        x = torch.Tensor(x)
        if use_cuda:
            x = x.cuda()
            model = model.cuda()
        model.eval()
        with torch.no_grad():
            _, y_probas = model(x)
        model.train()
        return y_probas.cpu()