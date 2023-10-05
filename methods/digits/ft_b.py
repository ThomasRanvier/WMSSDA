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
"""

import numpy as np
import torch
import os
import time
from torch.utils.data import TensorDataset, DataLoader


class Conv(torch.nn.Module):
    def __init__(self, in_c, output_dim):
        super(Conv, self).__init__()
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
        self._linears = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(in_f),
            torch.nn.Dropout(.2),
            torch.nn.Linear(in_f, output_dim),
            torch.nn.Softmax(dim=-1),
        ])
            
    def forward(self, x):
        for l in self._convs:
            x = l(x)
        x = torch.flatten(x, start_dim=1)
        for l in self._linears:
            x = l(x)
        return x


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
            'model': Conv(self._in_c, self._n_classes),
        }
        model = self._models['model']
        use_cuda = torch.cuda.is_available()
        ## Pre-training on source domain for half as many epochs as for fine-tuning
        optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)
        loss_function_s, dl_s = {}, {}
        for domain_s in x_sources.keys():
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
        if use_cuda:
            model = model.cuda()
        start_training = time.time()
        for e in range(self._epochs // 3):
            total_loss = 0
            n_batches = min([len(dl) for dl in dl_s.values()])
            iter_batches = {}
            for domain_s in x_sources.keys():
                iter_batches[domain_s] = iter(dl_s[domain_s])
            for b in range(n_batches):
                for domain_s in x_sources.keys():
                    x_batch, y_batch = next(iter_batches[domain_s])
                    optimizer.zero_grad()
                    out = model(x_batch)
                    loss = loss_function_s[domain_s](out, y_batch)
                    loss.backward()
                    total_loss += loss.item()
                    optimizer.step()
            if e == 0 or (e + 1) % 10 == 0:
                time_elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_training))
                print(f'Epoch {e+1}/{self._epochs // 3} - Time elapsed {time_elapsed}', flush=True)
        ## Fine-tuning on target domain
        optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)
        x, y = torch.Tensor(x_target), torch.LongTensor(y_target)
        ce_weights = None
        if weighted_loss:
            ce_weights = torch.Tensor(y.shape[0] / (self._n_classes * np.bincount(y)))
            if use_cuda:
                ce_weights = ce_weights.cuda()
        loss_function = torch.nn.CrossEntropyLoss(weight=ce_weights)
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        dl = DataLoader(TensorDataset(x, y), batch_size=self._batch_size, shuffle=True, drop_last=True)
        start_training = time.time()
        for e in range((2 * self._epochs) // 3):
            total_loss = 0
            for b, (x_batch, y_batch) in enumerate(dl):
                optimizer.zero_grad()
                out = model(x_batch)
                loss = loss_function(out, y_batch)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
            if e == 0 or (e + 1) % 10 == 0:
                time_elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_training))
                print(f'Epoch {e+1}/{(2 * self._epochs) // 3} - Time elapsed {time_elapsed}', flush=True)


    def predict(self, x):
        model = self._models['model']
        use_cuda = torch.cuda.is_available()
        x = torch.Tensor(x)
        if use_cuda:
            x = x.cuda()
            model = model.cuda()
        model.eval()
        with torch.no_grad():
            y_probas = model(x)
        model.train()
        return y_probas.cpu()