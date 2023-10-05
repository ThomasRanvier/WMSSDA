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

Using code from https://github.com/hanzhaoml/MDAN
"""

import numpy as np
import torch
import os
import time
from torch.utils.data import TensorDataset, DataLoader

from torch.autograd import Function


class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


class Encoder(torch.nn.Module):
    def __init__(self, input_dim):
        super(Encoder, self).__init__()
        self._linears = torch.nn.Sequential(
            torch.nn.BatchNorm1d(input_dim),
            torch.nn.Linear(input_dim, 128),
            torch.nn.LeakyReLU(.2, inplace=True),
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


class Method:
    def __init__(self, epochs, batch_size, lr, n_classes):
        super(Method, self).__init__()
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._n_classes = n_classes
        self._models = None
   

    def train(self, x_sources, y_sources, x_target, y_target, weighted_loss=True, domain_t=None):
        domains_s = list(x_sources.keys())
        self._models = {
            'enc': Encoder(x_target.shape[-1]),
            'clf_t': Clf(self._n_classes),
        }
        for domain_s in domains_s:
            self._models[domain_s] = {
                'clf_s': Clf(self._n_classes),
                'clf_d': Clf(2),
            }
        enc, clf_t = self._models['enc'], self._models['clf_t']
        clfs_s, clfs_d = {}, {}
        for domain_s in domains_s:
            clfs_s[domain_s] = self._models[domain_s]['clf_s']
            clfs_d[domain_s] = self._models[domain_s]['clf_d']
        use_cuda = torch.cuda.is_available()
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
        loss_domain_fn = torch.nn.CrossEntropyLoss()
        x_t, y_t = torch.Tensor(x_target), torch.LongTensor(y_target)
        if use_cuda:
            x_t, y_t = x_t.cuda(), y_t.cuda()
            enc = enc.cuda()
            clf_t = clf_t.cuda()
            for domain_s in domains_s:
                clfs_s[domain_s] = clfs_s[domain_s].cuda()
                clfs_d[domain_s] = clfs_d[domain_s].cuda()
        dl_t = DataLoader(TensorDataset(x_t, y_t), batch_size=self._batch_size, shuffle=True, drop_last=True)
        params_clfs_s, params_clfs_d = [], []
        for domain_s in domains_s:
            params_clfs_s += list(clfs_s[domain_s].parameters())
            params_clfs_d += list(clfs_d[domain_s].parameters())
        optimizer = torch.optim.Adam(list(enc.parameters()) +
                                     list(clf_t.parameters()) +
                                     params_clfs_s + params_clfs_d, lr=self._lr)
        n_batches = min([len(dl) for dl in dl_s.values()] + [len(dl_t)])
        start_training = time.time()
        for e in range(self._epochs):
            iter_batches_t = iter(dl_t)
            iter_batches_s = {}
            for domain_s in domains_s:
                iter_batches_s[domain_s] = iter(dl_s[domain_s])
            for batch in range(n_batches):
                optimizer.zero_grad()
                x_batch, y_batch = next(iter_batches_t)
                ## Feed target domain batch to encoder and then to classifier
                feats_t = enc(x_batch)
                y_probas_t = clf_t(feats_t)
                clf_losses = [loss_function_t(y_probas_t, y_batch)]
                y_domain_t = torch.ones(len(x_batch)).long()
                y_domain_s, y_probas_d_s, y_probas_d_t = {}, {}, {}
                for domain_s in domains_s:
                    ## Feed source domain batch to encoder and then to source specific classifier
                    x_batch, y_batch = next(iter_batches_s[domain_s])
                    feats_s = enc(x_batch)
                    y_probas_s = clfs_s[domain_s](feats_s)
                    ## Get domain probability outputs, backward gradient is reversed using grls
                    y_probas_d_s[domain_s] = clfs_d[domain_s](GradientReversalLayer.apply(feats_s))
                    y_probas_d_t[domain_s] = clfs_d[domain_s](GradientReversalLayer.apply(feats_t))
                    clf_losses.append(loss_function_s[domain_s](y_probas_s, y_batch))
                    y_domain_s[domain_s] = torch.zeros(len(x_batch)).long()
                clf_losses = torch.stack(clf_losses)
                ## Domain classification
                if use_cuda:
                    y_domain_t = y_domain_t.cuda()
                    for domain_s in domains_s:
                        y_domain_s[domain_s] = y_domain_s[domain_s].cuda()
                domain_losses = []
                for domain_s in domains_s:
                    ## Loss between clf_d of each source domain between source and target
                    domain_losses.append(loss_domain_fn(y_probas_d_s[domain_s], y_domain_s[domain_s]) +
                                         loss_domain_fn(y_probas_d_t[domain_s], y_domain_t))
                domain_losses = torch.stack(domain_losses)
                loss = torch.max(clf_losses) + 1e-2 * torch.min(domain_losses)
                loss.backward()
                optimizer.step()
            if e == 0 or (e + 1) % 10 == 0:
                time_elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_training))
                print(f'Epoch {e+1}/{self._epochs} - Time elapsed {time_elapsed}', flush=True)


    def predict(self, x):
        enc, clf = self._models['enc'], self._models['clf_t']
        use_cuda = torch.cuda.is_available()
        x = torch.Tensor(x)
        if use_cuda:
            x = x.cuda()
            enc = enc.cuda()
            clf = clf.cuda()
        enc.eval()
        clf.eval()
        with torch.no_grad():
            y_probas = clf(enc(x))
        enc.train()
        clf.train()
        return y_probas.cpu()