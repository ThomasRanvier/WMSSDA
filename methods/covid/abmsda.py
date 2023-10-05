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


class DomainRecognizer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DomainRecognizer, self).__init__()
        self._linears = torch.nn.Sequential(
            torch.nn.BatchNorm1d(input_dim),
            torch.nn.Linear(input_dim, 128),
            torch.nn.LeakyReLU(.2, inplace=True),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(.2),
            torch.nn.Linear(128, 32),
            torch.nn.LeakyReLU(.2, inplace=True),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(.2),
            torch.nn.Linear(32, output_dim),
            torch.nn.Softmax(dim=-1),
        )
        
    def forward(self, x):
        return self._linears(x)


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


def wmd(feats_s, feats_t, weights):
    moment = 0
    for k in [1, 2]:
        for domain_s in feats_s.keys():
            moment += weights[domain_s] * ((feats_s[domain_s]**k - feats_t**k)**2).sum().sqrt()
    return moment / len(feats_s)


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
            'clf': Clf(self._n_classes),
            'drm': DomainRecognizer(x_target.shape[-1], len(y_sources)),
        }
        enc = self._models['enc']
        clf = self._models['clf']
        drm = self._models['drm']
        use_cuda = torch.cuda.is_available()
        optimizer = torch.optim.Adam(list(enc.parameters()) + list(clf.parameters()) + list(drm.parameters()), lr=self._lr)
        domains_s = list(y_sources.keys())
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
        loss_function_d = torch.nn.CrossEntropyLoss()
        x_t, y_t = torch.Tensor(x_target), torch.LongTensor(y_target)
        if use_cuda:
            x_t, y_t = x_t.cuda(), y_t.cuda()
            enc = enc.cuda()
            clf = clf.cuda()
            drm = drm.cuda()
        dl_t = DataLoader(TensorDataset(x_t, y_t), batch_size=self._batch_size, shuffle=True, drop_last=True)
        n_batches = min([len(dl) for dl in dl_s.values()] + [len(dl_t)])
        start_training = time.time()
        for e in range(self._epochs):
            iter_batches_t = iter(dl_t)
            iter_batches = {}
            for domain_s in domains_s:
                iter_batches[domain_s] = iter(dl_s[domain_s])
            for batch in range(n_batches):
                optimizer.zero_grad()
                x_batch, y_batch = next(iter_batches_t)
                feats_t = enc(x_batch)
                y_probas_t = clf(feats_t)
                loss_clf_t = loss_function_t(y_probas_t, y_batch)
                feats_s, loss_clf_s, weights = {}, {}, {}
                reg_dmr = 0
                for i, domain_s in enumerate(domains_s):
                    x_batch, y_batch = next(iter_batches[domain_s])
                    feats_s[domain_s] = enc(x_batch)
                    y_probas_s = clf(feats_s[domain_s])
                    y_probas_d = drm(x_batch)
                    y_domain = torch.zeros(len(x_batch)).long() + i
                    if use_cuda:
                        y_domain = y_domain.cuda()
                    loss_clf_s[domain_s] = loss_function_s[domain_s](y_probas_s, y_batch)
                    reg_dmr += loss_function_d(y_probas_d, y_domain)
                    ## Compute attention weight given domain recognizer output
                    hat_dn = torch.argmax(y_probas_d, -1) ## Index of predicted domain for each element of batch
                    weights[domain_s] = torch.sum(hat_dn==i) / len(x_batch) ## wi = sum(sign(hat_dn, i)) / b
                reg_dmr /= i + 1
                ## Apply attention weights to clf loss
                loss_clf = loss_clf_t
                for domain_s in domains_s:
                    loss_clf += weights[domain_s] * loss_clf_s[domain_s]
                ## WMD regularization
                reg_wmd = .002 * wmd(feats_s, feats_t, weights)
                ## Compute total loss
                loss = loss_clf + reg_wmd + reg_dmr
                loss.backward()
                optimizer.step()
            if e == 0 or (e + 1) % 10 == 0:
                time_elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_training))
                print(f'Epoch {e+1}/{self._epochs} - Time elapsed {time_elapsed}', flush=True)


    def predict(self, x):
        enc = self._models['enc']
        clf = self._models['clf']
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