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

Using code from https://github.com/fungtion/DANN
"""

import numpy as np
import torch
import os
import time
from torch.utils.data import TensorDataset, DataLoader

from torch.autograd import Function


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DANN(torch.nn.Module):
    def __init__(self, input_dim, n_classes, n_domains):
        super(DANN, self).__init__()
        self._linears = torch.nn.Sequential(
            torch.nn.BatchNorm1d(input_dim),
            torch.nn.Linear(input_dim, 128),
            torch.nn.LeakyReLU(.2, inplace=True),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(.2),
            torch.nn.Linear(128, 32),
            torch.nn.LeakyReLU(.2, inplace=True),
        )
        ## Class classififer
        self._class_clf = torch.nn.Sequential(
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(.2),
            torch.nn.Linear(32, n_classes),
            torch.nn.Softmax(dim=-1),
        )
        ## Domain classififer
        self._domain_clf = torch.nn.Sequential(
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(.2),
            torch.nn.Linear(32, n_domains),
            torch.nn.Softmax(dim=-1),
        )

    def forward(self, x, alpha=None):
        x = self._linears(x)
        y_probas = self._class_clf(x)
        domain_probas = None
        if alpha is not None:
            reverse_x = ReverseLayerF.apply(x, alpha)
            domain_probas = self._domain_clf(reverse_x)
        return y_probas, domain_probas


class Method:
    def __init__(self, epochs, batch_size, lr, n_classes):
        super(Method, self).__init__()
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._n_classes = n_classes
        self._models = None
   

    def train(self, x_sources, y_sources, weighted_loss=True):
        domains_s = list(x_sources.keys())
        self._models = {
            'model': DANN(x_sources[domains_s[0]].shape[-1], self._n_classes, len(x_sources))
        }
        model = self._models['model']
        use_cuda = torch.cuda.is_available()
        ## Put model on GPU is used
        if use_cuda:
            model = model.cuda()
        ## Not sure if required
        for p in model.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)
        ## Losses definition
        loss_s_class_fn, dl_s = {}, {}
        for domain_s in domains_s:
            ce_weights = None
            if weighted_loss:
                ce_weights = torch.Tensor(y_sources[domain_s].shape[0] / (self._n_classes * np.bincount(y_sources[domain_s])))
                if use_cuda:
                    ce_weights = ce_weights.cuda()
            loss_s_class_fn[domain_s] = torch.nn.CrossEntropyLoss(weight=ce_weights)
            x, y = torch.Tensor(x_sources[domain_s]), torch.LongTensor(y_sources[domain_s])
            if use_cuda:
                x, y = x.cuda(), y.cuda()
            ## Create dataloaders
            dl_s[domain_s] = DataLoader(TensorDataset(x, y), batch_size=self._batch_size, shuffle=True, drop_last=True)
        loss_domain_fn = torch.nn.CrossEntropyLoss()
        ## Actual training
        n_batches = min([len(dl) for dl in dl_s.values()])
        start_training = time.time()
        for e in range(self._epochs):
            iter_batches_s = {}
            for domain_s in domains_s:
                iter_batches_s[domain_s] = iter(dl_s[domain_s])
            for batch in range(n_batches):
                ## Compute alpha, used to scale gradients from domain classifier
                p = float(batch + e * n_batches) / self._epochs / n_batches
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                ## Train model on source domains
                for i, domain_s in enumerate(domains_s):
                    optimizer.zero_grad()
                    x_s_batch, y_s_batch = next(iter_batches_s[domain_s])
                    y_domain = torch.zeros(len(y_s_batch)).long() + i
                    if use_cuda:
                        y_domain = y_domain.cuda()
                    ## Feed forward of source domain data through DANN
                    y_probas, domain_probas = model(x_s_batch, alpha)
                    loss_class = loss_s_class_fn[domain_s](y_probas, y_s_batch)
                    loss_domain = loss_domain_fn(domain_probas, y_domain)
                    loss = loss_class + loss_domain
                    loss.backward()
                    optimizer.step()
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
            y_probas, _ = model(x)
        model.train()
        return y_probas.cpu()