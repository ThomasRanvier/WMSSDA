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
    def __init__(self, in_c, n_classes):
        super(DANN, self).__init__()
        ## Feature extractor
        self._feature_extractor = torch.nn.Sequential(
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
        )
        in_f = 64 * 4 * 4
        ## Class classififer
        self._class_clf = torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_f),
            torch.nn.Dropout(.2),
            torch.nn.Linear(in_f, n_classes),
            torch.nn.Softmax(dim=-1),
        )
        ## Domain classififer
        self._domain_clf = torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_f),
            torch.nn.Dropout(.2),
            torch.nn.Linear(in_f, 2),
            torch.nn.Softmax(dim=-1),
        )

    def forward(self, x, alpha=None):
        x = self._feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        y_probas = self._class_clf(x)
        domain_probas = None
        if alpha is not None:
            reverse_x = ReverseLayerF.apply(x, alpha)
            domain_probas = self._domain_clf(reverse_x)
        return y_probas, domain_probas


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
            'model': DANN(self._in_c, self._n_classes)
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
        loss_domain_fn = torch.nn.CrossEntropyLoss()
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
                ## Compute alpha, used to scale gradients from domain classifier
                p = float(batch + e * n_batches) / self._epochs / n_batches
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                ## Train model on source domain
                x_s_batch, y_s_batch = next(iter_batches_s)
                optimizer.zero_grad()
                y_domain = torch.zeros(len(y_s_batch)).long()
                if use_cuda:
                    y_domain = y_domain.cuda()
                ## Feed forward of source domain data through DANN
                class_output, domain_output = model(x_s_batch, alpha)
                loss_s_class = loss_s_class_fn(class_output, y_s_batch)
                loss_s_domain = loss_domain_fn(domain_output, y_domain)
                ## Train model on target domain
                x_t_batch, y_t_batch = next(iter_batches_t)
                y_domain = torch.ones(len(y_t_batch)).long()
                if use_cuda:
                    y_domain = y_domain.cuda()
                class_output, domain_output = model(x_t_batch, alpha)
                loss_t_class = loss_t_class_fn(class_output, y_t_batch)
                loss_t_domain = loss_domain_fn(domain_output, y_domain)
                loss = loss_s_class + loss_s_domain + loss_t_class + loss_t_domain
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