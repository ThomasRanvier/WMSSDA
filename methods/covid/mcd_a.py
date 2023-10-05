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

Using code from https://github.com/mil-tokyo/MCD_DA
"""

import numpy as np
import torch
import os
import time
from torch.utils.data import TensorDataset, DataLoader


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
   

    def train(self, x_source, y_source, x_target, y_target, weighted_loss=True):
        self._models = {
            'enc': Encoder(x_target.shape[-1]),
            'c_1': Clf(self._n_classes),
            'c_2': Clf(self._n_classes),
        }
        enc, c_1, c_2 = self._models['enc'], self._models['c_1'], self._models['c_2']
        use_cuda = torch.cuda.is_available()
        ## Put model on GPU if used
        if use_cuda:
            enc, c_1, c_2 = enc.cuda(), c_1.cuda(), c_2.cuda()
        opt_enc = torch.optim.Adam(enc.parameters(), lr=self._lr)
        opt_c_1 = torch.optim.Adam(c_1.parameters(), lr=self._lr)
        opt_c_2 = torch.optim.Adam(c_2.parameters(), lr=self._lr)
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
                ## Train enc and clfs on both source and target domain
                x_s_batch, y_s_batch = next(iter_batches_s)
                x_t_batch, y_t_batch = next(iter_batches_t)
                enc.train()
                c_1.train()
                c_2.train()
                opt_enc.zero_grad()
                opt_c_1.zero_grad()
                opt_c_2.zero_grad()
                feat_s = enc(x_s_batch)
                output_s1 = c_1(feat_s)
                output_s2 = c_2(feat_s)
                feat_t = enc(x_t_batch)
                output_t1 = c_1(feat_t)
                output_t2 = c_2(feat_t)
                loss_s1 = loss_s_class_fn(output_s1, y_s_batch)
                loss_s2 = loss_s_class_fn(output_s2, y_s_batch)
                loss_t1 = loss_t_class_fn(output_t1, y_t_batch)
                loss_t2 = loss_t_class_fn(output_t2, y_t_batch)
                loss_enc_clfs = (loss_s1 + loss_s2 + loss_t1 + loss_t2) / 4
                loss_enc_clfs.backward()
                opt_enc.step()
                opt_c_1.step()
                opt_c_2.step()
                ## Train classifiers to maximize discrepancy on target samples between them while encoder is fixed, the
                ## discrepancy is added as a regularization to the classification loss, without this loss the original authors
                ## have experimentally found that MCD's performance dropped significantly
                enc.eval()
                c_1.train()
                c_2.train()
                opt_enc.zero_grad()
                opt_c_1.zero_grad()
                opt_c_2.zero_grad()
                feat_s = enc(x_s_batch)
                output_s1 = c_1(feat_s)
                output_s2 = c_2(feat_s)
                feat_t = enc(x_t_batch)
                output_t1 = c_1(feat_t)
                output_t2 = c_2(feat_t)
                loss_s1 = loss_s_class_fn(output_s1, y_s_batch)
                loss_s2 = loss_s_class_fn(output_s2, y_s_batch)
                loss_t1 = loss_t_class_fn(output_t1, y_t_batch)
                loss_t2 = loss_t_class_fn(output_t2, y_t_batch)
                loss_clfs = (loss_s1 + loss_s2 + loss_t1 + loss_t2) / 4
                loss_dis = torch.mean(torch.abs(output_t1 - output_t2))
                loss_clfs_max_dis = loss_clfs - loss_dis
                loss_clfs_max_dis.backward()
                opt_c_1.step()
                opt_c_2.step()
                ## Train encoder to minimize discrepancy on target samples between classifiers while they are fixed
                enc.train()
                c_1.eval()
                c_2.eval()
                for i in range(3):
                    opt_enc.zero_grad()
                    opt_c_1.zero_grad()
                    opt_c_2.zero_grad()
                    feat_t = enc(x_t_batch)
                    output_t1 = c_1(feat_t)
                    output_t2 = c_2(feat_t)
                    loss_gen_min_dis = torch.mean(torch.abs(output_t1 - output_t2))
                    loss_gen_min_dis.backward()
                    opt_enc.step()
            if e == 0 or (e + 1) % 10 == 0:
                time_elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_training))
                print(f'Epoch {e+1}/{self._epochs} - Time elapsed {time_elapsed}', flush=True)


    def predict(self, x):
        enc, c_1, c_2 = self._models['enc'], self._models['c_1'], self._models['c_2']
        use_cuda = torch.cuda.is_available()
        x = torch.Tensor(x)
        if use_cuda:
            x = x.cuda()
            enc = enc.cuda()
            c_1 = c_1.cuda()
            c_2 = c_2.cuda()
        enc.eval()
        c_1.eval()
        c_2.eval()
        with torch.no_grad():
            feat = enc(x)
            y_probas_1 = c_1(feat)
            y_probas_2 = c_2(feat)
        y_probas = (y_probas_1 + y_probas_2) / 2
        enc.train()
        c_1.train()
        c_2.train()
        return y_probas.cpu()