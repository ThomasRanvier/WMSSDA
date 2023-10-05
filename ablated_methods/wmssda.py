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

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import torch
import os
import time
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import balanced_accuracy_score


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        return x

    @staticmethod
    def backward(self, grad_output):
        return grad_output.neg()


class Encoder(torch.nn.Module):
    def __init__(self, in_c):
        super(Encoder, self).__init__()
        ## Convs
        self._convs = torch.nn.Sequential(
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
            
    def forward(self, x):
        x = self._convs(x)
        f = torch.flatten(x, start_dim=1)
        return f


class Neck(torch.nn.Module):
    def __init__(self, output_dim):
        super(Neck, self).__init__()
        in_f = 64 * 4 * 4
        self._linears = torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_f),
            torch.nn.Dropout(.2),
            torch.nn.Linear(in_f, output_dim),
            torch.nn.LeakyReLU(.2, inplace=True),
        )
            
    def forward(self, x):
        return self._linears(x)


class Clf(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Clf, self).__init__()
        self._linears = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.Softmax(dim=-1),
        )
            
    def forward(self, x):
        return self._linears(x)


def md(feats_s, feats_t):
    moment = 0
    for k in [1, 2]:
        moment += ((feats_s**k - feats_t**k)**2).sum().sqrt()
    return moment / 2


class Method:
    def __init__(self, epochs, batch_size, lr, in_c, n_classes):
        super(Method, self).__init__()
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._in_c = in_c
        self._n_classes = n_classes
        self._models = None
        self._domains_s = None
        self._sp_weights = None
        self._weights = None
   

    def train(self, x_sources, y_sources, x_target, y_target, weighted_loss=True, domain_t=None):
        input_dims = [x_sources[domain_s].shape[-1] for domain_s in x_sources.keys()] + [x_target.shape[-1]]
        self._models = {
            'enc': Encoder(self._in_c),
            'com_neck': Neck(64),
            'com_clf': Clf(64, self._n_classes),
            'adv_clf_d': Clf(64, 2),
        }
        self._domains_s = list(x_sources.keys())
        in_f = 16
        for domain_s in self._domains_s:
            self._models[domain_s] = {
                'sp_neck': Neck(in_f),
                'sp_clf': Clf(in_f, self._n_classes),
                'sp_clf_d': Clf(in_f, 2),
            }
        enc = self._models['enc']
        com_neck = self._models['com_neck']
        com_clf = self._models['com_clf']
        adv_clf_d = self._models['adv_clf_d']
        sp_necks, sp_clfs, sp_clfs_d = {}, {}, {}
        for domain in self._domains_s:
            sp_necks[domain] = self._models[domain]['sp_neck']
            sp_clfs[domain] = self._models[domain]['sp_clf']
            sp_clfs_d[domain] = self._models[domain]['sp_clf_d']
        use_cuda = torch.cuda.is_available()
        ## Put model on GPU if used
        if use_cuda:
            enc = enc.cuda()
            com_neck = com_neck.cuda()
            adv_clf_d = adv_clf_d.cuda()
            com_clf = com_clf.cuda()
            for domain in self._domains_s:
                sp_necks[domain] = sp_necks[domain].cuda()
                sp_clfs[domain] = sp_clfs[domain].cuda()
                sp_clfs_d[domain] = sp_clfs_d[domain].cuda()
        params_sp_necks, params_sp_clfs, params_sp_clfs_d = [], [], []
        for domain in self._domains_s:
            params_sp_necks += list(sp_necks[domain].parameters())
            params_sp_clfs += list(sp_clfs[domain].parameters())
            params_sp_clfs_d += list(sp_clfs_d[domain].parameters())
        optimizer = torch.optim.Adam(list(enc.parameters()) +
                                     list(com_neck.parameters()) +
                                     params_sp_necks +
                                     params_sp_clfs_d +
                                     list(adv_clf_d.parameters()) +
                                     list(com_clf.parameters()) +
                                     params_sp_clfs, lr=self._lr)
        ## Losses definition
        loss_s_class_fn, dl_s = {}, {}
        for domain_s in self._domains_s:
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
        ce_weights = None
        if weighted_loss:
            ce_weights = torch.Tensor(y_target.shape[0] / (self._n_classes * np.bincount(y_target)))
            if use_cuda:
                ce_weights = ce_weights.cuda()
        loss_t_class_fn = torch.nn.CrossEntropyLoss(weight=ce_weights)
        loss_domain_fn = torch.nn.CrossEntropyLoss()
        ## Create domain classes
        y_domain_s = torch.zeros(self._batch_size).long()
        y_domain_t = torch.ones(self._batch_size).long()
        ## Create dataloaders
        x_t, y_t = torch.Tensor(x_target), torch.LongTensor(y_target)
        if use_cuda:
            x_t, y_t = x_t.cuda(), y_t.cuda()
            y_domain_s, y_domain_t = y_domain_s.cuda(), y_domain_t.cuda()
        dl_t = DataLoader(TensorDataset(x_t, y_t), batch_size=self._batch_size, shuffle=True, drop_last=True)
        ## Actual training
        n_batches = min([len(dl) for dl in dl_s.values()] + [len(dl_t)])
        ## Save MD values for each source domain at each batch and use them as scaling hyper-parameter of sp clf loss
        sp_md_reg = {domain_s: None for domain_s in self._domains_s}
        start_training = time.time()
        for e in range(self._epochs):
            iter_batches_s = {}
            for domain_s in self._domains_s:
                iter_batches_s[domain_s] = iter(dl_s[domain_s])
            iter_batches_t = iter(dl_t)
            for batch in range(n_batches):
                if sp_md_reg[domain_s] is not None:
                    vals = list(sp_md_reg.values())
                    e_x = np.exp(vals - np.max(vals))
                    sp_md_weights = dict(zip(sp_md_reg.keys(), (1 - (e_x / e_x.sum())) + (1 / len(sp_md_reg))))
                else:
                    sp_md_weights = dict(zip(sp_md_reg.keys(), [1.] * len(sp_md_reg)))
                x_t_batch, y_t_batch = next(iter_batches_t)
                for domain_s in self._domains_s:
                    optimizer.zero_grad()
                    ## Feed target batch
                    feat_t = enc(x_t_batch)
                    com_z_t = com_neck(feat_t)
                    com_y_probas_t = com_clf(com_z_t)
                    sp_zs_t, sp_y_probas_t = {}, {}
                    #sp_y_probas_t_avg = []
                    for _domain in self._domains_s:
                        sp_zs_t[_domain] = sp_necks[_domain](feat_t)
                        ## Save specific clf probas on target
                        sp_y_probas_t[_domain] = sp_clfs[_domain](sp_zs_t[_domain])
                    y_probas_t = (com_y_probas_t + sp_y_probas_t[domain_s]) / 2
                    ## Feed source batch
                    x_s_batch, y_s_batch = next(iter_batches_s[domain_s])
                    feat_s = enc(x_s_batch)
                    com_z_s = com_neck(feat_s)
                    sp_z_s = sp_necks[domain_s](feat_s)
                    com_y_probas_s = com_clf(com_z_s)
                    sp_y_probas_s = sp_clfs[domain_s](sp_z_s)
                    y_probas_s = (com_y_probas_s + sp_y_probas_s) / 2
                    ## Adversarial domain classification on common neck
                    y_probas_d_s = adv_clf_d(GradReverse.apply(com_z_s))
                    y_probas_d_t = adv_clf_d(GradReverse.apply(com_z_t))
                    adv_clfd_reg = 1e-1 * (loss_domain_fn(y_probas_d_s, y_domain_s) +
                                           loss_domain_fn(y_probas_d_t, y_domain_t)) / 2
                    ## Domain classification on specific neck
                    y_probas_d_s = sp_clfs_d[domain_s](sp_z_s)
                    y_probas_d_t = sp_clfs_d[domain_s](sp_zs_t[domain_s])
                    clfd_reg = 1e-1 * (loss_domain_fn(y_probas_d_s, y_domain_s) +
                                       loss_domain_fn(y_probas_d_t, y_domain_t)) / 2
                    ## MD regularization
                    gamma = 2 / (1 + np.exp(-10 * (e * n_batches + batch) / (self._epochs * n_batches))) - 1
                    md_reg_raw = md(com_z_s, com_z_t)
                    sp_md_reg[domain_s] = md_reg_raw.detach().cpu().numpy()
                    md_reg = gamma * .002 * md_reg_raw
                    ## Classifier loss on source + target
                    loss_cls = (sp_md_weights[domain_s] * loss_s_class_fn[domain_s](y_probas_s, y_s_batch) +
                                loss_t_class_fn(y_probas_t, y_t_batch)) / (1 + sp_md_weights[domain_s])
                    ## Compute total loss
                    loss = loss_cls + adv_clfd_reg + clfd_reg + md_reg
                    loss.backward()
                    optimizer.step()
            if e == 0 or (e + 1) % 10 == 0:
                time_elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_training))
                print(f'Epoch {e+1}/{self._epochs} - Time elapsed {time_elapsed}', flush=True)


    def predict(self, x):
        x = torch.Tensor(x)
        enc = self._models['enc']
        com_neck = self._models['com_neck']
        com_clf = self._models['com_clf']
        sp_necks, sp_clfs = {}, {}
        for domain in self._domains_s:
            sp_necks[domain] = self._models[domain]['sp_neck']
            sp_clfs[domain] = self._models[domain]['sp_clf']
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            x = x.cuda()
            enc = enc.cuda()
            com_neck = com_neck.cuda()
            com_clf = com_clf.cuda()
            for domain in self._domains_s:
                sp_necks[domain] = sp_necks[domain].cuda()
                sp_clfs[domain] = sp_clfs[domain].cuda()
        enc.eval()
        com_neck.eval()
        com_clf.eval()
        for domain in self._domains_s:
            sp_necks[domain].eval()
            sp_clfs[domain].eval()
        with torch.no_grad():
            feat_t = enc(x)
            com_z_t = com_neck(feat_t)
            com_y_probas_t = com_clf(com_z_t)
            sp_y_probas_t_avg = []
            for domain in self._domains_s:
                sp_z_t = sp_necks[domain](feat_t)
                sp_y_probas_t = sp_clfs[domain](sp_z_t)
                sp_y_probas_t_avg.append(sp_y_probas_t)
            sp_y_probas_t_avg = torch.sum(torch.stack(sp_y_probas_t_avg, 0), 0) / len(self._domains_s)
            y_probas = (com_y_probas_t + sp_y_probas_t_avg) / 2
        enc.train()
        com_neck.train()
        com_clf.train()
        for domain in self._domains_s:
            sp_necks[domain].train()
            sp_clfs[domain].train()
        return y_probas.cpu()