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

Using code from https://github.com/easezyc/deep-transfer-learning/tree/master/UDA/pytorch1.0/DSAN
"""

import numpy as np
import torch
import os
import time
from torch.utils.data import TensorDataset, DataLoader


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
            torch.nn.Linear(in_f, 256),
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
        self._clf = torch.nn.ModuleList([
            torch.nn.Linear(256, output_dim),
            torch.nn.Softmax(dim=-1),
        ])
            
    def forward(self, x):
        for l in self._clf:
            x = l(x)
        return x


class LMMD_loss(torch.nn.Module):
    def __init__(self, class_num, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(LMMD_loss, self).__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def get_loss(self, source, target, s_label, t_label):
        batch_size = source.size()[0]
        weight_ss, weight_tt, weight_st = self.cal_weight(s_label, t_label,
                                                          batch_size=batch_size, class_num=self.class_num)
        weight_ss = torch.from_numpy(weight_ss).cuda()
        weight_tt = torch.from_numpy(weight_tt).cuda()
        weight_st = torch.from_numpy(weight_st).cuda()

        kernels = self.gaussian_kernel(source, target,
                                kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        loss = torch.Tensor([0]).cuda()
        if torch.sum(torch.isnan(sum(kernels))):
            return loss
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]

        loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
        return loss

    def convert_to_onehot(self, sca_label, class_num=31):
        return np.eye(class_num)[sca_label]

    def cal_weight(self, s_label, t_label, batch_size=32, class_num=31):
        batch_size = s_label.size()[0]
        s_sca_label = s_label.cpu().data.numpy()
        s_vec_label = self.convert_to_onehot(s_sca_label, class_num=self.class_num)
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum

        t_sca_label = t_label.cpu().data.numpy()
        t_vec_label = self.convert_to_onehot(t_sca_label, class_num=self.class_num)
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        index = list(set(s_sca_label) & set(t_sca_label))
        mask_arr = np.zeros((batch_size, class_num))
        mask_arr[:, index] = 1
        t_vec_label = t_vec_label * mask_arr
        s_vec_label = s_vec_label * mask_arr

        weight_ss = np.matmul(s_vec_label, s_vec_label.T)
        weight_tt = np.matmul(t_vec_label, t_vec_label.T)
        weight_st = np.matmul(s_vec_label, t_vec_label.T)

        length = len(index)
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')


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
            'enc': Encoder(self._in_c),
            'clf': Clf(self._n_classes),
        }
        enc, clf = self._models['enc'], self._models['clf']
        use_cuda = torch.cuda.is_available()
        ## Put model on GPU is used
        if use_cuda:
            enc, clf = enc.cuda(), clf.cuda()
        optimizer = torch.optim.Adam(list(enc.parameters()) + list(clf.parameters()), lr=self._lr)
        ## Losses definition
        loss_s_class_fn, dl_s = {}, {}
        for domain_s in x_sources.keys():
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
        lmmd = LMMD_loss(self._n_classes)
        ## Create dataloaders
        x_t, y_t = torch.Tensor(x_target), torch.LongTensor(y_target)
        if use_cuda:
            x_t, y_t = x_t.cuda(), y_t.cuda()
        dl_t = DataLoader(TensorDataset(x_t, y_t), batch_size=self._batch_size, shuffle=True, drop_last=True)
        ## Actual training
        n_batches = min([len(dl) for dl in dl_s.values()] + [len(dl_t)])
        start_training = time.time()
        for e in range(self._epochs):
            iter_batches_s = {}
            for domain_s in x_sources.keys():
                iter_batches_s[domain_s] = iter(dl_s[domain_s])
            iter_batches_t = iter(dl_t)
            for batch in range(n_batches):
                x_t_batch, y_t_batch = next(iter_batches_t)
                for domain_s in x_sources.keys():
                    optimizer.zero_grad()
                    ## Feed source batch
                    x_s_batch, y_s_batch = next(iter_batches_s[domain_s])
                    feat_s = enc(x_s_batch)
                    y_probas_s = clf(feat_s)
                    ## Feed target batch
                    feat_t = enc(x_t_batch)
                    y_probas_t = clf(feat_t)
                    ## Classifier loss on source + target
                    loss_cls = (loss_s_class_fn[domain_s](y_probas_s, y_s_batch) + loss_t_class_fn(y_probas_t, y_t_batch)) / 2
                    ## LMMD regularization, using real target label since we are in a supervised case
                    lambd = 2 / (1 + np.exp(-10 * (e) / self._epochs)) - 1
                    lmmd_reg = .5 * lambd * lmmd.get_loss(feat_s, feat_t, y_s_batch, y_t_batch)
                    ## Compute total loss
                    loss = loss_cls + lmmd_reg
                    loss.backward()
                    optimizer.step()
            if e == 0 or (e + 1) % 10 == 0:
                time_elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_training))
                print(f'Epoch {e+1}/{self._epochs} - Time elapsed {time_elapsed}', flush=True)


    def predict(self, x):
        encoder, clf = self._models['enc'], self._models['clf']
        use_cuda = torch.cuda.is_available()
        x = torch.Tensor(x)
        if use_cuda:
            x = x.cuda()
            encoder, clf = encoder.cuda(), clf.cuda()
        encoder.eval()
        clf.eval()
        with torch.no_grad():
            y_probas = clf(encoder(x))
        encoder.train()
        clf.train()
        return y_probas.cpu()