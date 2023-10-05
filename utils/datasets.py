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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import torchvision
import os
import scipy.io
from utils import mnistm_loader, syn_loader, usps_loader
import pickle


DATA_FOLDER = 'data/'

HIGH_IMBALANCE_AMOUNTS_DIGITS_DIFFERENT_PY = {
    'mnist':  [500, 500, 500, 500, 200, 100,  40,  10,  10,  10],
    'mnistm': [10,  200, 100, 500, 40,  500,  10, 500, 500,  10],
    'svhn':   [200,  10,  10, 500, 500, 100,  40,  10, 500, 500],
    'syn':    [500, 100, 500,  10,  10,  10, 500, 500,  40, 200],
    'usps':   [10,  200,  10,  10, 100, 500, 500, 500,  40, 500],
}


IMBALANCE_AMOUNTS_DOMAINNET_DIFFERENT_PY = {
    'clipart':   [120, 120, 120, 120, 80, 60, 60, 40, 40, 25, 25, 25, 25, 10, 10, 10, 10],
    'infograph': [60, 25, 40, 120, 120, 25, 60, 40, 120, 25, 10, 80, 25, 10, 10, 120, 10],
    'painting':  [120, 10, 10, 40, 120, 25, 10, 60, 25, 40, 120, 60, 10, 80, 25, 25, 120],
    'real':      [120, 10, 25, 40, 25, 10, 25, 25, 60, 120, 120, 40, 60, 120, 80, 10, 10],
    'sketch':    [60, 25, 10, 120, 10, 40, 40, 10, 120, 120, 120, 25, 80, 25, 25, 10, 60],
    'quickdraw': [40, 60, 25, 10, 120, 10, 40, 10, 120, 25, 120, 25, 120, 80, 25, 10, 60],
}


def load_digits():
    classes = tuple(str(i) for i in range(10))
    data_path = '4_digits/'
    domains = ['mnist', 'mnistm', 'svhn', 'syn', 'usps']
    x_train, x_test, y_train, y_test = {}, {}, {}, {}
    for d in domains:
        if d == 'usps':
            train_ds = usps_loader.USPS(root=DATA_FOLDER + data_path + 'usps/',
                                        train=True, download=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor()
                                        ]))
            x_temp, y_temp = [], []
            for e in train_ds:
                x_temp.append(e[0].numpy())
                y_temp.append(e[1].numpy())
            x_train[d], y_train[d] = np.asarray(x_temp).astype(float), np.asarray(y_temp).squeeze()
            test_ds = usps_loader.USPS(root=DATA_FOLDER + data_path + 'usps/',
                                       train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor()
                                       ]))
            x_temp, y_temp = [], []
            for e in test_ds:
                x_temp.append(e[0].numpy())
                y_temp.append(e[1].numpy())
            x_test[d], y_test[d] = np.asarray(x_temp).astype(float), np.asarray(y_temp).squeeze()
        if d == 'mnist':
            train_ds = torchvision.datasets.MNIST(root=DATA_FOLDER + data_path + 'mnist/',
                                                  train=True, download=True,
                                                  transform=torchvision.transforms.Compose([
                                                      torchvision.transforms.ToTensor()
                                                  ]))
            x_train[d] = train_ds.data.numpy().astype(float)[:, None, :, :]
            y_train[d] = train_ds.targets.numpy()
            test_ds = torchvision.datasets.MNIST(root=DATA_FOLDER + data_path + 'mnist/',
                                                  train=False, download=True,
                                                  transform=torchvision.transforms.Compose([
                                                      torchvision.transforms.ToTensor()
                                                  ]))
            x_test[d] = test_ds.data.numpy().astype(float)[:, None, :, :]
            y_test[d] = test_ds.targets.numpy()
        if d == 'mnistm':
            train_ds = mnistm_loader.MNISTM(root=DATA_FOLDER + data_path + 'mnistm/',
                                              train=True, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor()
                                              ]))
            x_temp, y_temp = [], []
            for e in train_ds:
                x_temp.append(e[0].numpy())
                y_temp.append(e[1].numpy())
            x_train[d], y_train[d] = np.asarray(x_temp).astype(float), np.asarray(y_temp)
            test_ds = mnistm_loader.MNISTM(root=DATA_FOLDER + data_path + 'mnistm/',
                                              train=False, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor()
                                              ]))
            x_temp, y_temp = [], []
            for e in test_ds:
                x_temp.append(e[0].numpy())
                y_temp.append(e[1].numpy())
            x_test[d], y_test[d] = np.asarray(x_temp).astype(float), np.asarray(y_temp)
        if d == 'svhn':
            if not os.path.exists(DATA_FOLDER + data_path + 'svhn/svhn_train.mat'):
                print('You have to download the SVHN dataset (http://ufldl.stanford.edu/housenumbers/), rename files to svhn_train.mat and svhn_test.mat and add them to folder \'data/4_digits/svhn/\'')
            train_ds = scipy.io.loadmat(DATA_FOLDER + data_path + 'svhn/svhn_train.mat')
            x_train[d] = train_ds['X'].astype(float).swapaxes(3, 0).swapaxes(2, 1).swapaxes(3, 2)
            y_train[d] = train_ds['y'].astype(int).squeeze() % 10
            test_ds = scipy.io.loadmat(DATA_FOLDER + data_path + 'svhn/svhn_test.mat')
            x_test[d] = test_ds['X'].astype(float).swapaxes(3, 0).swapaxes(2, 1).swapaxes(3, 2)
            y_test[d] = test_ds['y'].astype(int).squeeze() % 10
        if d == 'syn':
            train_ds = syn_loader.SynDigits(root=DATA_FOLDER + data_path + 'syn/',
                                              train=True, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor()
                                              ]))
            ## Arbitrarily chosen 60k training samples (about 500k by default otherwise)
            x_train[d] = train_ds.data.numpy().astype(float).swapaxes(3, 1).swapaxes(3, 2)[:60000]
            y_train[d] = train_ds.targets.numpy()[:60000]
            test_ds = syn_loader.SynDigits(root=DATA_FOLDER + data_path + 'syn/',
                                              train=False, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor()
                                              ]))
            x_test[d] = test_ds.data.numpy().astype(float).swapaxes(3, 1).swapaxes(3, 2)
            y_test[d] = test_ds.targets.numpy()
    x_train['mnist'] = x_train['mnist'].repeat(3, 1)
    x_test['mnist'] = x_test['mnist'].repeat(3, 1)
    x_train['usps'] = x_train['usps'].repeat(3, 1)
    x_test['usps'] = x_test['usps'].repeat(3, 1)
    return domains, classes, x_train, x_test, y_train, y_test


def imbalance_ds(domains, n_classes, x_original, y_original, amounts):
    """
    This method returns imbalanced training set given the amount of each class instance for each domain
    amounts is a dict of lists of dim=n_classes containing the amount of each class instances in each domain
    Necessitates a shuffle before and after to avoid bias sinec there is no stochasticity in this method
    """
    x, y = {}, {}
    for d in domains:
        x[d] = None
        y[d] = None
        for c in range(n_classes):
            if amounts[d][c] > np.count_nonzero(y_original[d] == c):
                print(f'Not enough class instances: {d}-{c}, {amounts[d][c]}/{np.count_nonzero(y_original[d] == c)}')
            ## Extract target class indices and select only the n first instances (n=amounts[d][c])
            ## (dataset has been shuffled before so the selection is random)
            class_subselect_idcs = np.where(y_original[d] == c)[0][:amounts[d][c]]
            ## Add subset to data
            if x[d] is None:
                x[d] = x_original[d][class_subselect_idcs]
            else:
                x[d] = np.concatenate((x[d], x_original[d][class_subselect_idcs]), axis=0)
            if y[d] is None:
                y[d] = y_original[d][class_subselect_idcs]
            else:
                y[d] = np.concatenate((y[d], y_original[d][class_subselect_idcs]), axis=0)
    return x, y


def load_covid():
    classes = ('DEAD', 'ALIVE')
    data_path = 'covid/'
    with open(DATA_FOLDER + data_path + 'preprocessed_covid.pickle', 'rb') as handle:
        (x_train, x_test, y_train, y_test) = pickle.load(handle)
    domains = list(x_train.keys())
    return domains, classes, x_train, x_test, y_train, y_test


def load_domainnet():
    classes = (
        'bird',
        'whale',
        'circle',
        'suitcase',
        'squirrel',
        'feather',
        'strawberry',
        'triangle',
        'teapot',
        'sea_turtle',
        'bread',
        'windmill',
        'zebra',
        'submarine',
        'tiger',
        'headphones',
        'shark',
    )
    data_path = 'domainnet/'
    with open(DATA_FOLDER + data_path + 'preprocessed_domainnet.pickle', 'rb') as handle:
        (x_train, x_test, y_train, y_test) = pickle.load(handle)
    domains = list(x_train.keys())
    return domains, classes, x_train, x_test, y_train, y_test


def shuffle_ds(x, y, shuffle_seed=0):
    for i, domain in enumerate(x.keys()):
        np.random.seed(shuffle_seed + i)
        p = np.random.permutation(len(x[domain]))
        x[domain], y[domain] = x[domain][p], y[domain][p]
    return x, y


def load(dataset_name, shuffle_seed=0, imbalance=None):
    if dataset_name == 'digits':
        domains, classes, x_train, x_test, y_train, y_test = load_digits()
    elif dataset_name == 'covid':
        domains, classes, x_train, x_test, y_train, y_test = load_covid()
    elif dataset_name == 'domainnet':
        domains, classes, x_train, x_test, y_train, y_test = load_domainnet()
    else:
        print(f'Undefined dataset: {dataset_name}')
        return None
    for d in domains:
        if dataset_name == 'digits' or dataset_name == 'domainnet':
            x_train[d] += np.abs(x_train[d].min())
            x_train[d] /= x_train[d].max()
            x_test[d] += np.abs(x_test[d].min())
            x_test[d] /= x_test[d].max()
        else:
            x_train[d] = ((x_train[d] - x_train[d].min(0)) / x_train[d].ptp(0)).astype(np.float32)
            x_test[d] = ((x_test[d] - x_test[d].min(0)) / x_test[d].ptp(0)).astype(np.float32)
    ## Shuffle
    x_train, y_train = shuffle_ds(x_train, y_train, shuffle_seed=shuffle_seed)
    x_test, y_test = shuffle_ds(x_test, y_test, shuffle_seed=shuffle_seed+42)
    ## Create imbalance given imbalance_amounts
    if imbalance is not None:
        x_train, y_train = imbalance_ds(domains, len(classes), x_train, y_train, imbalance)
        ## Shuffle training set again after subsetting
        x_train, y_train = shuffle_ds(x_train, y_train, shuffle_seed=shuffle_seed+1000)
    return x_train, x_test, y_train, y_test, classes