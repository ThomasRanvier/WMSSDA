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

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch


def run_1s1t(m, m_params, domains, x_train, x_test, y_train, y_test, evaluation_metrics):
    """
    Go through all possible domain pairs, train and eval model for each, saved results are the best obtained for each target domain.
    """
    metrics = {domain_t: None for domain_t in domains}
    for domain_s in domains:
        for domain_t in domains:
            if domain_s != domain_t:
                print(f'    - {domain_s} -> {domain_t}', flush=True)
                ## Instantiate method
                method = m.Method(*m_params)
                ## Train method (resets models)
                method.train(x_train[domain_s], y_train[domain_s], x_train[domain_t], y_train[domain_t])
                ## Eval trained method
                y_probas = method.predict(x_test[domain_t])
                met = eval_results(y_probas, y_test[domain_t], evaluation_metrics)
                ## If results of domain_s->domain_t are better than previous replace saved metrics for domain_t
                if metrics[domain_t] is None or met['auc'] > metrics[domain_t]['auc']:
                    metrics[domain_t] = met
                ## Del method instance to free GPU
                del method
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return metrics


def run_ms1t(m, m_params, domains, x_train, x_test, y_train, y_test, evaluation_metrics):
    """
    Multi-source domain adaptation setting.
    """
    metrics = {domain_t: None for domain_t in domains}
    for domain_t in domains:
        print(f'    - all -> {domain_t}', flush=True)
        x_sources, y_sources = {}, {}
        for domain_s in domains:
            if domain_s != domain_t:
                x_sources[domain_s] = x_train[domain_s]
                y_sources[domain_s] = y_train[domain_s]
        ## Instantiate method
        method = m.Method(*m_params)
        ## Train method (resets models)
        method.train(x_sources, y_sources, x_train[domain_t], y_train[domain_t], domain_t=domain_t)
        ## Eval trained method
        y_probas = method.predict(x_test[domain_t])
        sp_weights, weights = None, None
        if type(y_probas) == tuple:
            ## Specific case of our method where we want to be able to visualize computed weights
            y_probas, sp_weights, weights = y_probas
        met = eval_results(y_probas, y_test[domain_t], evaluation_metrics)
        if sp_weights is not None:
            met['sp_weights'] = sp_weights
            met['weights'] = weights
        ## If results of domain_s->domain_t are better than previous replace saved metrics for domain_t
        if metrics[domain_t] is None or met['auc'] > metrics[domain_t]['auc']:
            metrics[domain_t] = met
        ## Del method instance to free GPU
        del method
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return metrics


def run_nsnt(m, m_params, domains, x_train, x_test, y_train, y_test, evaluation_metrics):
    """
    Multi-source multi-target domain adaptation setting.
    """
    metrics = {domain_t: None for domain_t in domains}
    print(f'    - all -> all', flush=True)
    x_sources, y_sources = {}, {}
    for domain_s in domains:
        x_sources[domain_s] = x_train[domain_s]
        y_sources[domain_s] = y_train[domain_s]
    ## Instantiate method
    method = m.Method(*m_params)
    ## Train method (resets models)
    method.train(x_sources, y_sources)
    ## Eval trained method
    for domain_t in domains:
        y_probas = method.predict(x_test[domain_t])
        met = eval_results(y_probas, y_test[domain_t], evaluation_metrics)
        ## If results of domain_s->domain_t are better than previous replace saved metrics for domain_t
        if metrics[domain_t] is None or met['auc'] > metrics[domain_t]['auc']:
            metrics[domain_t] = met
    ## Del method instance to free GPU
    del method
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def run_0s1t(m, m_params, domains, x_train, x_test, y_train, y_test, evaluation_metrics):
    """
    No domain adaptation, training is done only on target domain.
    """
    metrics = {domain_t: None for domain_t in domains}
    for domain_t in domains:
        print(f'    - {domain_t} -> {domain_t}', flush=True)
        ## Instantiate method
        method = m.Method(*m_params)
        ## Train method (resets models)
        method.train(x_train[domain_t], y_train[domain_t])
        ## Eval trained method
        y_probas = method.predict(x_test[domain_t])
        met = eval_results(y_probas, y_test[domain_t], evaluation_metrics)
        ## If results of domain_s->domain_t are better than previous replace saved metrics for domain_t
        if metrics[domain_t] is None or met['auc'] > metrics[domain_t]['auc']:
            metrics[domain_t] = met
    return metrics


def eval_results(y_probas, y_true, evaluation_metrics):
    y_pred = np.argmax(y_probas, axis=-1)
    metrics = {}
    for metric in evaluation_metrics:
        if metric == 'acc':
            metrics[metric] = accuracy_score(y_true, y_pred) * 100.
        if metric == 'bal_acc':
            metrics[metric] = balanced_accuracy_score(y_true, y_pred) * 100.
        if metric == 'auc':
            metrics[metric] = roc_auc_score(y_true, y_probas[:, 1] if y_probas.shape[1] == 2 else y_probas, multi_class='ovo') * 100.
        if metric == 'f1':
            metrics[metric] = f1_score(y_true, y_pred, average='weighted') * 100.
    return metrics


def plot_cm(y_true, y_probas, classes, figsize=(8, 8)):
    y_pred = np.argmax(y_probas, axis=-1)
    cm = confusion_matrix(y_true, y_pred)
    cmn = 100. * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cmn, annot=True, fmt='.1f', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)