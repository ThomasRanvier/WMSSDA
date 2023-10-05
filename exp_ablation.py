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

This script is used to run our ablation study on the DIGITS dataset
Run python3 exp_ablation.py --help to access the help on how to use this script.
"""

import os.path
import argparse
import pickle
from utils import datasets, utils
from ablated_methods import wmssda_a, wmssda_b, wmssda_c, wmssda_d, wmssda_e, wmssda


available_methods = {
    'WMSSDA-A': wmssda_a,
    'WMSSDA-B': wmssda_b,
    'WMSSDA-C': wmssda_c,
    'WMSSDA-D': wmssda_d,
    'WMSSDA-E': wmssda_e,
    'WMSSDA': wmssda,
}


evaluation_metrics = [
    'acc',
    'bal_acc',
    'auc',
    'f1',
]


domains = [
    'mnist',
    'mnistm',
    'svhn',
    'syn',
    'usps',
]


imbalances = {
    'HIDIFF': datasets.HIGH_IMBALANCE_AMOUNTS_DIGITS_DIFFERENT_PY,
}


def main(method, py, n_runs, epochs, batch_size, learning_rate):
    exp_name = f'ABLATION_{method.replace("-", "_")}_r{n_runs}_e{epochs}_b{batch_size}'
    if os.path.isfile(f'results/{exp_name}.pickle'):
        metrics = pickle.load(open(f'results/{exp_name}.pickle', 'rb'))
        runs_done = len(metrics[domains[0]][evaluation_metrics[0]])
        if runs_done >= n_runs:
            print(f'## ' + '-' * len(f'Skip ABLATION - {method}') + ' ##', flush=True)
            print(f'## Skip ABLATION - {method} ##', flush=True)
            print(f'## ' + '-' * len(f'Skip ABLATION - {method}') + ' ##', flush=True)
            return
    else:
        # Dictionary to collect results
        metrics = {
            domain_t: {metric: [] for metric in evaluation_metrics} for domain_t in domains
        }
        runs_done = 0
    # Load dataset
    x_train, x_test, y_train, y_test, classes = datasets.load('digits', imbalance=imbalances[py])
    ## Exec remaining runs
    print(f'## ' + '-' * len(f'ABLATION - {method}') + ' ##', flush=True)
    print(f'## ABLATION - {method} ##', flush=True)
    for run in range(runs_done, n_runs):
        print(f'    RUN {run + 1:03d}/{n_runs:03d}', flush=True)
        ## Instantiate method
        m = available_methods[method]
        m_params = (epochs, batch_size, learning_rate, 3, len(classes))
        ## Run method depending on its DA setting
        run_metrics = utils.run_ms1t(m, m_params, domains, x_train, x_test, y_train, y_test, evaluation_metrics)
        ## Extract run metrics to global metrics
        for domain_t, domain_t_metrics in run_metrics.items():
            for met in evaluation_metrics:
                metrics[domain_t][met].append(domain_t_metrics[met])
            if 'sp_weights' in domain_t_metrics:
                if 'sp_weights' in metrics[domain_t]:
                    for w, v in domain_t_metrics['sp_weights'].items():
                        metrics[domain_t]['sp_weights'][w].append(v)
                else:
                    metrics[domain_t]['sp_weights'] = {w: [v] for w, v in domain_t_metrics['sp_weights'].items()}
            if 'weights' in domain_t_metrics:
                if 'weights' in metrics[domain_t]:
                    for w, v in domain_t_metrics['weights'].items():
                        metrics[domain_t]['weights'][w].append(v)
                else:
                    metrics[domain_t]['weights'] = {w: [v] for w, v in domain_t_metrics['weights'].items()}
        ## Save metrics to pickle after each run
        if not os.path.exists('results'):
            os.makedirs('results')
        pickle.dump(metrics, open(f'results/{exp_name}.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print(f'## ' + '-' * len(f'ABLATION - {method}') + ' ##\n', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', type=str, choices=list(available_methods.keys()),
                        default='WMSSDA', help='method, default is WMSSDA')
    parser.add_argument('-p', '--py', type=str, choices=list(imbalances.keys()),
                        default='HIDIFF', help='P(Y) distribution, default is HIDIFF')
    parser.add_argument('-r', '--n-runs', type=int,
                        default=5, help='number of runs, default is 5')
    parser.add_argument('-e', '--epochs', type=int,
                        default=150, help='number of epochs, default is 150')
    parser.add_argument('-b', '--batch-size', type=int,
                        default=128, help='batch size, default is 128')
    parser.add_argument('-l', '--learning-rate', type=float,
                        default=1e-4, help='learning rate, default is 1e-4')
    args = vars(parser.parse_args())
    
    main(**args)
