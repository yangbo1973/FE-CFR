import copy
import os
import glob
import numpy as np
from numpy import load
import argparse
import scipy.stats

def load_data(knob='default', replication=1, model='baseline', train_test='test',syns_dir='idhp'):
    """
    loading train test experiment results
    """

    file_path = '../../result/{}/{}/'.format(syns_dir,knob)
    data = load(file_path + '{}/{}/0_replication_{}.npz'.format(replication, model, train_test))

    return data['q_t0'].reshape(-1, 1), data['q_t1'].reshape(-1, 1), data['g'].reshape(-1, 1), \
           data['t'].reshape(-1, 1), data['y'].reshape(-1, 1), data['eps'].reshape(-1, 1)


def truncate_by_g(attribute, g, level=0.01):
    keep_these = np.logical_and(g >= level, g <= 1. - level)
    return attribute[keep_these]


def psi_naive(q_t0, q_t1, g, t, y, truncate_level=0.):
    ite = (q_t1 - q_t0)
    return np.mean(truncate_by_g(ite, g, level=truncate_level))


def truncate_all_by_g(q_t0, q_t1, g, t, y, eps, e, i, truncate_level=0.05):
    """
    Helper function to clean up nuisance parameter estimates.
    """

    orig_g = np.copy(g)
    q_t0 = truncate_by_g(np.copy(q_t0), orig_g, truncate_level)
    q_t1 = truncate_by_g(np.copy(q_t1), orig_g, truncate_level)
    g = truncate_by_g(np.copy(g), orig_g, truncate_level)
    t = truncate_by_g(np.copy(t), orig_g, truncate_level)
    y = truncate_by_g(np.copy(y), orig_g, truncate_level)
    eps = truncate_by_g(np.copy(eps), orig_g, truncate_level)
    e = truncate_by_g(np.copy(e.reshape(-1, 1)), orig_g, truncate_level)
    i = truncate_by_g(np.copy(i).reshape(-1, 1), orig_g, truncate_level)

    return q_t0, q_t1, g, t, y, eps, e, i


def mse(x, y):
    return np.mean(np.square(x - y))


def psi_tmle_cont_outcome(q_t0, q_t1, g, t, y, eps_hat=None, truncate_level=0.05):
    # q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)
    g_loss = mse(g, t)
    h = t * (1.0 / g) - (1.0 - t) / (1.0 - g)
    full_q = (1.0 - t) * q_t0 + t * q_t1  # predictions from unperturbed model

    if eps_hat is None:
        eps_hat = np.sum(h * (y - full_q)) / np.sum(np.square(h))

    def q1(t_cf):
        h_cf = t_cf * (1.0 / g) - (1.0 - t_cf) / (1.0 - g)
        full_q = (1.0 - t_cf) * q_t0 + t_cf * q_t1  # predictions from unperturbed model
        return full_q + eps_hat * h_cf

    ite = q1(np.ones_like(t)) - q1(np.zeros_like(t))
    psi_tmle = np.mean(ite)

    # standard deviation computation relies on asymptotic expansion of non-parametric estimator, see van der Laan and Rose p 96
    ic = h * (y - q1(t)) + ite - psi_tmle
    psi_tmle_std = np.std(ic) / np.sqrt(t.shape[0])
    initial_loss = np.mean(np.square(full_q - y))
    final_loss = np.mean(np.square(q1(t) - y))

    return psi_tmle, psi_tmle_std, eps_hat, initial_loss, final_loss, g_loss



def make_table(train_test='train',
               syns_dir='idhp',
               truncate_level=0.01):

    dict, tmle_dict = {}, {}
    knob_list = sorted(glob.glob('../../result/{}/*'.format(syns_dir)))
    print(knob_list)
    knob_list = [ aa.split('\\')[-1] for aa in knob_list]
    print("knob_list::",knob_list)

    for knob in list(knob_list):
        file_path = '../../result/{}/{}/*'.format(syns_dir,knob)
        simulation_files = sorted(glob.glob(file_path))
        print(knob,"-->FOUND::",len(simulation_files),"simulation files in ",file_path)

        dict[knob], tmle_dict[knob] = {}, {}
        for model in ['baseline', 'targeted_regularization']:
            simple_errors, tmle_errors = [], []
            pehe_errors = []
            # simple_errors, tmle_errors = [], []
            # dict[knob],tmle_dict[knob]={},{}
            for rep in range(len(simulation_files)):
                # print(rep)
                file_dir = '../../result/{}/{}/{}/{}'.format(syns_dir, knob, rep, model)
                if os.path.exists(file_dir):
                    q_t0, q_t1, g, t, y, eps = load_data(knob, rep, model, train_test,syns_dir=syns_dir)
                    truth = 2
                    psi_n = np.mean(q_t1-q_t0)
                    err =  abs(truth - psi_n).mean()
                    simple_errors.append(err)
            # print(model)
            dict[knob][model] = np.mean(simple_errors)
            tmle_dict[knob][model] = np.mean(tmle_errors)
            dict[knob][model+'_std'] = scipy.stats.sem(simple_errors)
            tmle_dict[knob][model+'_std'] = scipy.stats.sem(tmle_errors)

    return dict, tmle_dict


def main(syns_dir='jobs'):
    print("************ TRAIN *********************")
    print(syns_dir)
    dict, tmle_dict = make_table(train_test='train', syns_dir=syns_dir)
    print("--------------------------")
    print(">>>> Results for Non-TMLE estimator (=BCAUSS, by default):")
    print(dict)

    # print("--------------------------")
    # print("Results for TMLE estimator:")
    # print(tmle_dict)
    # print()

    print("************ TEST *********************")
    dict, tmle_dict = make_table(train_test='test', syns_dir=syns_dir)
    print("--------------------------")
    print(">>>> Results for Non-TMLE estimator (=BCAUSS, by default):")
    print(dict)

    # print("--------------------------")
    # print("Results for TMLE estimator:")
    # print(tmle_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_base_dir', type=str, help="path to directory LBIDD", default='syn-data')
    args = parser.parse_args()
    main(syns_dir=args.data_base_dir)

