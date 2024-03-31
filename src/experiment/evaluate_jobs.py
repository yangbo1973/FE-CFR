import copy
import os
import glob
import numpy as np
from numpy import load
import argparse
import scipy.stats

POL_CURVE_RES = 40
def load_data(knob='default', replication=1, model='baseline', train_test='test',jobs_dir='jobs'):
    """
    loading train test experiment results
    """

    file_path = '../../result/{}/{}/'.format(jobs_dir,knob)
    data = load(file_path + '{}/{}/0_replication_{}.npz'.format(replication, model, train_test))

    return data['q_t0'].reshape(-1, 1), data['q_t1'].reshape(-1, 1), data['g'].reshape(-1, 1), \
           data['t'].reshape(-1, 1), data['y'].reshape(-1, 1), data['eps'].reshape(-1, 1), data['E'].reshape(-1, 1),  data['I'].reshape(-1, 1)


def truncate_by_g(attribute, g, level=0.01):
    keep_these = np.logical_and(g >= level, g <= 1.-level)
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
    e = truncate_by_g(np.copy(e.reshape(-1,1)), orig_g, truncate_level)
    i = truncate_by_g(np.copy(i).reshape(-1,1), orig_g, truncate_level)

    return q_t0, q_t1, g, t, y, eps, e, i

def mse(x, y):
    return np.mean(np.square(x-y))


def psi_tmle_cont_outcome(q_t0, q_t1, g, t, y, eps_hat=None, truncate_level=0.05):
    # q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)
    g_loss = mse(g, t)
    h = t * (1.0/g) - (1.0-t) / (1.0 - g)
    full_q = (1.0-t)*q_t0 + t*q_t1 # predictions from unperturbed model

    if eps_hat is None:
        eps_hat = np.sum(h*(y-full_q)) / np.sum(np.square(h))

    def q1(t_cf):
        h_cf = t_cf * (1.0 / g) - (1.0 - t_cf) / (1.0 - g)
        full_q = (1.0 - t_cf) * q_t0 + t_cf * q_t1  # predictions from unperturbed model
        return full_q + eps_hat * h_cf
    ite = q1(np.ones_like(t)) - q1(np.zeros_like(t))
    psi_tmle = np.mean(ite)

    # standard deviation computation relies on asymptotic expansion of non-parametric estimator, see van der Laan and Rose p 96
    ic = h*(y-q1(t)) + ite - psi_tmle
    psi_tmle_std = np.std(ic) / np.sqrt(t.shape[0])
    initial_loss = np.mean(np.square(full_q-y))
    final_loss = np.mean(np.square(q1(t)-y))

    return psi_tmle, psi_tmle_std, eps_hat, initial_loss, final_loss, g_loss

def policy_range(n, res=10):
    step = int(float(n)/float(res))
    n_range = range(0,int(n+1),step)
    if not n_range[-1] == n:
        n_range.append(n)
    while len(n_range) > res:
        k = np.random.randint(len(n_range)-2)+1
        del n_range[k]

    return n_range

def policy_val(t, yf, eff_pred, compute_policy_curve=False):
    """ Computes the value of the policy defined by predicted effect """

    if np.any(np.isnan(eff_pred)):
        return np.nan, np.nan

    policy = eff_pred>0
    treat_overlap = (policy==t)*(t>0)
    control_overlap = (policy==t)*(t<1)

    if np.sum(treat_overlap)==0:
        treat_value = 0
    else:
        treat_value = np.mean(yf[treat_overlap])

    if np.sum(control_overlap)==0:
        control_value = 0
    else:
        control_value = np.mean(yf[control_overlap])

    pit = np.mean(policy)
    policy_value = pit*treat_value + (1-pit)*control_value

    policy_curve = []

    if compute_policy_curve:
        n = t.shape[0]
        I_sort = np.argsort(-eff_pred)

        n_range = policy_range(n, POL_CURVE_RES)

        for i in n_range:
            I = I_sort[0:i]

            policy_i = 0*policy
            policy_i[I] = 1
            pit_i = np.mean(policy_i)

            treat_overlap = (policy_i>0)*(t>0)
            control_overlap = (policy_i<1)*(t<1)

            if np.sum(treat_overlap)==0:
                treat_value = 0
            else:
                treat_value = np.mean(yf[treat_overlap])

            if np.sum(control_overlap)==0:
                control_value = 0
            else:
                control_value = np.mean(yf[control_overlap])

            policy_curve.append(pit_i*treat_value + (1-pit_i)*control_value)

    return policy_value, policy_curve

def make_table(train_test='train',
               jobs_dir='jobs',
               truncate_level=0.01):
    
    dict, tmle_dict = {}, {}
    knob_list = sorted(glob.glob('../../result/{}/*'.format(jobs_dir)))
    print(knob_list)
    knob_list = [ aa.split('\\')[-1] for aa in knob_list]
    print("knob_list::",knob_list)

    for knob in list(knob_list):
        file_path = '../../result/{}/{}/*'.format(jobs_dir,knob)
        simulation_files = sorted(glob.glob(file_path))
        print(knob,"-->FOUND::",len(simulation_files),"simulation files in ",file_path)

        dict[knob], tmle_dict[knob] = {}, {}
        for model in ['baseline', 'targeted_regularization']:
            simple_errors, tmle_errors = [], []
            policy_risk = []
            pehe_errors = []
            # dict[knob],tmle_dict[knob]={},{}
            for rep in range(len(simulation_files)):
                # print(rep)
                file_dir = '../../result/{}/{}/{}/{}'.format(jobs_dir, knob, rep, model)
                if os.path.exists(file_dir):
                    q_t0, q_t1, g, t, y_dragon, eps, e, i = load_data(knob, rep, model, train_test,jobs_dir=jobs_dir)
                    att = np.mean(y_dragon[t > 0]) - np.mean(y_dragon[(1 - t + e) > 1])
                    if np.any(np.isnan(q_t0)) or np.any(np.isnan(q_t1)):
                        raise FloatingPointError('NaN encountered')
                    ite_p = q_t1 - q_t0
                    # print(ite_p[(t+e)>1].shape)
                    att_p = np.mean(ite_p[(t+e)>1])
                    # print(model, att, att_p)
                    err = abs(att - att_p)

                    policy_value, policy_curve = \
                        policy_val(t[e > 0], y_dragon[e > 0], ite_p[e > 0], False)
                    simple_errors.append(err)
                    policy_risk.append((1-policy_value))

            # print(model)
            dict[knob][model+' ATT'] = np.mean(simple_errors)
            dict[knob][model + ' ATT_std'] = scipy.stats.sem(simple_errors)
            dict[knob][model + ' policy_risk'] = np.mean(policy_risk)
            dict[knob][model+' policy_risk_std'] = scipy.stats.sem(policy_risk)
    return dict, tmle_dict



def main(jobs_dir='jobs'):
    print("************ TRAIN *********************")
    print(jobs_dir)
    dict, tmle_dict = make_table(train_test='train',jobs_dir=jobs_dir)
    print("--------------------------")
    print(">>>> Results for Non-TMLE estimator (=BCAUSS, by default):")
    print(dict)

    # print("--------------------------")
    # print("Results for TMLE estimator:")
    # print(tmle_dict)
    # print()

    print("************ TEST *********************")
    dict, tmle_dict = make_table(train_test='test',jobs_dir=jobs_dir)
    print("--------------------------")
    print(">>>> Results for Non-TMLE estimator (=BCAUSS, by default):")
    print(dict)

    # print("--------------------------")
    # print("Results for TMLE estimator:")
    # print(tmle_dict)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_base_dir', type=str, help="path to directory LBIDD" , default='Jobs-20230510')
    args = parser.parse_args()
    main(jobs_dir=args.data_base_dir)
    
