import numpy as np
import pandas as pd

def _check_constraints(A, b, coef_, eps=1e-5):
    return all(A @ coef_ + b <= eps)

def obj(C, y, out, loss={'name': 'svm'}):
    if loss['name']=='svm':
        return np.sum(C*np.maximum(1 - y*out, 0))
    elif loss['name']=='QR':
        kappa = loss['qt']
        loss_mat = np.zeros((len(y), len(kappa)))
        for i, kappa_tmp in enumerate(kappa):
            loss_mat[:,i] = kappa_tmp * np.maximum(y - out, 0) + (1 - kappa_tmp) * np.maximum(out - y, 0)
        return np.sum(loss_mat)
    elif loss['name']=='TV':
        return C*np.sum(abs(y - out))
    else:
        pass

def _append_result(df, n, d, C, err, method, time):
    df['method'].append(method)
    df['time'].append(time)
    df['err'].append(err)
    df['neg_log_err'].append(-np.log10(abs(err)+1e-10))
    df['n'].append(n)
    df['dim'].append(d)
    df['C'].append(C)
    return df
