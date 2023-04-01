import numpy as np
import pandas as pd

def _check_constraints(A, b, coef_, eps=1e-5):
    return all(A @ coef_ <= b + eps)

def obj(C, coef_, X, y, loss='svm'):
    if loss=='svm':
        return np.sum(C*np.maximum(1 - y*( X @ coef_), 0)) + .5*np.sum(coef_**2)
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