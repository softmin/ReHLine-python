import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
import sys
from statsmodels.regression.quantile_regression import QuantReg
sys.path.insert(0, '../../') # the code for ReHLine is in this directory
from _rehline import ReHLine

sys.path.insert(0, '../') # the code for benchmark is in this directory
from benchmark_base import _check_constraints, obj, _append_result

import time
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from funs import cvxQR_elastic, cvxQR_l2
from matplotlib.pyplot import figure
tol = 1e-2

n, d, random_state = 5000, 100, 8

iter_range = 10**np.arange(2, 8, 0.2)
iter_range = np.array(iter_range, dtype=int)
tol_range = 10**np.arange(-10, -3, 0.3)
C_range = np.logspace(-3, 3)

def run_example(df, n=3000, d=10, random_state=0):
    np.random.seed(random_state)
    q = np.random.rand()

    X = np.random.randn(n,d)
    beta0 = np.random.randn(d)
    y = X@beta0 + 0.1*np.random.randn(n)
    # X, y, beta0 = make_regression(n_samples=n, n_features=d, n_informative=int(d/2), noise=np.random.rand(), coef=True, random_state=random_state)

    # out0 = X@beta0
    # obj0 = obj(C=1., y=y, out=out0, loss={'name':'QR', 'qt': [q]})

    # l2_term = np.sum(beta0**2) / 2
    # C = l2_term / obj0
    C = np.random.choice(C_range) / n

    ## true solution
    # https://www.cvxpy.org/tutorial/advanced/index.html
    try:
        beta0, _, _ = cvxQR_l2(np.c_[X, np.ones(n)], y, lam2=1./C, q=q, solver_config={'solver':'ECOS', 'max_iters':10000})
        
        out0 = X@beta0[:-1] + beta0[-1]
        obj0 = obj(C=C, y=y, out=out0, loss={'name':'QR', 'qt': [q]}) + 0.5*np.sum(beta0**2)
    
    except:
        return df

    print('-'*25)
    print('Minmizing via ECOS')
    print('-'*25)
    ## 'max_iters', 'abstol', 'reltol', 'feastol', 'abstol_inacc', 'reltol_inacc', 'feastol_inacc'
    try:
        for max_iter_tmp in iter_range:        
            beta_ECOS, time_ECOS, res_ECOS = cvxQR_l2(np.c_[X, np.ones(n)], y, lam2=1./C, q=q, 
                    solver_config={'solver':'ECOS', 'reltol_inacc':1e-2, 'abstol_inacc': 1e-2})

            out_ECOS = beta_ECOS[-1] + X @ beta_ECOS[:-1]
            obj_ECOS = obj(C=C, y=y, out=out_ECOS, loss={'name':'QR', 'qt': [q]}) + 0.5*np.sum(beta_ECOS**2)

            err_tmp = (obj_ECOS - obj0) / obj0

            if err_tmp <= tol*max(1, obj0):
                df = _append_result(df, n, d, C, err_tmp, "ECOS", time_ECOS)
                print('ECOS ends with max_iter: %d' %max_iter_tmp)
                break
    except:
        pass

    # print('-'*25)
    # print('Minmizing via SCS')
    # print('-'*25)

    # try:
    #     for max_iter_tmp in iter_range:        
    #         beta_ECOS, time_ECOS = cvxQR_l2(np.c_[X, np.ones(n)], y, lam2=1./C, q=q, solver_config={'solver':'ECOS', 'max_iters':max_iter_tmp})

    #         out_ECOS = beta_ECOS[-1] + X @ beta_ECOS[:-1]
    #         obj_ECOS = obj(C=C, y=y, out=out_ECOS, loss={'name':'QR', 'qt': [q]}) + 0.5*np.sum(beta_ECOS**2)

    #         err_tmp = (obj_ECOS - obj0) / obj0

    #         if err_tmp <= tol:
    #             df = _append_result(df, n, d, C, err_tmp, "ECOS", time_ECOS)
    #             print('ECOS ends with max_iter: %d' %max_iter_tmp)
    #             break
    # except:
    #     pass

    print('-'*25)
    print('Minmizing via ReHLine')
    print('-'*25)

    try:
        for max_iter_tmp in iter_range:
            cue = ReHLine(C=C, verbose=False, tol=1e-7, max_iter=max_iter_tmp)
            X_fake=cue.make_ReLHLoss(X=X, y=y, loss={'name':'QR', 'qt':[q]})

            st = time.time()
            cue.fit(X_fake)
            et = time.time()

            out_rehl = X @ cue.coef_[:-1] + cue.coef_[-1]
            obj_rehl = obj(C=C, y=y, out=out_rehl, loss={'name':'QR', 'qt': [q]}) + 0.5*np.sum(cue.coef_**2)

            err_tmp = (obj_rehl - obj0) / obj0
            print('ReHLine: err: %.3f' %err_tmp)

            if err_tmp <= tol:
                df = _append_result(df, n, d, C, err_tmp, "ReHLine", et-st)
                print('ReHLine ends with max_iter: %d' %max_iter_tmp)
                break
    except:
        pass

    return df


if __name__ == '__main__':

    df = {'err': [], 'method': [], 'time': [], 'neg_log_err': [], 'n': [], 'dim': [], 'C': []}
    # for n in [10000, 20000, 30000, 40000, 50000, 60000]:
    for n in [5000]:
        for d in [100]:
            for i in range(10):
                df = run_example(df, n=n, d=d, random_state=i)
                print(pd.DataFrame(df).to_string())

    df = pd.DataFrame(df)
    sns.set_theme(style="white")

    # df.to_csv('perf.csv')

    sns.lmplot(data=df, x='n', y='time', col='method', hue='method', robust=True,
                legend=True, truncate=True, facet_kws={"sharey":False},
                hue_order=['ECOS', 'ReHLine'], height=8)
    plt.tight_layout()
    plt.show()