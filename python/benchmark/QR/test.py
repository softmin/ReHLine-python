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
from funs import QR_cvx
n, d, random_state = 1000, 50, 0

iter_range = np.arange(500, 50000, 500)
tol_range = 10**np.arange(-10, -3, 0.3)

def run_example(df, n=3000, d=10, random_state=0):
    np.random.seed(random_state)
    q = np.random.rand()

    X = np.random.randn(n,d)
    beta0 = np.random.randn(d)
    y = X@beta0 + np.random.rand()*np.random.randn(n)

    out0 = X@beta0
    obj0 = obj(C=1., y=y, out=out0, loss={'name':'QR', 'qt': [q]})

    l2_term = np.sum(beta0**2) / 2
    C = l2_term / obj0
    lam1 = l2_term / np.sum(abs(beta0))
    # X, y = make_regression(n_samples=n, n_features=d, n_informative=d, noise=np.random.rand(), random_state=random_state)
    ## true solution
    # https://www.cvxpy.org/tutorial/advanced/index.html
    beta0, _ = QR_cvx(np.c_[X, np.ones(n)], y, lam1=lam1/C, lam2=1./C, q=q, solver_config={'solver':'ECOS', 'max_iters':10000})
    
    out0 = X@beta0[:-1] + beta0[-1]
    obj0 = obj(C=C, y=y, out=out0, loss={'name':'QR', 'qt': [q]}) + 0.5*np.sum(beta0**2) + lam1*np.sum(abs(beta0))
    

    print('-'*25)
    print('Minmizing via ECOS')
    print('-'*25)

    for max_iter_tmp in iter_range:        
        beta_ECOS, time_ECOS = QR_cvx(np.c_[X, np.ones(n)], y, lam1=lam1/C, lam2=1./C, q=q, solver_config={'solver':'ECOS', 'max_iters':max_iter_tmp})

        out_ECOS = beta_ECOS[-1] + X @ beta_ECOS[:-1]
        obj_ECOS = obj(C=C, y=y, out=out_ECOS, loss={'name':'QR', 'qt': [q]}) + 0.5*np.sum(beta_ECOS**2) + lam1*np.sum(abs(beta_ECOS))

        err_tmp = obj_ECOS - obj0

        if err_tmp <= 1e-5:
            df = _append_result(df, n, d, C, err_tmp, "ECOS", time_ECOS)
            print('ECOS ends with max_iter: %d' %max_iter_tmp)
            break

    print('-'*25)
    print('Minmizing via ReHLine')
    print('-'*25)

    for max_iter_tmp in iter_range:
        cue = ReHLine(C=C, verbose=False, tol=1e-7, max_iter=max_iter_tmp)
        X_fake=cue.make_ReLHLoss(X=X, y=y, loss={'name':'QR', 'qt':[q]})
        X_fake=cue.append_l1(X_fake, l1_pen=lam1)

        st = time.time()
        cue.fit(X_fake)
        et = time.time()

        out_rehl = X @ cue.coef_[:-1] + cue.coef_[-1]
        obj_rehl = obj(C=C, y=y, out=out_rehl, loss={'name':'QR', 'qt': [q]}) + 0.5*np.sum(cue.coef_**2) + lam1*np.sum(abs(cue.coef_))

        err_tmp = obj_rehl - obj0
        print('ReHLine: err: %.3f' %err_tmp)

        if err_tmp <= 1e-5:
            df = _append_result(df, n, d, C, err_tmp, "ReHLine", et-st)
            print('ReHLine ends with max_iter: %d' %max_iter_tmp)
            break
        
    return df


if __name__ == '__main__':

    df = {'err': [], 'method': [], 'time': [], 'neg_log_err': [], 'n': [], 'dim': [], 'C': []}
    for n in [1000]:
    # for n in [50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000]:
        for d in [100]:
            for i in range(20):
                df = run_example(df, n=n, d=d, random_state=i)
                print(pd.DataFrame(df).to_string())

    df = pd.DataFrame(df)

    sns.set_theme(style="white")

    # sns.relplot(
    #     data=df, x="n", y="time",
    #     col="C", hue="method", style="method", kind="scatter")

    sns.lmplot(x="n", y="time", data=df, hue="method",
            lowess=True, scatter_kws= {'alpha': 0.5})

    # sns.lineplot(data=df,
    #     x="n", y="time", hue="method", style="method",
    #     markers=True, dashes=True)
    plt.tight_layout()
    plt.legend(loc='upper left', ncol=2, borderaxespad=3)
    plt.show()