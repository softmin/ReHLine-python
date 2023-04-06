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

iter_range = np.arange(10, 10000, 50)

def run_example(df, n=3000, d=10, random_state=0):
    np.random.seed(random_state)
    q = np.random.rand()

    X = np.random.randn(n,d)
    beta0 = np.random.randn(d)
    beta0 = beta0 / np.sum(beta0**2)
    y = X@beta0 + np.random.rand()*np.random.randn(n)

    # X, y = make_regression(n_samples=n, n_features=d, n_informative=d, noise=np.random.rand(), random_state=random_state)

    model = QuantReg(y, sm.add_constant(X))
    res = model.fit(q=q, max_iter=10000000, p_tol=1e-8)
    out_true = res.params[0] + X @ res.params[1:]
    
    obj0 = obj(C=1., y=y, out=out_true, loss={'name':'QR', 'qt': [q]})

    print('-'*25)
    print('Minmizing via QuantReg(iterative reweighted least squares)')
    print('-'*25)

    for max_iter_tmp in iter_range:
        model_tmp = QuantReg(y, sm.add_constant(X))
        
        st = time.time()
        res = model.fit(q=q, max_iter=max_iter_tmp, p_tol=1e-10)
        et = time.time()
        
        out_sm = res.params[0] + X @ res.params[1:]
        obj_sm = obj(C=1., y=y, out=out_sm, loss={'name':'QR', 'qt': [q]})

        err_tmp = obj_sm - obj0

        if err_tmp <= 1e-5:
            df = _append_result(df, n, d, 1.0, err_tmp, "QuantReg", et-st)
            break

    print('-'*25)
    print('Minmizing via ReHLine')
    print('-'*25)

    for max_iter_tmp in iter_range:
        cue = ReHLine(C=1., verbose=False, tol=1e-10, max_iter=max_iter_tmp)
        X_fake=cue.make_ReLHLoss(X=X, y=y, loss={'name':'QR', 'qt':[q]})

        st = time.time()
        cue.fit(X_fake)
        et = time.time()

        out_rehl = X @ cue.coef_[:-1] + cue.coef_[-1]
        obj_rehl = obj(C=1., y=y, out=out_rehl, loss={'name':'QR', 'qt': [q]})

        err_tmp = obj_rehl - obj0

        if err_tmp <= 1e-5:
            df = _append_result(df, n, d, 1.0, err_tmp, "ReHLine", et-st)
            break

    return df


if __name__ == '__main__':

    df = {'err': [], 'method': [], 'time': [], 'neg_log_err': [], 'n': [], 'dim': [], 'C': []}
    for n in [1000, 2000, 5000, 10000]:
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