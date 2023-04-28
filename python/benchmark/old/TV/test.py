import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '../../') # the code for ReHLine is in this directory
from _rehline import ReHLine

sys.path.insert(0, '../') # the code for benchmark is in this directory
from benchmark_base import _check_constraints, obj, _append_result

import time
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.pyplot import figure
from skimage.restoration import denoise_tv_chambolle, denoise_tv_bregman
from scipy.sparse import csr_array
tol = 1e-2

n, d, random_state = 5000, 100, 8
max_iter_tmp=1000
iter_range = 10**np.arange(2, 8, 0.2)
iter_range = np.array(iter_range, dtype=int)
tol_range = 10**np.arange(-10, -3, 0.3)
C_range = np.logspace(-3, 3)

def run_example(df, n=3000, d=10, random_state=0):
    np.random.seed(random_state)

    x0 = np.ones(n)

    for j in range(3):
        idx = np.random.randint(n)
        k = np.random.randint(10)

        x0[int(idx/2):idx] = k * x0[int(idx/2):idx]

    y = x0 + np.random.randn(n)
    C = np.random.choice(C_range) / n

    try:
        y_chambolle = denoise_tv_chambolle(y, weight=C)
        # y_bregman = denoise_tv_bregman(y, weight=C)

    except:
        return df

    print('-'*25)
    print('Minmizing via ReHLine')
    print('-'*25)

    try:
        for max_iter_tmp in iter_range:
            cue = ReHLine(C=C, verbose=False, tol=1e-7, max_iter=max_iter_tmp)
            # X = np.zeros((n-1, n))
            X = csr_array((n-1, n))
            X[:,:-1].setdiag(-1)
            X[:,1:].setdiag(-1)

            # np.fill_diagonal(X[:,:-1], -1)
            # np.fill_diagonal(X[:,1:], 1)

            cue.make_ReLHLoss(X=X, y=y, loss={'name':'TV'})

            st = time.time()
            cue.fit(X)
            et = time.time()

            out_rehl = cue.coef_ + y
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