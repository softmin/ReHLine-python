import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import sys
sys.path.insert(0, './fair_classification/') # the code for fair classification is in this directory
from linear_clf_pref_fairness import LinearClf

sys.path.insert(0, '../../') # the code for ReHLine is in this directory
from _rehline import ReHLine

sys.path.insert(0, '../') # the code for ReHLine is in this directory
from benchmark_base import _check_constraints, obj, _append_result

from sklearn.svm import LinearSVC
import time
import seaborn as sns
import matplotlib.pyplot as plt
C_range = np.logspace(-3, 3)
iter_range = 10**np.arange(2, 8, 0.1)
iter_range = np.array(iter_range, dtype=int)
tol = 1e-2

n, d, random_state = 3000, 50, 0

def run_example(df, n=3000, d=50, random_state=0):

    X, y = make_classification(n_samples=n, n_features=d, n_informative=int(d/2), random_state=random_state)
    x_sensitive = X[:,np.random.randint(d)]
    x_sensitive = x_sensitive - np.mean(x_sensitive)

    ## Fairness constraints
    cov_thresh = np.random.rand()
    A = np.repeat([x_sensitive @ X], repeats=[2], axis=0) / n
    A[1] = -A[1]
    b = np.array([cov_thresh, cov_thresh])

    loss_function = "svm_linear"
    cons_params = {}
    cons_params["EPS"] = 1e-4
    cons_params["cons_type"] = 0

    C = np.random.choice(C_range)
    C0 = C/n
    lam = .5/C0/n

    ## fitted by FairSVM original paper
    ## GitHub: https://github.com/mbilalzafar/fair-classification
    ## Paper: http://proceedings.mlr.press/v54/zafar17a/zafar17a-supp.pdf
    print('-'*25)
    print('Minmizing via DCCP')
    print('-'*25)
    try:
        clf = LinearClf(loss_function, cov_thresh=cov_thresh, lam=lam, train_multiple=False, max_iters=100)
        st = time.time()
        clf.fit(X, y, x_sensitive, cons_params)
        et = time.time()
        out0 = X @ clf.w
        obj0 = obj(C=C0, y=y, out=out0, loss={'name': 'svm'}) + .5*np.sum(clf.w**2)
        b0 = np.sum((A @ clf.w - b)**2)

        if clf.result[2] == "Converged" or clf.result[2] == "optimal":
            ## check constraints
            if _check_constraints(A, b0, clf.w):
                df = _append_result(df, n, d, C, 0., "fairSVM-dccp", et-st)
            else:
                pass
    except:
        pass

    print('-'*25)
    print('Minmizing via ReHLine')
    print('-'*25)

    for max_iter_tmp in iter_range:

        # For the proposed ReMin method
        cue = ReHLine(C=C0, verbose=False, tol=1e-15, max_iter=max_iter_tmp)
        cue.make_ReLHLoss(X=X, y=y, loss={'name':'SVM'})
        cue.A = A
        cue.b = b
        st = time.time()
        cue.fit(X=X)
        et = time.time()

        out_tmp = X@cue.coef_ 
        obj_tmp = obj(C=C0, y=y, out=out_tmp, loss={'name': 'svm'}) + .5*np.sum(cue.coef_**2)
        err_tmp = obj_tmp - obj0

        print('ReHLine with err: %.6f' %err_tmp)
        if err_tmp <= tol*max(1, obj0):
            if _check_constraints(A, b, cue.coef_):
                df = _append_result(df, n, d, C, err_tmp, "ReHLine", et-st)
                break
            else:
                continue
    return df


if __name__ == '__main__':

    df = {'err': [], 'method': [], 'time': [], 'neg_log_err': [], 'n': [], 'dim': [], 'C': []}
    for n in [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]:
    # for n in [1000, 2000, 5000]:
        for d in [100]:
            for i in range(20):
                df = run_example(df, n=n, d=d, random_state=i)

    df = pd.DataFrame(df)

    sns.set_theme(style="white")

    # sns.relplot(
    #     data=df, x="n", y="time",
    #     col="C", hue="method", style="method", kind="scatter")

    sns.lmplot(data=df, x='n', y='time', col='method', hue='method', robust=True,
                legend=True, truncate=True, facet_kws={"sharey":False}, height=8)
    plt.tight_layout()
    plt.show()

    # sns.lineplot(data=df,
    #     x="n", y="time", hue="method", style="method",
    #     markers=True, dashes=True)
    # plt.tight_layout()
    # plt.show()