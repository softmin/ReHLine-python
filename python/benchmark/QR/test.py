import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
import sys

sys.path.insert(0, '../../') # the code for ReHLine is in this directory
from _rehline import ReHLine

sys.path.insert(0, '../') # the code for benchmark is in this directory
from benchmark_base import _check_constraints, obj, _append_result

from sklearn.svm import LinearSVC
import time
import seaborn as sns
import matplotlib.pyplot as plt

def run_example(df, n=3000, d=50, random_state=0):

    X, y = make_regression(n_samples=n, n_features=d, n_informative=40, random_state=random_state)
    x_sensitive = X[:,np.random.randint(d)]
    x_sensitive = x_sensitive - np.mean(x_sensitive)

    ## Fairness constraints
    cov_thresh = .3
    A = np.repeat([x_sensitive @ X], repeats=[2], axis=0) / n
    A[1] = -A[1]
    b = np.array([cov_thresh, cov_thresh])

    loss_function = "svm_linear"
    cons_params = {}
    cons_params["EPS"] = 1e-4
    cons_params["cons_type"] = 0

    for C in [10.]:
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
            obj0 = obj(C=C0, coef_=clf.w, X=X, y=y, loss='svm')
            b0 = np.sum((A @ clf.w - b)**2)

            if clf.result[2] == "Converged" or clf.result[2] == "optimal":
                pass
            else:
                ## skip the case that clf fails
                continue
        except:
            continue

        ## check constraints
        if _check_constraints(A, b0, clf.w):
            df = _append_result(df, n, d, C, 0., "fairSVM-dccp", et-st)
        else:
            continue

        print('-'*25)
        print('Minmizing via ReHLine')
        print('-'*25)

        for max_iter_tmp in [50, 100, 300, 500, 1000, 2000, 3000, 10000, 20000, 50000, 70000, 100000, 200000, 500000]:

            # For the proposed ReMin method
            cue = ReHLine(C=C0, verbose=False, tol=1e-15, max_iter=max_iter_tmp)
            cue.make_ReLHLoss(X=X, y=y, loss={'name':'SVM'})
            cue.A = A
            cue.b = b
            st = time.time()
            cue.fit(X=X)
            et = time.time()

            err_tmp = obj(C=C0, coef_=cue.coef_, X=X, y=y, loss='svm') - obj0
            print('ReHLine with err: %.6f' %err_tmp)
            if err_tmp <= 1e-5:
                if _check_constraints(A, b, cue.coef_):
                    df = _append_result(df, n, d, C, err_tmp, "ReHLine", et-st)
                    break
                else:
                    continue
    return df


if __name__ == '__main__':

    df = {'err': [], 'method': [], 'time': [], 'neg_log_err': [], 'n': [], 'dim': [], 'C': []}
    for n in [1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]:
        for d in [50]:
            for i in range(20):
                df = run_example(df, n=n, d=d, random_state=i)

    df = pd.DataFrame(df)

    sns.set_theme(style="white")

    # sns.relplot(
    #     data=df, x="n", y="time",
    #     col="C", hue="method", style="method", kind="scatter")

    sns.lineplot(data=df,
        x="n", y="time", hue="method", style="method",
        markers=True, dashes=True)
    plt.tight_layout()
    plt.show()