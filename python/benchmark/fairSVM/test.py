import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import sys
sys.path.insert(0, './fair_classification/') # the code for fair classification is in this directory
from linear_clf_pref_fairness import LinearClf

sys.path.insert(0, '../../') # the code for ReHLine is in this directory
from _l3solver import ReMin_solver, ReMin

sys.path.insert(0, '../') # the code for ReHLine is in this directory
from base import _check_constraints, obj, _append_result

from sklearn.svm import LinearSVC
import time
import seaborn as sns
import matplotlib.pyplot as plt

def run_example(df, n=3000, d=50):

    X, y = make_classification(n_samples=n, n_features=d, n_informative=40, random_state=0)
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

    for C in [.001, .01, .01, .1, 1., 10., 100., 1000.]:
        C = C/n
        lam = .5/C/n

        ## fitted by FairSVM original paper
        ## GitHub: https://github.com/mbilalzafar/fair-classification
        ## Paper: http://proceedings.mlr.press/v54/zafar17a/zafar17a-supp.pdf
        print('-'*25)
        print('Minmizing via DCCP')
        print('-'*25)

        clf = LinearClf(loss_function, cov_thresh=cov_thresh, lam=lam, train_multiple=False, max_iters=100)
        st = time.time()
        clf.fit(X, y, x_sensitive, cons_params)
        et = time.time()
        obj0 = obj(C=C, coef_=clf.w, X=X, y=y, loss='svm')

        if clf.result[2] == "Converged" or clf.result[2] == "optimal":
            pass
        else:
            ## skip the case that clf fails
            continue

        err_tmp = 1e-10
        ## check constraints
        if _check_constraints(A, b, clf.w):
            df = _append_result(df, n, d, C, err_tmp, "fairSVM-dccp", et-st)
        else:
            continue

        print('-'*25)
        print('Minmizing via ReMin')
        print('-'*25)

        for max_iter_tmp in [100, 300, 500, 1000, 2000, 3000, 10000, 20000, 50000, 70000, 100000]:

            # For the proposed ReMin method
            cue = ReMin(C=C, verbose=False, tol=1e-10, max_iter=max_iter_tmp)
            cue.make_ReLoss(X=X, y=y, loss={'name':'SVM'})
            cue.A = A
            cue.b = b
            st = time.time()
            cue.fit(X=X)
            et = time.time()

            err_tmp = obj(C=C, coef_=cue.coef_, X=X, y=y, loss='svm') - obj0
            if err_tmp <= 1e-6:
                if _check_constraints(A, b, cue.coef_):
                    df = _append_result(df, n, d, C, err_tmp, "ReHLine", et-st)
                    break
                else:
                    continue
    return df


if __name__ == '__main__':

    df = {'err': [], 'method': [], 'time': [], 'neg_log_err': [], 'n': [], 'dim': [], 'C': []}
    for n in [2000, 4000]:
        for d in [50]:
            for i in range(25):
                df = run_example(df, n=n, d=d)

    df = pd.DataFrame(df)

    sns.set_theme(style="white")

    sns.relplot(
        data=df, x="n", y="time",
        col="C", hue="method", style="method", kind="scatter")

    # sns.stripplot(x="C", y="time",
    #              dodge=True, alpha=.25, zorder=1,
    #              hue="method",
    #              data=df)

    # sns.catplot(x="n", y="time", hue="method", col="C", aspect=.5,
    #             kind='strip')

    plt.show()