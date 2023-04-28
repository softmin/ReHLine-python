### Test for SVM
import sys
sys.path.append('../../')

import unittest
import numpy as np
from numpy.testing import assert_array_equal
import io 
from sklearn.datasets import make_classification
from _rehline import ReHLine
# from _l3solver import ReMin
from sklearn.svm import LinearSVC
import pandas as pd
import time
from qp_solver import qp_admm, qp_cvxpy
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

df = {'err': [], 'method': [], 'time': [], 'neg_log_err': [], 'n': [], 'dim': [], 'C': []}

## Primal objective
def obj(C, coef_, X, y):
    return np.sum(C*np.maximum(1 - y*( X @ coef_), 0)) + .5*np.sum(coef_**2)

iter_range = range(200, 10000, 200)
tol_range = 10**np.array(list(reversed(np.arange(-4,-1,.3))))

for n in [2000, 4000, 6000]:
    for d in [50]:
        print('--'*20)
        print('simulation for n: %d, d: %d' %(n ,d))
        print('--'*20)

        for i in range(25):
            print('#'*15)
            X, y = make_classification(n_samples=n, n_features=d, n_informative=40, random_state=i)
            y = 2*y - 1
            C = 10./n

            ## generate dataset for `ReHLine`
            n, d = X.shape
            U = -(C*y).reshape(1,-1)
            L = U.shape[0]
            V = (C*np.array(np.ones(n))).reshape(1,-1)

            ## parameters for a general QP
            Ux = np.array([U.T*X])
            mat_Ux = np.reshape(Ux, (-1, d)).T
            P = np.dot(mat_Ux.T, mat_Ux) + 1e-8*np.eye(L*n)
            q = -V.flatten()
            lb = np.zeros(L*n)
            ub = np.ones(L*n)

            ## True solution
            clf_true = LinearSVC(C=C, loss='hinge', fit_intercept=False, random_state=0, tol=1e-10, max_iter=1000000)
            clf_true.fit(X, y)
            sol = clf_true.coef_
            obj0 = obj(C, sol, X, y)

            ## find the starting max_iter 
            # cue = ReMin(U=U, V=V, verbose=False, tol=1e-4, max_iter=100000)
            # cue.fit(X)
            # ReMin_base = cue.n_iter_*.7
            # print('ReMin base iteraction is: %d' %ReMin_base)

            # clf = LinearSVC(C=C, loss='hinge', fit_intercept=False, random_state=1, tol=1e-3, max_iter=100000)
            # clf.fit(X, y)
            # liblinear_base = clf.n_iter_*.5
            # print('LibLinear base iteraction is: %d' %liblinear_base)

            ## Increase max_iter until diff obj reaches tol
            for max_iter_tmp in iter_range:
            # for max_iter_tmp in range(int(ReMin_base), 10000, 20):
                
                ## solve by ReHLine
                cue = ReHLine(U=U, V=V, verbose=False, tol=1e-6, max_iter=max_iter_tmp)
                st = time.time()
                cue.fit(X)
                et = time.time()
                # eps_remin = np.sqrt(np.sum((cue.coef_ - sol)**2))
                eps_remin = abs(obj(C, cue.coef_, X, y) - obj0)
                
                if eps_remin < 1e-5:
                    break

                if eps_remin < 1e-4:
                    df['method'].append('ReHLine')
                    df['time'].append(et - st)
                    df['err'].append(eps_remin)
                    df['neg_log_err'].append(-np.log10(eps_remin))
                    df['n'].append(n)
                    df['dim'].append(d)
                    df['C'].append(C)
                    print('ReHLine is done with #Iters: %d' %max_iter_tmp)
                    # break


            # for tol_tmp in tol_range:

            #     ## solve by CVXOPT
            #     st = time.time()
            #     Dsol_cvxopt, __ = qp_cvxpy(P, q, lb, ub, max_iter=max_iter_tmp, abstol=tol_tmp, reltol=tol_tmp, feastol=tol_tmp)
            #     Psol_cvxopt = - mat_Ux.dot(Dsol_cvxopt)
            #     et = time.time()
            #     eps_cvxopt = abs(obj(C, Psol_cvxopt, X, y) - obj0)

            #     if eps_cvxopt < 1e-5:
            #         break

            #     if eps_cvxopt < 1e-4:
            #         df['method'].append('CVXOPT')
            #         df['time'].append(et - st)
            #         df['err'].append(eps_cvxopt)
            #         df['neg_log_err'].append(-np.log10(eps_cvxopt))
            #         df['n'].append(n)
            #         df['dim'].append(d)
            #         df['C'].append(C)
            #         print('CVXOPT is done with #Iters: %d' %max_iter_tmp)

            # for max_iter_tmp in iter_range:

            #     ## solve by ADMM
            #     st = time.time()
            #     Dsol_admm, __ = qp_cvxpy(P, q, lb, ub, solver='SCS', max_iters=max_iter_tmp)
            #     Psol_admm = - mat_Ux.dot(Dsol_admm)
            #     et = time.time()
            #     eps_admm = abs(obj(C, Psol_admm, X, y) - obj0)

            #     if eps_admm < 1e-5:
            #         break

            #     if eps_admm < 1e-4:
            #         df['method'].append('ADMM')
            #         df['time'].append(et - st)
            #         df['err'].append(eps_admm)
            #         df['neg_log_err'].append(-np.log10(eps_admm))
            #         df['n'].append(n)
            #         df['dim'].append(d)
            #         df['C'].append(C)
            #         print('ADMM is done with #Iters: %d' %max_iter_tmp)

            # for max_iter_tmp in range(int(liblinear_base), 10000, 20):
            for max_iter_tmp in iter_range:

                ## solve by liblinear
                clf = LinearSVC(C=C, loss='hinge', fit_intercept=False, random_state=0, tol=1e-6, max_iter=max_iter_tmp)
                st = time.time()
                clf.fit(X, y)
                et = time.time()
                # eps_liblinear = np.sqrt(np.sum((clf.coef_.flatten() - sol)**2))
                eps_liblinear = abs(obj(C, clf.coef_.flatten(), X, y) - obj0)

                if eps_liblinear < 1e-5:
                    break

                if eps_liblinear < 1e-4:
                    df['method'].append('liblinear')
                    df['time'].append(et - st)
                    df['err'].append(eps_liblinear)
                    df['neg_log_err'].append(-np.log10(eps_liblinear))
                    df['n'].append(n)
                    df['dim'].append(d)
                    df['C'].append(C)
                    print('liblinear is done with #Iters: %d' %max_iter_tmp)
                    # break

## Show the results
df = pd.DataFrame(df)


import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="white")

sns.relplot(
    data=df, x="neg_log_err", y="time",
    col="n", hue="method", style="method",
    kind="scatter"
)

# sns.stripplot(x="n", y="time",
#              dodge=True, alpha=.25, zorder=1,
#              hue="method",
#              data=df)

plt.show()