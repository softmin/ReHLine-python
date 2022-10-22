### Test for SVM

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import unittest
import math
import l3solver
import numpy as np
from numpy.testing import assert_array_equal
import io 
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.datasets import make_classification
from _l3solver import ReMin_solver, ReMin
from sklearn.svm import LinearSVC
import pandas as pd
import time

df = {'err': [], 'method': [], 'time': [], '-log_err': [], 'n': [], 'dim': []}

## Primal objective
def obj(C, coef_, X, y):
    return np.sum(C*np.maximum(1 - y*( X @ coef_), 0)) + .5*np.sum(coef_**2)


for n in [1000, 2500, 4000]:
    for d in [50]:
        print('--'*20)
        print('simulation for n: %d, d: %d' %(n ,d))
        print('--'*20)

        for i in range(25):
            print('#'*15)
            X, y = make_classification(n_samples=n, n_features=d, n_informative=40, random_state=i)
            y = 2*y - 1
            C = 10./n

            ## generate dataset for `L3-solver`
            n, d = X.shape
            U = -(C*y).reshape(1,-1)
            L = U.shape[0]
            V = (C*np.array(np.ones(n))).reshape(1,-1)

            ## True solution
            cue = ReMin(U=U, V=V, verbose=False, tol=1e-9, max_iter=1000000)
            cue.fit(X)
            sol = cue.coef_
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
            for max_iter_tmp in range(500, 10000, 100):
            # for max_iter_tmp in range(int(ReMin_base), 10000, 20):
                ## solve by ReMin
                cue = ReMin(U=U, V=V, verbose=False, tol=1e-6, max_iter=max_iter_tmp)
                st = time.time()
                cue.fit(X)
                et = time.time()
                # eps_remin = np.sqrt(np.sum((cue.coef_ - sol)**2))
                eps_remin = abs(obj(C, cue.coef_, X, y) - obj0)
                
                if eps_remin < 1e-5:
                    break

                if eps_remin < 1e-4:
                    df['method'].append('ReMin')
                    df['time'].append(et - st)
                    df['err'].append(eps_remin)
                    df['-log_err'].append(-np.log10(eps_remin))
                    df['n'].append(n)
                    df['dim'].append(d)
                    df['C'].append(C)
                    print('ReMin is done with #Iters: %d' %max_iter_tmp)
                    # break

            # for max_iter_tmp in range(int(liblinear_base), 10000, 20):

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
                    df['-log_err'].append(-np.log10(eps_liblinear))
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
    data=df, x="-log_err", y="time",
    col="n", hue="method", style="method",
    kind="scatter"
)

# sns.stripplot(x="n", y="time",
#              dodge=True, alpha=.25, zorder=1,
#              hue="method",
#              data=df)

plt.show()