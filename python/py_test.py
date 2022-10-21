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

df = {'err': [], 'method': [], 'time': [], '-log_err': []}

for i in range(10):
    X, y = make_classification(n_samples=2000, n_features=50, random_state=0)
    y = 2*y - 1
    C = .1

    ### Test for SVM

    ## generate dataset for `L3-solver`
    n, d = X.shape
    U = -(C*y).reshape(1,-1)
    L = U.shape[0]
    V = (C*np.array(np.ones(n))).reshape(1,-1)

    ## True solution
    clf = LinearSVC(C=C, loss='hinge', fit_intercept=False, random_state=0, tol=1e-8, max_iter=1000000)
    clf.fit(X, y)
    sol = clf.coef_.flatten()


    for max_iter_tmp in range(70, 1000, 5):
        ## solve by ReMin
        cue = ReMin(U=U, V=V, verbose=False, tol=1e-5, max_iter=max_iter_tmp)
        st = time.time()
        cue.fit(X)
        et = time.time()
        eps_remin = np.sqrt(np.sum((cue.coef_ - sol)**2))
        
        if eps_remin < 1e-4:
            break

        df['method'].append('ReMin')
        df['time'].append(et - st)
        df['err'].append(eps_remin)
        df['-log_err'].append(-np.log10(eps_remin))
        

        # 537 µs ± 10.3 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

    for max_iter_tmp in range(200, 1000, 5):

        ## solve by liblinear
        clf = LinearSVC(C=C, loss='hinge', fit_intercept=False, random_state=0, tol=1e-5, max_iter=max_iter_tmp)
        st = time.time()
        clf.fit(X, y)
        et = time.time()
        eps_liblinear = np.sqrt(np.sum((clf.coef_ - sol)**2))

        if eps_liblinear < 1e-4:
            print(max_iter_tmp)
            break

        df['method'].append('liblinear')
        df['time'].append(et - st)
        df['err'].append(eps_liblinear)
        df['-log_err'].append(-np.log10(eps_liblinear))
        # 644 µs ± 4.98 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)


## Show the results

df = pd.DataFrame(df)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")

# Load an example dataset with long-form data

# Plot the responses for different events and regions
sns.lineplot(x="-log_err", y="time",
             hue="method", style="method", markers=True,
             data=df)
plt.show()