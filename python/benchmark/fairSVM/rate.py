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

n, d = 3000, 50
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

C = 10.
lam = .5/C/n

## fitted by FairSVM original paper
## GitHub: https://github.com/mbilalzafar/fair-classification
## Paper: http://proceedings.mlr.press/v54/zafar17a/zafar17a-supp.pdf
# print('-'*25)
# print('Minmizing via DCCP')
# print('-'*25)

# clf = LinearClf(loss_function, cov_thresh=cov_thresh, lam=lam, train_multiple=False, max_iters=100)
# st = time.time()
# clf.fit(X, y, x_sensitive, cons_params)
# et = time.time()
# obj0 = obj(C=C, coef_=clf.w, X=X, y=y, loss='svm')


print('-'*25)
print('Minmizing via ReMin with A \beta <= b')
print('-'*25)

## ReMin Optimal
cue = ReMin(C=C, verbose=True, tol=1e-8, max_iter=10000000)
cue.make_ReLoss(X=X, y=y, loss={'name':'SVM'})
cue.A = A
cue.b = b
st = time.time()
cue.fit(X=X)
et = time.time()
obj0 = cue.opt_result_.dual_objfns[-1]
err_path = -obj0 + np.array(cue.opt_result_.dual_objfns)

plt.plot(np.log10(err_path+1e-10))
plt.show()

print('-'*25)
print('Minmizing via ReMin with no constraints')
print('-'*25)

## ReMin Optimal
cue = ReMin(C=C, verbose=True, tol=1e-8, max_iter=10000000)
cue.make_ReLoss(X=X, y=y, loss={'name':'SVM'})
st = time.time()
cue.fit(X=X)
et = time.time()
obj0 = cue.opt_result_.dual_objfns[-1]
err_path = -obj0 + np.array(cue.opt_result_.dual_objfns)

plt.plot(np.log10(err_path+1e-10))
plt.show()