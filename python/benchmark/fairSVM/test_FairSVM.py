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

np.random.seed(0)

n, d = 3000, 50
X, y = make_classification(n_samples=n, n_features=d, n_informative=40, random_state=0)
y = 2*y - 1
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

C = .5
lam = .5/C/n

## fitted by FairSVM original paper
## GitHub: https://github.com/mbilalzafar/fair-classification
## Paper: http://proceedings.mlr.press/v54/zafar17a/zafar17a-supp.pdf
print('-'*25)
print('Minmizing via DCCP')
print('-'*25)

clf = LinearClf(loss_function, cov_thresh=cov_thresh, lam=lam, train_multiple=False, max_iters=1000)
st = time.time()
clf.fit(X, y, x_sensitive, cons_params)
et = time.time()
obj0 = obj(C=C, coef_=clf.w, X=X, y=y, loss='svm')
dccp_time = et - st


print('-'*25)
print('Minmizing via ReHLine')
print('-'*25)


# For the proposed ReMin method
cue = ReMin(C=C, verbose=True, tol=1e-10, max_iter=1000000)
cue.make_ReLoss(X=X, y=y, loss={'name':'SVM'})
cue.A = A
cue.b = b
st = time.time()
cue.fit(X=X)
et = time.time()
ReHLine_time = et - st
obj_ReH = obj(C=C, coef_=cue.coef_, X=X, y=y, loss='svm')


print('DCCP obj: %.5f time: %.2f' %(obj0, dccp_time))
print('ReHLine obj: %.5f time: %.2f' %(obj_ReH, ReHLine_time))


# np.savez('fairSVM_slow', X=X, y=y, U=cue.U, V=cue.V, A=cue.A, b=cue.b)