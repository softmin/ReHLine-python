### Test for SVM

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import sys

sys.path.insert(0, '../../') # the code for ReHLine is in this directory
from _l3solver import ReMin_solver, ReMin

sys.path.insert(0, '../') # the code for ReHLine is in this directory
from base import _check_constraints, obj

from sklearn.svm import LinearSVC

n, d = 2000, 50

X, y = make_classification(n_samples=n, n_features=d, n_informative=40, random_state=0)
y = 2*y - 1
C = .001/n

## True solution
cue = ReMin(C=C, verbose=False, tol=1e-9, max_iter=1000000)
cue.make_ReLoss(X=X, y=y, loss={'name':'SVM'})
cue.fit(X)
sol = cue.coef_
obj0 = obj(C, sol, X, y)


## solve by liblinear
clf = LinearSVC(C=C, loss='hinge', fit_intercept=False, random_state=0, tol=1e-6, max_iter=10000)
clf.fit(X, y)
obj_libL = obj(C, clf.coef_.flatten(), X, y)

print("obj_ReMin: %.3f; obj_libL: %.3f" %(obj0, obj_libL))