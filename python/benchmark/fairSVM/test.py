import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.datasets import make_classification
import sys
sys.path.insert(0, './fair_classification/') # the code for fair classification is in this directory

from linear_clf_pref_fairness import LinearClf

n, d = 1000, 50

X, y = make_classification(n_samples=n, n_features=d, n_informative=40, random_state=0)
x_sensitive = np.random.randn(n)

loss_function = "svm_linear"
cons_params = {}
cons_params["EPS"] = 1e-4
cons_params["cons_type"] = 0


clf = LinearClf(loss_function, lam=1e-5, train_multiple=False)
clf.fit(X, y, x_sensitive, cons_params)