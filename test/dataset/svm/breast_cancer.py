## Test SVM based on Breast cancer wisconsin (diagnostic) dataset 

from sklearn.datasets import load_breast_cancer
import numpy as np

data = load_breast_cancer()
## load orginial dataset
X, y = data.data, data.target
y = 2*y - 1
n, d = X.shape
C = .0001
## generate dataset for `L3-solver`
U = C*np.array([-y[:,np.newaxis]*X])
v = C*np.array([np.ones(n)])
## note that in this case: A = 0; b = 0

## results from `linearSVC`
from sklearn.svm import LinearSVC
clf = LinearSVC(C=C, loss='hinge', fit_intercept=False, random_state=0, tol=1e-6, max_iter=1000000)
clf.fit(X, y)
sol = clf.coef_

import numpy as np
np.savez('svm', X=X, y=y, U=U, v=v, sol=sol)
