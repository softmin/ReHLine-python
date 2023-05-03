import numpy as np
import warnings
import sys
from benchopt import BaseSolver, safe_import_context
sys.path.insert(0, '../') # the code for ReHLine is in this directory
import numpy as np
from _rehline import ReHLine

n = 500
d = 100
lam1=0.01
lam2=0.01

tau = abs(np.random.randn())
X = np.random.randn(n, d)
beta0 = np.random.randn(d)
out = X@beta0
y = out + 0.1*np.random.randn(n)

clf = ReHLine(C=1./n/lam2, verbose=False, tol=1e-7)
clf.make_ReLHLoss(X=X, y=y, loss={'name':'huber', 'tau':tau})
X_fake=clf.append_l1(X, l1_pen=lam1/lam2)
clf.fit(X_fake)

def obj(beta):
    out = np.dot(X, beta)
    res = abs(y - out)
    loss = np.where(res>tau, tau * res - tau**2 / 2, res**2/2)
    reg = lam1 * np.sum(np.abs(beta)) + lam2 * np.sum(beta**2) / 2
    return np.mean(loss) + reg

for n_iter in [10, 50, 100, 500, 100000]:
    clf.max_iter = n_iter
    clf.fit(X_fake)
    print('obj: %.3f' %obj(clf.coef_))
