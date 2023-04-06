"""Test ReHL-Loss function."""

# Authors: Ben Dai <bendai@cuhk.edu.hk>
#          Yixuan Qiu <yixuanq@gmail.com>

# License: MIT License

import numpy as np
import sys
sys.path.insert(0, '../') # the code for ReHLine is in this directory
from _rehline import ReHLine
from rehloss import ReHLoss
from sklearn.datasets import make_classification, make_regression

## Test QR (check loss)
def check_loss(y, out, kappa=[0.25, 0.75]):
    loss_mat = np.zeros((len(y), len(kappa)))
    for i, kappa_tmp in enumerate(kappa):
        loss_mat[:,i] = kappa_tmp * np.maximum(y - out, 0) + (1 - kappa_tmp) * np.maximum(out - y, 0)
    return loss_mat

X, y = make_regression(n_samples=1000, n_features=5, random_state=1024)
qt_clf = ReHLine(C=1.)
X_fake = qt_clf.make_ReLHLoss(X=X,y=y,loss={'name':'QR', 'qt':[0.25, 0.75]})
n_qt = len(qt_clf.loss['qt'])

z = np.random.randn(len(y))
loss_mat1 = check_loss(y, z)

loss_mat2 = qt_clf.call_ReLHLoss(input=np.tile(z, n_qt))
loss_mat2 = loss_mat2.reshape(n_qt,-1).swapaxes(0,1)

assert ((loss_mat1 - loss_mat2) < 1e-10).all(), "QR: ReHLoss is wrong."
