import warnings
import sys
from benchopt import BaseSolver, safe_import_context
sys.path.insert(0, '../') # the code for ReHLine is in this directory
import numpy as np

with safe_import_context() as import_ctx:
    from _rehline import ReHLine

class Solver(BaseSolver):
    name = 'rehline'

    install_cmd = 'pip'
    requirements = ['scikit-learn']

    parameters = {
        'shrink': [True, False],
    }
    parameter_template = "shrink={shrink}"

    def set_objective(self, X, y, q, lam1, lam2):
        self.X, self.y, self.q, self.lam1, self.lam2 = X, y, q, lam1, lam2
        n, d = X.shape

        self.clf = ReHLine(C=1./n/lam2, tol=1e-5, shrink=self.shrink, verbose=False)
        X_fake=self.clf.make_ReLHLoss(X=X, y=y, loss={'name':'QR', 'qt':[q]})
        self.X_fake=self.clf.append_l1(X_fake, l1_pen=lam1/lam2)

    def run(self, n_iter):
        self.clf.max_iter = n_iter
        self.clf.fit(self.X_fake)

    def get_result(self):
        return self.clf.coef_
