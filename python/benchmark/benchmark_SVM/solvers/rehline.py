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

    def set_objective(self, X, y, C):
        self.X, self.y, self.C = X, y, C
        n, d = X.shape

        self.clf = ReHLine(C=self.C/n, tol=1e-12, shrink=self.shrink, verbose=False)
        self.clf.make_ReLHLoss(X=X, y=y, loss={'name':'SVM'})

    def run(self, n_iter):
        self.clf.max_iter = n_iter
        self.clf.fit(self.X)

    def get_result(self):
        return self.clf.coef_
