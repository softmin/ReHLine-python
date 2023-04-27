import warnings
import sys
sys.path.insert(0, './fair_classification/') # the code for fair classification is in this directory
from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    import sys
    sys.path.insert(0, './fair_classification/')
    from linear_clf_pref_fairness import LinearClf


class Solver(BaseSolver):
    name = 'dccp'

    install_cmd = 'pip'
    requirements = ['scikit-learn', 'dccp']

    # parameters = {
    #     'solver': ['liblinear'],
    # }
    # parameter_template = "solver={solver}"

    def set_objective(self, X, y, Z, C, rho):
        self.X, self.y, self.Z, self.C, self.rho = X, y, Z, C, rho
        loss_function = "svm_linear"
        lam = .5/self.C

        self.clf = LinearClf(loss_function, cov_thresh=rho, lam=lam, train_multiple=False, max_iters=100)

        # self.clf = LinearSVC(C=self.C, penalty='l2', dual=True,
        #                      fit_intercept=False, tol=1e-12,
        #                      loss='hinge')

    def run(self, n_iter):
        
        cons_params = {}
        cons_params["EPS"] = 1e-12
        cons_params["cons_type"] = 0

        self.clf.max_iter = n_iter
        # self.clf.fit(self.X, self.y)
        self.clf.fit(self.X, self.y, self.Z, cons_params)

    def get_result(self):
        return self.clf.w.flatten()
