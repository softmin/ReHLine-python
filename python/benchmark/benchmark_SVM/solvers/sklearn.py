import warnings

from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.svm import LinearSVC


class Solver(BaseSolver):
    name = 'sklearn'

    install_cmd = 'conda'
    requirements = ['scikit-learn']

    parameters = {
        'solver': ['liblinear'],
    }
    parameter_template = "solver={solver}"

    def set_objective(self, X, y, C):
        self.X, self.y, self.C = X, y, C
        n, d = X.shape
        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        self.clf = LinearSVC(C=self.C/n, penalty='l2', dual=True,
                             fit_intercept=False, tol=1e-12,
                             loss='hinge')

    def run(self, n_iter):
        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.y)

    def get_result(self):
        return self.clf.coef_.flatten()
