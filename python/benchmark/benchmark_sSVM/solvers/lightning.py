import warnings

from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    from lightning.classification import SDCAClassifier, SAGClassifier, SAGAClassifier, SVRGClassifier

class Solver(BaseSolver):
    name = 'lightning'

    install_cmd = 'pip'
    requirements = ['sklearn-contrib-lightning']

    parameters = {
        'solver': ['SDCA', 'SAG', 'SAGA', 'SVRG'],
    }
    parameter_template = "algo={solver}"

    def set_objective(self, X, y, C):
        self.X, self.y, self.C = X, y, C
        n, d = X.shape
        if self.solver == 'SDCA':
            self.clf = SDCAClassifier(loss='smooth_hinge', alpha=1.0, l1_ratio=0.0, gamma=1.0, tol=1e-20)
        elif self.solver == 'SAG':
            self.clf = SAGClassifier(loss='smooth_hinge', alpha=1.0, gamma=1.0, tol=1e-20)
        elif self.solver == 'SAGA':
            self.clf = SAGAClassifier(loss='smooth_hinge', alpha=1.0, gamma=1.0, tol=1e-20)
        elif self.solver == 'SVRG':
            self.clf = SAGAClassifier(loss='smooth_hinge', alpha=1.0, gamma=1.0, tol=1e-20)
    def run(self, n_iter):
        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.y)

    def get_result(self):
        return self.clf.coef_.flatten()
