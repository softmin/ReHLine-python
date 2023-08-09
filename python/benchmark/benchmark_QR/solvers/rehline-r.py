from pathlib import Path

from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

    from rpy2 import robjects
    from rpy2.robjects import numpy2ri
    from benchopt.helpers.r_lang import import_func_from_r_file

    # Setup the system to allow rpy2 running
    R_FILE = str(Path(__file__).with_suffix('.R'))
    import_func_from_r_file(R_FILE)
    numpy2ri.activate()


class Solver(BaseSolver):
    name = 'rehline-r'

    install_cmd = 'conda'
    requirements = ['r-base', 'rpy2']
    stopping_strategy = 'iteration'

    def set_objective(self, X, y, q, lam1, lam2):
        self.X, self.y, self.q, self.lam1, self.lam2 = X, y, q, lam1, lam2
        self.elastic_qr_r = robjects.r['elastic_qr']

    def run(self, n_iter):
        beta = self.elastic_qr_r(
            self.X, self.y, kappa=self.q,
            lam1=self.lam1, lam2=self.lam2,
            max_iter=n_iter, tol=1e-8, verbose=0)
        self.beta = np.array(beta)

    def get_result(self):
        return self.beta.flatten()
