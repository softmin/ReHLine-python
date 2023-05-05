import warnings
import sys
from benchopt import BaseSolver, safe_import_context
import numpy as np

with safe_import_context() as import_ctx:
    import cvxpy as cp

class Solver(BaseSolver):
    name = 'cvxpy'
    stopping_strategy = "tolerance"

    install_cmd = 'pip'
    requirements = ['cvxpy']

    parameters = {
        # 'solver': ['ECOS', 'OSQP', 'CVXOPT', 'MOSEK', 'SCS'],
        'solver': ['ECOS', 'OSQP', 'MOSEK', 'SCS'],
    }

    parameter_template = "solver={solver}"

    def set_objective(self, X, y, tau, lam1, lam2):
        self.X, self.y, self.tau, self.lam1, self.lam2 = X, y, tau, lam1, lam2
        self.n, self.d = X.shape
        ## Huber parameters
        self.w = cp.Variable(self.d)
        res = y - X @ self.w
        loss = cp.huber(res, tau)
        if lam1 > 0:
            reg = lam1*cp.sum(cp.abs(self.w)) + 1/2*lam2*cp.square(cp.norm(self.w))
        else:
            reg = 1/2*lam2*cp.square(cp.norm(self.w))
        
        objective = cp.Minimize(cp.sum(loss) / self.n + reg)
        self.prob = cp.Problem(objective)
        

    def run(self, tol):
        solver = self.solver
        if solver in ['OSQP']:
            algo_tol = {'eps_abs': 1e-2, 
                        'eps_rel': 1e-2}
        elif solver in ['ECOS']:
            algo_tol = {'abstol': 1e-2, 
                        'reltol': 1e-2, 
                        'feastol': 1e-2, 
                        'abstol_inacc': 1e-2, 
                        'reltol_inacc': 1e-2, 
                        'feastol_inacc': 1e-2}
        elif solver in ['CVXOPT']:
            algo_tol = {'abstol': 1e-2, 
                        'reltol': 1e-2, 
                        'feastol': 1e-2}
        elif solver in ['SCS']:
            algo_tol = {'eps': 1e-2}
        elif solver in ['MOSEK']:
            algo_tol={'MSK_DPAR_INTPNT_QO_TOL_DFEAS': 1e-3,
                      'MSK_DPAR_INTPNT_QO_TOL_INFEAS': 1e-3,
                      'MSK_DPAR_INTPNT_QO_TOL_MU_RED': 1e-3,
                      'MSK_DPAR_INTPNT_QO_TOL_PFEAS': 1e-3,
                      'MSK_DPAR_INTPNT_QO_TOL_REL_GAP': 1e-3,
                      'MSK_DPAR_QCQO_REFORMULATE_REL_DROP_TOL': 1e-3
                      }
        else:
            algo_tol={ }

        for key in algo_tol.keys():
            algo_tol[key] = min(tol,1e-2)
        
        if solver in ['OSQP']:
            result = self.prob.solve(solver=solver,  **algo_tol)
        elif solver in ['ECOS', 'CVXOPT', 'SCS']:
            result = self.prob.solve(solver=solver, max_iters=10000, **algo_tol)
        elif solver in ['MOSEK']:
            result = self.prob.solve(solver=solver, mosek_params=algo_tol)
        else:
            result = self.prob.solve(solver=solver)

    def get_result(self):
        return self.w.value

# benchopt run ./benchmark_linear_svm_binary_classif_no_intercept -d simulated --max-runs 15 --n-repetitions 10