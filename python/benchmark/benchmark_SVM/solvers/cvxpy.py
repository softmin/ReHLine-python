import warnings
import sys
from benchopt import BaseSolver, safe_import_context
import numpy as np

with safe_import_context() as import_ctx:
    import cvxpy as cp

class Solver(BaseSolver):
    name = 'cvxpy'

    install_cmd = 'pip'
    requirements = ['cvxpy']

    parameters = {
        # 'solver': ['ECOS', 'OSQP', 'CVXOPT', 'MOSEK', 'SCS'],
        'solver': ['ECOS', 'OSQP', 'MOSEK', 'SCS'],
    }

    parameter_template = "solver={solver}"

    def set_objective(self, X, y, C):
        self.X, self.y, self.C = X, y, C
        self.n, self.d = X.shape
        ## SVM parameters

        self.w = cp.Variable(self.d)
        self.xi = cp.Variable(self.n)

        objective = cp.Minimize(1/2*cp.square(cp.norm(self.w))+ C * cp.sum(self.xi) / self.n)
        constraints = [cp.multiply(y, X @ self.w) + self.xi >=1, self.xi>=0]
        self.prob = cp.Problem(objective, constraints)
        

    def run(self, n_iter):
        solver = self.solver
        if solver in ['OSQP']:
            algo_tol = {'eps_abs': 1e-3, 
                        'eps_rel': 1e-3}
        elif solver in ['ECOS']:
            algo_tol = {'abstol': 1e-3, 
                        'reltol': 1e-3, 
                        'feastol': 1e-3, 
                        'abstol_inacc': 1e-3, 
                        'reltol_inacc': 1e-3, 
                        'feastol_inacc': 1e-3}
        elif solver in ['CVXOPT']:
            algo_tol = {'abstol': 1e-3, 
                        'reltol': 1e-3, 
                        'feastol': 1e-3}
        elif solver in ['SCS']:
            algo_tol = {'eps': 1e-3}
        else:
            algo_tol={}

        for key in algo_tol.keys():
            algo_tol[key] = algo_tol[key]* 2 **(-n_iter)
        
        if solver in ['OSQP']:
            result = self.prob.solve(solver=solver,  **algo_tol)
        elif solver in ['ECOS', 'CVXOPT', 'SCS']:
            result = self.prob.solve(solver=solver, max_iters=10000, **algo_tol)
        elif solver in ['MOSEK']:
            result = self.prob.solve(solver=solver, mosek_params={'MSK_DPAR_OPTIMIZER_MAX_TIME':2**n_iter})
        else:
            result = self.prob.solve(solver=solver)

    def get_result(self):
        return self.w.value

# benchopt run ./benchmark_linear_svm_binary_classif_no_intercept -d simulated --max-runs 15 --n-repetitions 10