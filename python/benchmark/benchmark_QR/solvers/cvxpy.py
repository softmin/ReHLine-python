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
        'solver': ['ECOS', 'MOSEK', 'SCS', 'CPLEX', 'GUROBI'],
    }

    parameter_template = "solver={solver}"

    def set_objective(self, X, y, q, lam1, lam2):
        self.X, self.y, self.q, self.lam1, self.lam2 = X, y, q, lam1, lam2
        self.n, self.d = X.shape
        ## SVM parameters

        self.w = cp.Variable(self.d + 1)
        tau = cp.Parameter()
        tau.value = q

        X_fake = np.ones((self.n, self.d+1))
        X_fake[:,:-1] = X
        res = y - X_fake @ self.w
        loss = 0.5 * cp.abs(res) + (tau - 0.5) * res
        reg = lam1*cp.sum(cp.abs(self.w)) + 1/2*lam2*cp.square(cp.norm(self.w))
        
        objective = cp.Minimize(cp.sum(loss) / self.n + reg)
        self.prob = cp.Problem(objective)
        

    def run(self, n_iter):
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