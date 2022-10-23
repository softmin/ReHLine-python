## Solve QP 

import numpy as np
from numpy.linalg import inv, norm
import pandas as pd
import cvxpy as cp

def objective(P, q, x):
    """Return the value of the Standard form QP using the current value of x."""
    return 0.5 * np.dot(x, np.dot(P, x)) + np.dot(q, x)

def qp_admm(P, q, lb, ub,
            max_iter=100000,
            rho=1.0, 
            alpha=1.2,              
            atol=1e-5, 
            rtol=1e-5):

    n = P.shape[0]
    
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)
    
    history = []

    R = inv(P + rho * np.eye(n))
    
    for k in range(1, max_iter+1):
        x = np.dot(R, (z - u) - q)
        
        # z-update with relaxation
        z_old = z
        x_hat = alpha * x +(1 - alpha) * z_old
        z = np.minimum(ub, np.maximum(lb, x_hat + u))

        # u-update
        u = u + (x_hat - z)

        # diagnostics, and termination checks
        objval = objective(P, q, x)

        r_norm = norm(x - z)
        s_norm = norm(-rho * (z - z_old))
        eps_pri = np.sqrt(n) * atol + rtol * np.maximum(norm(x), norm(-z))
        eps_dual = np.sqrt(n)* atol + rtol * norm(rho*u)
        
        history.append({
            'objval'  : objval, 
            'r_norm'  : r_norm, 
            's_norm'  : s_norm,
            'eps_pri' : eps_pri,
            'eps_dual': eps_dual,
        })
        
        if r_norm < eps_pri and s_norm < eps_dual:
            print('Optimization terminated after {} iterations'.format(k))
            break;
        
    history = pd.DataFrame(history)
    return x, history

## Solve QP by cvxopt

def qp_cvxpy(P, q, lb, ub, solver='CVXOPT', **kwargs):
    n = P.shape[0]
    
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x), [x >= lb, x <= ub]) 
    prob.solve(solver=solver, **kwargs)
    x_opt = np.array(x.value).squeeze()
    return x_opt, prob
