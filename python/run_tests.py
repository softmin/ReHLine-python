import unittest

import L3solver
import numpy as np
from numpy.testing import assert_array_equal

import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=20, random_state=0)
y = 2*y - 1
C = .5

### Test for SVM

## generate dataset for `L3-solver`
n, d = X.shape
U = -(C*y).reshape(1,-1)
L = U.shape[0]
V = (C*np.array(np.ones(n))).reshape(1,-1)


A = np.empty( shape=(0, 0) )
b = np.empty( shape=(0) )
S = np.empty( shape=(0, 0) )
T = np.empty( shape=(0, 0) )

K = 0
H = 0

sol_beta = np.zeros(d, order='F')
sol_xi = np.zeros(K, order='F')
sol_Lambda = np.zeros((L, n), order='F')
sol_Gamma = np.zeros((H, n), order='F')
sol_Omega = np.zeros((H, n), order='F')

niter = 0
sol_dual_obj = 0.


L3solver.l3solver_py(X=X, A=A, b=b, U=U, V=V, S=S, T=T, 
                        tau=1., max_iter=1000, tol=1e-4, 
                        sol_beta=sol_beta, 
                        sol_xi=sol_xi,
                        sol_Lambda=sol_Lambda,
                        sol_Gamma=sol_Gamma,
                        sol_Omega=sol_Omega,
                        niter=niter,
                        sol_dual_obj=sol_dual_obj,
                        verbose=True)
