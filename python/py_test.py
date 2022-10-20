import unittest

import L3_solver
import numpy as np
from numpy.testing import assert_array_equal
import io 
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

sol_beta = np.zeros((d, 1))
sol_xi = np.zeros((K, 1))
sol_Lambda = np.zeros((L, n))
sol_Gamma = np.zeros((H, n))
sol_Omega = np.zeros((H, n))

niter = 0
sol_dual_obj = 0.

result = L3_solver.L3Result()

# .def_readwrite("beta", &L3Result::beta)
# .def_readwrite("xi", &L3Result::xi)
# .def_readwrite("Lambda", &L3Result::Lambda)
# .def_readwrite("Gamma", &L3Result::Gamma)
# .def_readwrite("Omega", &L3Result::Omega)
# .def_readwrite("niter", &L3Result::niter)
# .def_readwrite("dual_objfns", &L3Result::dual_objfns);
L3_solver.l3solver_internal(result, X, A, b, U, V, S, T, 1., 1000, 1e-4, True)
