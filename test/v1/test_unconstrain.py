## Test constrained L3 

import numpy as np
import pandas as pd
from qp_solver import qp_admm, qp_cvxpy
from numpy.linalg import norm
import cvxpy as cp
from obj import P_obj


data = np.load('./dataset/sim/exp1.npz')
U, v = data['U'], data['v']
K, n, d = U.shape

assert v.shape[0] == K
assert v.shape[1] == n

mat_U = np.reshape(U, (-1, d)).T

## solution by prime-CVXOPT
beta = cp.Variable(d)
pi = cp.Variable(K*n)

primal_obj = cp.Minimize(cp.sum(pi) + (1/2)*cp.norm(beta, p=2)**2)
primal_constraints = [pi >= mat_U.T @ beta + v.flatten(),
                     pi >= np.zeros(K*n)]

prob = cp.Problem(primal_obj, primal_constraints)
prob.solve()
sol = beta.value

## data for a dual QP
P = np.dot(mat_U.T, mat_U) + 1e-4*np.eye(K*n)
q = v.flatten()
lb = np.zeros(K*n)
ub = np.ones(K*n)

## solution by ADMM
Dsol_admm, __ = qp_admm(P, -q, lb, ub, atol=1e-5, rtol=1e-5)
Psol_admm = -mat_U.dot(Dsol_admm)

## solution by CVXOPT
Dsol_cvxopt, __ = qp_cvxpy(P, -q, lb, ub)
Psol_cvxopt = -mat_U.dot(Dsol_cvxopt)

## Compare the results for different methods
print('diff: admm-primal: %.4f' %norm(Psol_admm - sol))
print('diff: cvxopt-primal: %.4f' %norm(Psol_cvxopt - sol))
print('diff: cvxopt-admm: %.4f' %norm(Psol_cvxopt - Psol_admm))

## Primal CVXOPT
# array([ 0.01379876, -0.01203579,  0.00990021, -0.00749876, -0.00288679,
#        -0.0139854 , -0.02737326,  0.01965769,  0.02126496, -0.04225128])

## Psol_admm
# array([ 0.01369427, -0.01200735,  0.01003851, -0.00707008, -0.00283105,
#        -0.01397377, -0.027546  ,  0.01963784,  0.02101462, -0.04186331])

## Psol_cvxopt
# array([ 0.01384035, -0.01206327,  0.00983727, -0.00751609, -0.00288951,
#        -0.01397833, -0.02735297,  0.0196676 ,  0.02123235, -0.04222114])

## check objective function
print('obj: linlinear: %.5f' %P_obj(U, v, sol))
print('obj: admm: %.5f' %P_obj(U, v, Psol_admm))
print('obj: cvxopt: %.5f' %P_obj(U, v, Psol_cvxopt))

# obj: linlinear: 2.00872
# obj: admm: 2.00870
# obj: cvxopt: 2.00872