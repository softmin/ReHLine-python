## Test constrained L3 

import numpy as np
import pandas as pd
from qp_solver import qp_admm, qp_cvxpy
from numpy.linalg import norm
import cvxpy as cp
from obj import P_obj

data = np.load('./dataset/sim/exp2.npz')
U, v, A, b = data['U'], data['v'], data['A'], data['b']
K, n, d = U.shape
L = len(A)

assert v.shape[0] == K
assert v.shape[1] == n
assert len(b) == L
assert A.shape[1] == d

mat_U = np.reshape(U, (-1, d)).T

## solution by prime-CVXOPT
beta = cp.Variable(d)
pi = cp.Variable(K*n)

primal_obj = cp.Minimize(cp.sum(pi) + (1/2)*cp.norm(beta, p=2)**2)
primal_constraints = [A @ beta + b >= np.zeros(L), 
                     pi >= mat_U.T @ beta + v.flatten(),
                     pi >= np.zeros(K*n)]

prob = cp.Problem(primal_obj, primal_constraints)
prob.solve()
sol = beta.value

## generate data for dual QP
P1 = np.dot(A, A.T) + 1e-6*np.eye(L) 
P2 = np.dot(mat_U.T, mat_U) + 1e-6*np.eye(K*n) 
P12 = - A @ mat_U

q1 = b
q2 = -v.flatten()
lb1 = np.zeros(L)
lb2 = np.zeros(K*n)
ub1 = 1e10*np.ones(L)
ub2 = np.ones(K*n)

## solution by dual boxed-QP
P = np.block([[P1, P12],
             [P12.T, P2]])
q = np.block([q1, q2])
lb = np.block([lb1, lb2])
ub = np.block([ub1, ub2])

## solution by ADMM
Dsol_admm, __ = qp_admm(P, q, lb, ub, atol=1e-5, rtol=1e-5)
alpha_admm, lam_admm = Dsol_admm[:L], Dsol_admm[L:]
Psol_admm = A.T @ alpha_admm - mat_U @ lam_admm

## solution by CVXOPT
Dsol_cvxopt, __ = qp_cvxpy(P, q, lb, ub)
alpha_cp, lam_cp = Dsol_cvxopt[:L], Dsol_cvxopt[L:]
Psol_cvxopt = A.T @ alpha_cp - mat_U @ lam_cp

## Compare the results for different methods
print('diff: admm-primal: %.4f' %norm(Psol_admm - sol))
print('diff: cvxopt-primal: %.4f' %norm(Psol_cvxopt - sol))
print('diff: cvxopt-admm: %.4f' %norm(Psol_cvxopt - Psol_admm))

## 
# Primal cvxopt 
# array([ 0.06176925, -0.22015066, -0.07568035,  0.78107721, -0.4443236 ,
#        -0.11707473,  0.11095109,  0.82554241,  0.0438759 ,  0.13180259])

# Psol_admm
# array([ 0.06098296, -0.2206119 , -0.07601194,  0.77894802, -0.44502981,
#        -0.1169704 ,  0.11139399,  0.82511667,  0.04364525,  0.13131839])

# Psol_cvxopt
# array([ 0.06136998, -0.21952933, -0.07635897,  0.78027488, -0.44306767,
#        -0.11653435,  0.11158851,  0.82619721,  0.04363481,  0.13233961])

print('obj: linlinear: %.5f' %P_obj(U, v, sol))
print('obj: admm: %.5f' %P_obj(U, v, Psol_admm))
print('obj: cvxopt: %.5f' %P_obj(U, v, Psol_cvxopt))

# obj: linlinear: 3.99687
# obj: admm: 3.99306
# obj: cvxopt: 3.99520