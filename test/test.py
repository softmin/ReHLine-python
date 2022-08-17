## Test `l3-solver` in various datasets

## SVM in breast_cancer dataset
import numpy as np
import pandas as pd
from qp_solver import qp_admm, qp_cvxpy
import cvxpy as cp

## load breast cancer dataset; sol is the solution from liblinear
svm_dataset = np.load('./dataset/svm/svm.npz')
U, v, sol = svm_dataset['U'], svm_dataset['v'], svm_dataset['sol']

K, n, d = U.shape
assert sol.shape[1] == d
assert v.shape[0] == K
assert v.shape[1] == n

mat_U = np.reshape(U, (d, -1))
P = np.dot(mat_U.T, mat_U) + 1e-8*np.eye(K*n)
q = v.flatten()
lb = np.zeros(K*n)
ub = np.ones(K*n)

## solution by ADMM
Dsol_admm, __ = qp_admm(P, -q, lb, ub, max_iter=10000, atol=1e-4, rtol=1e-4)
Psol_admm = mat_U.dot(Dsol_admm)

## solution by CVXOPT
Dsol_cvxopt, __ = qp_cvxpy(P, -q, lb, ub, max_iter=10000, atol=1e-4, rtol=1e-4)
Psol_cvxopt = mat_U.dot(Dsol_cvxopt)

