## Test `l3-solver` in various datasets

## SVM in breast_cancer dataset
import numpy as np
import pandas as pd
from qp_solver import qp_admm, qp_cvxpy
import cvxpy as cp
from numpy.linalg import norm

def P_obj(U, v, beta):
    return np.sum(np.maximum(np.einsum('kij,j->ki', U, beta) + v, 0)) + 0.5*np.sum(beta**2)

## load breast cancer dataset; sol is the solution from liblinear
svm_dataset = np.load('./dataset/svm/svm.npz')
X, y, U, v, sol = svm_dataset['X'], svm_dataset['y'], svm_dataset['U'], svm_dataset['v'], svm_dataset['sol']

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
Dsol_admm, __ = qp_admm(P, -q, lb, ub, max_iter=100000)
Psol_admm = mat_U.dot(Dsol_admm)

## solution by CVXOPT
Dsol_cvxopt, __ = qp_cvxpy(P, -q, lb, ub, max_iter=100000)
Psol_cvxopt = mat_U.dot(Dsol_cvxopt)

print('diff: admm-liblinear: %.4f' %norm(Psol_admm - sol))
print('diff: cvxopt-liblinear: %.4f' %norm(Psol_cvxopt - sol))
print('diff: cvxopt-admm: %.4f' %norm(Psol_cvxopt - Psol_admm))

## SVM on simulated dataset
import numpy as np
import pandas as pd
from qp_solver import qp_admm, qp_cvxpy
import cvxpy as cp
from numpy.linalg import norm
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=20)
y = 2*y - 1

C = 1.
from sklearn.svm import LinearSVC
clf = LinearSVC(C=C, loss='hinge', fit_intercept=False, random_state=0, tol=1e-6, max_iter=1000000)
clf.fit(X, y)
sol = clf.coef_.flatten()

## generate dataset for `L3-solver`
U = C*np.array([-y[:,np.newaxis]*X])
K, n, d = U.shape
v = C*np.array([np.ones(n)])
Xy = y[:,np.newaxis]*X

assert len(sol) == d
assert v.shape[0] == K
assert v.shape[1] == n

# mat_U = np.reshape(U, (d, -1))
mat_U = U[0].T
# mat_U = Xy.T

P = np.dot(mat_U.T, mat_U) + 1e-8*np.eye(K*n)
q = v.flatten()
lb = np.zeros(K*n)
ub = np.ones(K*n)

## solution by ADMM
Dsol_admm, __ = qp_admm(P, -q, lb, ub, max_iter=100000)
Psol_admm = -mat_U.dot(Dsol_admm)

## solution by CVXOPT
Dsol_cvxopt, __ = qp_cvxpy(P, -q, lb, ub, max_iter=100000)
Psol_cvxopt = -mat_U.dot(Dsol_cvxopt)

print('diff: admm-liblinear: %.4f' %norm(Psol_admm - sol))
print('diff: cvxopt-liblinear: %.4f' %norm(Psol_cvxopt - sol))
print('diff: cvxopt-admm: %.4f' %norm(Psol_cvxopt - Psol_admm))
