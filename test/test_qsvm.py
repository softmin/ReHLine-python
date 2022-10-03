## SVM on simulated dataset
import numpy as np
import pandas as pd
from qp_solver import qp_admm, qp_cvxpy
import cvxpy as cp
from numpy.linalg import norm
from sklearn.datasets import make_classification
from obj import P_obj

X, y = make_classification(n_samples=100, n_features=20, random_state=0)
y = 2*y - 1
C = .5

### Test for SVM

## generate dataset for `L3-solver`
n, d = X.shape
S = -(np.sqrt(2*C)*y).reshape(1,-1)
H = S.shape[0]
T = (np.sqrt(2*C)*np.array(np.ones(n))).reshape(1,-1)
tau = np.inf

np.savez('./dataset/sim/exp_qsvm', X=X, y=y, S=S, T=T, tau=tau)

qsvm_data = np.load('/home/ben/github/L3-solver/test/dataset/sim/exp_qsvm.npz')
X, y, S, T, tau = qsvm_data['X'], qsvm_data['y'], qsvm_data['S'], qsvm_data['T'], qsvm_data['tau']

# test
assert S.shape[1] == n
assert T.shape[0] == H
assert T.shape[1] == n

## solution by liblinear
from sklearn.svm import LinearSVC
clf = LinearSVC(C=C, loss='squared_hinge', fit_intercept=False, random_state=0, tol=1e-6, max_iter=1000000)
clf.fit(X, y)
sol = clf.coef_.flatten()

## parameters for a general QP
Sx = np.array([S.T*X])
mat_Sx = np.reshape(Sx, (-1, d)).T
P = np.dot(mat_Sx.T, mat_Sx) + np.eye(H*n)
q = -T.flatten()
lb = np.zeros(H*n)
ub = np.ones(H*n)*np.inf

## solution by ADMM
Dsol_admm, __ = qp_admm(P, q, lb, ub, atol=1e-5, rtol=1e-5)
Psol_admm = - mat_Sx.dot(Dsol_admm)

## solution by CVXOPT
Dsol_cvxopt, __ = qp_cvxpy(P, q, lb, ub)
Psol_cvxopt = - mat_Sx.dot(Dsol_cvxopt)

## Check solution

## Compare the results for different methods
print('diff: admm-liblinear: %.4f' %norm(Psol_admm - sol))
print('diff: cvxopt-liblinear: %.4f' %norm(Psol_cvxopt - sol))
print('diff: cvxopt-admm: %.4f' %norm(Psol_cvxopt - Psol_admm))

# diff: admm-liblinear: 0.0003
# diff: cvxopt-liblinear: 0.0000
# diff: cvxopt-admm: 0.0003

## liblinear
# array([ 0.32647621,  0.19297425, -0.0196456 ,  0.63073258,  0.50440456,
#        -0.02638266,  0.01512483, -0.00537292,  0.13561768, -0.19491773,
#         0.48096541,  0.16616315, -0.07322499, -0.00776993,  0.21303252,
#         0.03834003,  0.18055742,  0.14129267,  0.19285945, -0.30080286])

## ADMM
# array([ 0.32635619,  0.19289979, -0.01962633,  0.6305745 ,  0.50425768,
#        -0.02631939,  0.01513435, -0.00532471,  0.13557696, -0.19487383,
#         0.48085707,  0.16612203, -0.07321543, -0.00773782,  0.2129585 ,
#         0.03833739,  0.1805581 ,  0.1412345 ,  0.19278989, -0.30071373])

## CVXOPT
# array([ 0.32647621,  0.19297423, -0.0196456 ,  0.63073258,  0.50440454,
#        -0.02638264,  0.01512483, -0.00537293,  0.13561767, -0.19491769,
#         0.48096541,  0.16616317, -0.073225  , -0.00776992,  0.21303252,
#         0.03834004,  0.18055741,  0.14129266,  0.19285943, -0.30080288])

## check objective function
obj_true = C * np.sum(np.maximum(1 - y[:,np.newaxis] * X @ sol, 0)**2) + .5*np.sum(sol**2)

assert obj_true == P_obj(X=X, beta=sol, U=0, V=0, S=S, T=T, tau=np.inf)

print('obj: linlinear: %.4f' %P_obj(X=X, beta=sol, U=0, V=0, S=S, T=T, tau=np.inf))
# obj: linlinear: 11.6221

print('obj: admm: %.4f' %P_obj(X=X, beta=Psol_admm, U=0, V=0, S=S, T=T, tau=np.inf))
# obj: admm: 11.6225

print('obj: cvxopt: %.4f' %P_obj(X=X, beta=Psol_cvxopt, U=0, V=0, S=S, T=T, tau=np.inf))
# obj: cvxopt: 11.6221

