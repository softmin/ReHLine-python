"""
ReHLine-python: test warmstart
v.0.0.6 - 2024-11-15
"""
# Authors: Ben Dai <bendai@cuhk.edu.hk>

import numpy as np

from rehline import ReHLine, plqERM_Ridge
from rehline._base import ReHLine_solver


def test_warmstart():
    np.random.seed(1024)
    
    # Simulate classification dataset
    n, d, C = 1000, 3, 0.5
    X = np.random.randn(n, d)
    beta0 = np.random.randn(d)
    y = np.sign(X.dot(beta0) + np.random.randn(n))

    # Test `ReHLine_solver`
    print("\nTesting ReHLine_solver")
    print("------------------------")
    
    # Test cold start
    print("Cold start:")
    U = -(C*y).reshape(1,-1)
    V = (C*np.array(np.ones(n))).reshape(1,-1)
    res = ReHLine_solver(X, U, V)
    print(f"Lambda shape: {res.Lambda.shape}")

    # Test warm start
    print("\nWarm start:")
    res_ws = ReHLine_solver(X, U, V, Lambda=res.Lambda)
    print(f"Lambda shape: {res_ws.Lambda.shape}")

    # Test `ReHLine`
    print("\nTesting ReHLine")
    print("-----------------")
    
    print("Cold start:")
    clf = ReHLine(verbose=1)
    clf.C = C
    clf.U = -y.reshape(1,-1)
    clf.V = np.array(np.ones(n)).reshape(1,-1)
    clf.fit(X)
    print(f"Coefficients: {clf.coef_}")

    print("\nWarm start:")
    clf.C = 2*C
    clf.warm_start = 1
    clf.fit(X)
    print(f"Coefficients: {clf.coef_}")

    # Test `plqERM_Ridge`
    print("\nTesting plqERM_Ridge")
    print("---------------------")
    
    print("Cold start:")
    clf = plqERM_Ridge(loss={'name': 'svm'}, C=C, verbose=1)
    clf.fit(X=X, y=y)
    print(f"Coefficients: {clf.coef_}")

    print("\nWarm start:")
    clf.C = 2*C
    clf.warm_start = 1
    clf.fit(X=X, y=y)
    print(f"Coefficients: {clf.coef_}")

if __name__ == "__main__":
    test_warmstart()
