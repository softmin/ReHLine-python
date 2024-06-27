import numpy as np
from numpy import linalg as LA
from rehline import ReHLineQuad, ReHLineLinear


"""Optimization of the problem
    \min_{w \in \R^d} 1/2*w'Gw - c'w + \sum_{i=1}^n ReLU(x_i' w - 3) + 
                                         + \sum_{i=1}^n ReHU_1(-2*x_i' w + 1)]
               s.t. w_j \in [-3, 3]
"""


def relu(x):
    return np.maximum(x, 0)

from scipy.special import huber
def rehu(x, cut=1):
    cut = cut * np.ones_like(x)
    u = np.maximum(x, 0)
    return huber(cut, u)


Z = np.array([
    [2.0, -1.0, 3.0],
    [-2.0, 1.0, 4.0],
    [5.0, 4.0, -2.0]])
Z = np.eye(3)
G = Z @ Z.T
c = np.array([2.0, 3.0, -1.0])
# c = np.zeros(3)
X = np.array([
       [-0.01882744, -1.3031826 , -0.9499709 ],
       [-0.13484637,  1.40204165, -0.47233268],
       [-0.79247053, -0.76120738, -0.16519892],
       [-0.22379675, -1.57109984, -0.95758743],
       [-0.77531044, -1.51580597,  0.05978438]])
n, d = len(X), len(c)
L, H = 1, 1
U = np.ones((L, n))
V = np.ones((L, n)) * -3.0
S = np.ones((H, n)) * -2.0
T = np.ones((H, n))
Tau = np.ones((H, n))
A = np.r_[np.eye(d), -np.eye(d)]
b = np.ones(2*d)*3.0


def objfn(w): 
    score = 1/2 * w.T @ G @ w - c.T @ w
    loss = 0
    for i in range(n):
        for l in range(L):
            loss += relu(U[l, i]*X[i, :].dot(w) + V[l, i])
        for h in range(H):
            loss += rehu(S[h, i]*X[i, :].dot(w) + T[h, i], cut=Tau[h, i])
    return score + loss

# Solution provided by RehLineQuad
rehline_quad = ReHLineQuad(
    loss='custom', C=1, U=U, V=V, S=S, T=T, Tau=Tau, G=G, mu=c, A=A, b=b)
rehline_quad.fit(X=X)
w_rq = rehline_quad.coef_
print("Solution provided by ReHLineQuad:", w_rq, f"(objfn: {objfn(w_rq)})")


# Solution provided by ReHLineLinear
Zinvt = LA.inv(Z).T
c_tilde = Zinvt.T @ c
A_tilde = A @ Zinvt
# x_i' Zt^-1 Zt w ==> i-th row: X @ Zt^{-1}
X_tilde = X @ Zinvt
rehline_linear = ReHLineLinear(
    loss='custom', C=1, U=U, V=V, S=S, T=T, Tau=Tau, mu=c_tilde, A=A_tilde, b=b)
rehline_linear.fit(X=X_tilde)
# shift it back
w_rl0 = rehline_linear.coef_
w_rl = Zinvt @ w_rl0
print("Solution provided by ReHLineLinear:", w_rl, f"(objfn: {objfn(w_rl)})")

assert np.isclose(2.0*LA.norm(w_rl - w_rq) / (LA.norm(w_rl + LA.norm(w_rq))), 0.0, atol=1e-4)