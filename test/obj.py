
import numpy as np

def P_obj(U, v, beta):
    score = np.einsum('kij,j->ki', U, beta) + v
    return np.sum(np.mean(np.maximum(score, 0), 1), 0) + .5 * np.sum(beta**2)

def D_obj(P, q, x):
    """Return the value of the Standard form QP using the current value of x."""
    return 0.5 * np.dot(x, np.dot(P, x)) + np.dot(q, x)

def check_c(A, b, beta):
    assert all(A @ beta >= b)
