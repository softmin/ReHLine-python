
import numpy as np

def P_obj(X, beta, U=0, V=0, S=0, T=0, tau=0):
    score = np.dot(X, beta)
    z_relu = score[np.newaxis,:]*U + V
    relu_loss = np.sum(relu(z_relu.flatten()))
    z_rehu = score[np.newaxis,:]*S + T
    rehu_loss = np.sum(rehu(z_rehu.flatten(), tau=tau))
    # relu_loss = np.sum(relu(np.einsum('li,i->li', U, score) + V), 0)
    # rehu_loss = np.sum(rehu(np.einsum('hi,i->hi', S, score) + T), 0)
    return relu_loss + rehu_loss + .5 * np.sum(beta**2)

def D_obj(P, q, x):
    """Return the value of the Standard form QP using the current value of x."""
    return 0.5 * np.dot(x, np.dot(P, x)) + np.dot(q, x)

def check_c(A, b, beta):
    assert all(A @ beta >= b)

def relu(z):
    return np.maximum(z, 0)

def rehu(z, tau):
    score = z**2 / 2
    score[z<=0] = 0
    if tau == np.inf:
        pass
    else:
        score[z>tau] = tau*(z[z>tau]-tau/2)
    return score
    