import numpy as np
import pandas as pd
import cvxpy as cvx
import time


def elastic_net(betas, lambda_l1, lambda_l2):
    return lambda_l1 * cvx.pnorm(betas, p=1) + \
           lambda_l2 / 2.0 * cvx.pnorm(betas, p=2)**2

def l2_pen(betas, lambda_l2):
    return lambda_l2 / 2.0 * cvx.pnorm(betas, p=2)**2

def cvxQR_elastic(X_train, y_train, lam1, lam2, q, solver_config):
    u = cvx.Variable(X_train.shape[0], nonneg=True)
    v = cvx.Variable(X_train.shape[0], nonneg=True)
    tau = cvx.Parameter()
    betas = cvx.Variable(shape=X_train.shape[1])
    lambda_l1 = cvx.Parameter(nonneg=True)
    lambda_l2 = cvx.Parameter(nonneg=True)

    tau.value = q
    lambda_l1.value = lam1
    lambda_l2.value = lam2

    objective = cvx.sum(tau * u) + cvx.sum((1-tau) * v) + elastic_net(betas, lambda_l1, lambda_l2)
    constraints = [cvx.matmul(X_train, betas) + u - v == y_train]

    problem = cvx.Problem(cvx.Minimize(objective), constraints)
    st = time.time()
    problem.solve(**solver_config)
    et = time.time()

    return betas.value, et - st

def cvxQR_l2(X_train, y_train, lam2, q, solver_config):
    u = cvx.Variable(X_train.shape[0], nonneg=True)
    v = cvx.Variable(X_train.shape[0], nonneg=True)
    tau = cvx.Parameter()
    betas = cvx.Variable(shape=X_train.shape[1])
    lambda_l2 = cvx.Parameter(nonneg=True)

    tau.value = q
    lambda_l2.value = lam2

    objective = cvx.sum(tau * u) + cvx.sum((1-tau) * v) + l2_pen(betas, lambda_l2)
    constraints = [cvx.matmul(X_train, betas) + u - v == y_train]

    problem = cvx.Problem(cvx.Minimize(objective), constraints)
    st = time.time()
    problem.solve(**solver_config)
    et = time.time()

    return betas.value, et - st
