"""Build small native test problems from the named loss families."""

import numpy as np

from rehline import _make_loss_rehline_param

from .core import _native_problem, objective
from .objectives import objective_value


def routine_problem(c):
    """Convert named losses and check their full independent objectives."""
    losses = {
        "mse": {"name": "MSE"},
        "mae": {"name": "MAE"},
        "quantile": {"name": "QR", "qt": c["qt"]},
        "quantile_eps": {"name": "check_eps", "qt": c["qt"], "epsilon": c["epsilon"]},
        "huber": {"name": "huber", "tau": c["tau"]},
        "svr": {"name": "svr", "epsilon": c["epsilon"]},
        "hinge": {"name": "svm"},
        "smooth_hinge": {"name": "sSVM"},
        "squared_hinge": {"name": "squared SVM"},
    }
    native_case = c
    if c["family"] in losses:
        U, V, Tau, S, T = _make_loss_rehline_param(losses[c["family"]], c["X"], c["y"])
        converted = {
            name: value.reshape(-1, len(c["X"])) for name, value in zip(("U", "V", "Tau", "S", "T"), (U, V, Tau, S, T))
        }
        native_case = dict(c, **converted)
    p = _native_problem(native_case)
    # An incorrect loss conversion is a test setup error, never a budget miss.
    for beta in (np.zeros(c["X"].shape[1]), np.ones(c["X"].shape[1]), -np.ones(c["X"].shape[1])):
        if not np.isclose(objective_value(p, beta) * (1 - c["l1_ratio"]), objective(c, beta), rtol=1e-12, atol=1e-10):
            raise ValueError("Native test problem disagrees with the full original objective")
    return p
