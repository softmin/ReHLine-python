import numpy as np
import time
import matplotlib.pyplot as plt
from rehline import plqERM_Ridge
from rehline import _make_loss_rehline_param
from ._loss import ReHLoss


# def relu(x):
#     return np.maximum(0, x)

# def rehu(x, tau):
#     return np.minimum(np.maximum(0, x), tau)

# def compute_custom_loss(beta, X, y, loss):
#     """
#     Compute the custom ReLU-ReHU loss function.
#     """
#     U, V, Tau, S, T = _make_loss_rehline_param(loss, X, y)

#     relu_loss = 0
#     rehu_loss = 0

#     if U.size > 0 and V.size > 0:
#         relu_input = (U @ (X @ beta)) + V
#         relu_loss = np.sum(relu(relu_input))

#     if S.size > 0 and T.size > 0 and Tau.size > 0:
#         rehu_input = (S @ (X @ beta)) + T
#         rehu_loss = np.sum(rehu(rehu_input, Tau))

#     l2_norm = 0.5 * np.linalg.norm(beta) ** 2

#     return round(relu_loss + rehu_loss + l2_norm, 4)


def plqERM_path_sol(
    X,
    y,
    *,
    loss,
    constraint=None,
    eps=1e-3,
    n_Cs=100,
    Cs=None,
    max_iter=5000,
    tol=1e-4,
    verbose=0,
    shrink=1,
    warm_start=False,
    return_time=True,
    plot_path=False
):
    """
    Compute the PLQ Empirical Risk Minimization (ERM) path over a range of regularization parameters.
    This function evaluates the model's performance for different values of the regularization parameter 
    and provides structured benchmarking output.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training data, where each row represents a sample and each column represents a feature.

    y : ndarray of shape (n_samples,)
        Target values corresponding to the training data.

    loss : dict
        Dictionary containing parameters for the loss function used in the PLQ ERM problem.

    constraint : list of dict, optional
        List of constraints for the optimization problem. The default is None.

    eps : float, default=1e-3
        Path length parameter. A value of eps=1e-3 means that the ratio of the minimum to maximum 
        regularization parameter \( C \) will be \( 1e-3 \).

    n_Cs : int, default=100
        Number of values for the regularization parameter \( C \) along the path.

    Cs : array-like, default=None
        List of specific \( C \) values to evaluate. If None, \( C \) values are automatically generated.

    max_iter : int, default=5000
        Maximum number of iterations allowed for the solver.

    tol : float, default=1e-4
        Convergence tolerance for the optimization algorithm.

    verbose : int, default=0
        Level of verbosity for logging output. Higher values produce more detailed output.

    shrink : float, default=1
        Shrinkage parameter used in the optimization process.

    warm_start : bool, default=False
        If True, reuse the solution of the previous call to fit as the starting point for the next call.

    return_time : bool, default=True
        If True, return the execution times for each \( C \).

    plot_path : bool, default=False
        If True, plot the regularization path of the coefficients.

    Returns
    -------
    Cs : ndarray of shape (n_Cs,)
        The regularization parameters evaluated.

    times : list of float
        Execution time for each regularization parameter \( C \).

    n_iters : list of int
        The number of iterations taken to converge for each \( C \).

    loss_values : list of float
        Computed loss values for each \( C \).

    L2_norms : list of float
        L2 norms of the coefficients for each \( C \).

    coefs : ndarray of shape (n_features, n_Cs)
        Coefficients corresponding to each regularization parameter \( C \) along the path.

    """

    n_samples, n_features = X.shape

    if Cs is None:
        Cs = np.logspace(-2, 3, n_Cs)

    # Sort Cs to ensure computation starts from the smallest value
    Cs = np.sort(Cs)
    n_Cs = len(Cs)
    coefs = np.zeros((n_features, n_Cs))
    n_iters = []
    times = []
    loss_values = []
    L2_norms = []

    Lambda, Gamma, xi = None, None, None
    total_start = time.time()

    for i, C in enumerate(Cs):
        start_time = time.time()

        clf = plqERM_Ridge(
            loss=loss, constraint=constraint, C=C,
            max_iter=max_iter, tol=tol, shrink=shrink, verbose=verbose,
            warm_start=warm_start
        )

        if warm_start and Lambda is not None:
            clf.Lambda = Lambda
            clf.Gamma = Gamma
            clf.xi = xi

        clf.fit(X, y)
        coefs[:, i] = clf.coef_

        # Compute loss function parameters for ReHLoss
        U, V, Tau, S, T = _make_loss_rehline_param(loss, X, y)
        loss_obj = ReHLoss(U, V, S, T, Tau)
        loss_values.append(round(loss_obj(X), 4))
        L2_norms.append(round(np.linalg.norm(clf.coef_), 4))

        if warm_start:
            Lambda = clf.Lambda
            Gamma = clf.Gamma
            xi = clf.xi

        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        n_iters.append(clf.n_iter_)

    if return_time:
        total_time = time.time() - total_start
        avg_time_per_iter = total_time / sum(n_iters) if sum(n_iters) > 0 else float("inf")


    if verbose:
        print("\nPLQ ERM Path Solution Results")
        print("=" * 90)
        print(f"{'C Value':<15}{'Iterations':<15}{'Time (s)':<20}{'Loss':<20}{'L2 Norm':<20}")
        print("-" * 90)

        for C, iters, t, loss_val, l2 in zip(Cs, n_iters, times, loss_values, L2_norms):
            if return_time:
                print(f"{C:<15.4g}{iters:<15}{t:<20.6f}{loss_val:<20.6f}{l2:<20.6f}")
            else:
                print(f"{C:<15.4g}{iters:<15}{loss_val:<20.6f}{l2:<20.6f}")

        print("=" * 90)
        print(f"{'Total Time':<12}{total_time:.6f} sec")
        print(f"{'Avg Time/Iter':<12}{avg_time_per_iter:.6f} sec")
        print("=" * 90)

    if plot_path:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for i in range(n_features):
            plt.plot(Cs, coefs[i, :], label=f'Feature {i+1}')
        plt.xscale('log')
        plt.xlabel('C')
        plt.ylabel('Coefficient Value')
        plt.title('Regularization Path')
        plt.legend()
        plt.show()

    if return_time:
        return Cs, times, n_iters, loss_values, L2_norms, coefs
    else:
        return Cs, n_iters, loss_values, L2_norms, coefs
