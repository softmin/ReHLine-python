import numpy as np
import time
import matplotlib.pyplot as plt
from rehline import plqERM_Ridge
from rehline import _make_loss_rehline_param
from ._loss import ReHLoss


def plqERM_Ridge_path_sol(
    X,
    y,
    *,
    loss,
    constraint=[ ],
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
        Training input samples.

    y : ndarray of shape (n_samples,)
        Target values corresponding to each input sample.

    loss : dict
        Dictionary describing the PLQ loss function parameters. Used to construct the loss object internally.

    constraint : list of dict, optional (default=[])
        List of constraints applied to the optimization problem. Each constraint should be represented
        as a dictionary compatible with the solver.

    eps : float, default=1e-3
        Defines the range of regularization values when `Cs` is not provided. Specifically, the smallest
        regularization value will be approximately `eps` times the largest.

    n_Cs : int, default=100
        Number of regularization values to evaluate if `Cs` is not provided.

    Cs : array-like of shape (n_Cs,), optional
        Explicit values of regularization strength `C` to use. If `None`, the values are generated
        logarithmically between 1e-2 and 1e3.

    max_iter : int, default=5000
        Maximum number of iterations allowed for the optimization solver at each `C`.

    tol : float, default=1e-4
        Tolerance for solver convergence.

    verbose : int, default=0
        Controls verbosity level of output. Set to higher values (e.g., 1 or 2) for detailed progress logs.

    shrink : float, default=1
        Shrinkage factor for the solver, potentially influencing convergence behavior.

    warm_start : bool, default=False
        If True, reuse the previous solution to warm-start the next solver step, speeding up convergence.

    return_time : bool, default=True
        If True, return timing information for each value of `C`.

    plot_path : bool, default=False
        If True, generate a plot of the coefficient paths as a function of `C`.

    Returns
    -------
    Cs : ndarray of shape (n_Cs,)
        Array of regularization parameters used in the path.

    times : list of float
        Time in seconds taken to fit the model at each `C`. Returned only if `return_time=True`.

    n_iters : list of int
        Number of iterations used by the solver at each regularization value.

    loss_values : list of float
        Final loss values (including regularization term) at each `C`.

    L2_norms : list of float
        L2 norm of the coefficients (excluding bias) at each `C`.

    coefs : ndarray of shape (n_features, n_Cs)
        Learned model coefficients at each regularization strength.

    Example
    -------

    >>> # generate data
    >>> np.random.seed(42)
    >>> n, d, C = 1000, 5, 0.5
    >>> X = np.random.randn(n, d)
    >>> beta0 = np.random.randn(d)
    >>> y = np.sign(X.dot(beta0) + np.random.randn(n))
    >>> # define loss function
    >>> loss = {'name': 'svm'}
    >>> Cs = [2000, 3000, 4000]
    >>> constrain = [{'name': 'none'}]


    >>> # calculate
    >>> Cs, times, n_iters, losses, norms, coefs = plqERM_path_sol(
    ...     X, y, loss=loss, Cs=Cs, max_iter=100000,tol=1e-4,verbose=1,
    ...     warm_start=False, constrain=constrain, return_time=True, plot_path=True
    ... )

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


    if return_time:
        total_start = time.time()
    
    U, V, Tau, S, T = _make_loss_rehline_param(loss, X, y)
    loss_obj = ReHLoss(U, V, S, T, Tau)

    for i, C in enumerate(Cs):
        if return_time:
            start_time = time.time()

        clf = plqERM_Ridge(
            loss=loss, constraint=constraint, C=C,
            max_iter=max_iter, tol=tol, shrink=shrink, verbose=verbose,
            warm_start=warm_start
        )

        if warm_start and (i>0):
            clf.Lambda = Lambda
            clf.Gamma = Gamma
            clf.xi = xi

        clf.fit(X, y)
        coefs[:, i] = clf.coef_

        # Compute loss function parameters for ReHLoss
        l2_norm = 0.5 * np.linalg.norm(clf.coef_) ** 2
        score = clf.decision_function(X)
        total_loss = loss_obj(score) + l2_norm
        loss_values.append(round(total_loss, 4))
        L2_norms.append(round(np.linalg.norm(clf.coef_), 4))

        if warm_start:
            Lambda = clf.Lambda
            Gamma = clf.Gamma
            xi = clf.xi

        if return_time:
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

