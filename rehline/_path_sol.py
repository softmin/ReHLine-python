import time

import numpy as np

from ._base import _make_loss_rehline_param
from ._class import plqERM_Ridge
from ._class import CQR_Ridge
from ._loss import ReHLoss


def plqERM_Ridge_path_sol(
    X,
    y,
    *,
    loss,
    constraint=[],
    eps=1e-3,
    n_Cs=100,
    Cs=None,
    max_iter=5000,
    tol=1e-4,
    verbose=0,
    shrink=1,
    warm_start=False,
    return_time=True,
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
        Defines the length of the regularization path when `Cs` is not provided.
        The values of `C` will range from `10^log10(eps)` to `10^-log10(eps)`.

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
        When verbose = 1, only print path results table;
        when verbose = 2, print path results table and path solution plot.

    shrink : float, default=1
        Shrinkage factor for the solver, potentially influencing convergence behavior.

    warm_start : bool, default=False
        If True, reuse the previous solution to warm-start the next solver step, speeding up convergence.

    return_time : bool, default=True
        If True, return timing information for each value of `C`.

    Returns
    -------
    Cs : ndarray of shape (n_Cs,)
        Array of regularization parameters used in the path.

    times : list of float
        Time in seconds taken to fit the model at each `C`. Returned only if `return_time=True`.

    n_iters : list of int
        Number of iterations used by the solver at each regularization value.

    obj_values : list of float
        Final objective values (including loss and regularization terms) at each `C`.

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
    >>> Cs = np.logspace(-1,3,15)
    >>> constraint = [{'name': 'nonnegative'}]


    >>> # calculate
    >>> Cs, times, n_iters, losses, norms, coefs = plqERM_Ridge_path_sol(
    ...     X, y, loss=loss, Cs=Cs, max_iter=100000,tol=1e-4,verbose=2,
    ...     warm_start=False, constraint=constraint, return_time=True
    ... )

    """

    n_samples, n_features = X.shape

    if Cs is None:
        log_eps = np.log10(eps)
        Cs = np.logspace(log_eps, -log_eps, n_Cs)

    # Sort Cs to ensure computation starts from the smallest value
    Cs = np.sort(Cs)
    n_Cs = len(Cs)
    coefs = np.zeros((n_features, n_Cs))
    n_iters = []
    times = []
    obj_values = []
    L2_norms = []


    if return_time:
        total_start = time.time()
    
    U, V, Tau, S, T = _make_loss_rehline_param(loss, X, y)
    loss_obj = ReHLoss(U, V, S, T, Tau)

    # Lambda_ws = np.empty(shape=(0, 0))
    # Gamma_ws = np.empty(shape=(0, 0))
    # xi_ws = np.empty(shape=(0, 0))

    clf = plqERM_Ridge(
        loss=loss, constraint=constraint, C=Cs[0],
        max_iter=max_iter, tol=tol, shrink=shrink, 
        verbose=1*(verbose>=2), # ben: if verbose is 1, then the fit function will not show the progress
        warm_start=warm_start
    )

    for i, C in enumerate(Cs):
        if return_time:
            start_time = time.time()

        clf.C = C

        # clf = plqERM_Ridge(
        #     loss=loss, constraint=constraint, C=C,
        #     max_iter=max_iter, tol=tol, shrink=shrink, verbose=verbose,
        #     warm_start=warm_start
        # )

        # if (warm_start and (i>0)):
        #     clf.Lambda = Lambda_ws
        #     clf.Gamma = Gamma_ws
        #     clf.xi = xi_ws

        clf.fit(X, y)
        coefs[:, i] = clf.coef_

        # Compute loss function parameters for ReHLoss
        l2_norm = np.linalg.norm(clf.coef_) ** 2
        score = clf.decision_function(X)
        total_obj = loss_obj(score) + 0.5*l2_norm
        obj_values.append(round(total_obj, 4))
        L2_norms.append(round(np.linalg.norm(clf.coef_), 4))

        # if warm_start:
        #     Lambda_ws = clf.Lambda
        #     Gamma_ws = clf.Gamma
        #     xi_ws = clf.xi

        if return_time:
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
        
        n_iters.append(clf.n_iter_)

    if return_time:
        total_time = time.time() - total_start
        avg_time_per_iter = total_time / sum(n_iters) if sum(n_iters) > 0 else float("inf")


    if verbose >= 1:
        print("\nPLQ ERM Path Solution Results")
        print("=" * 90)
        print(f"{'C Value':<15}{'Iterations':<15}{'Time (s)':<20}{'Loss':<20}{'L2 Norm':<20}")
        print("-" * 90)

        for C, iters, t, loss_val, l2 in zip(Cs, n_iters, times, obj_values, L2_norms):
            if return_time:
                print(f"{C:<15.4g}{iters:<15}{t:<20.6f}{loss_val:<20.6f}{l2:<20.6f}")
            else:
                print(f"{C:<15.4g}{iters:<15}{loss_val:<20.6f}{l2:<20.6f}")

        print("=" * 90)
        print(f"{'Total Time':<12}{total_time:.6f} sec")
        print(f"{'Avg Time/Iter':<12}{avg_time_per_iter:.6f} sec")
        print("=" * 90)


    if return_time:
        return Cs, times, n_iters, obj_values, L2_norms, coefs
    else:
        return Cs, n_iters, obj_values, L2_norms, coefs



def CQR_Ridge_path_sol(
    X,
    y,
    *,
    quantiles,
    eps=1e-5,
    n_Cs=50,
    Cs=None,
    max_iter=5000,
    tol=1e-4,
    verbose=0,
    shrink=1,
    warm_start=False,
    return_time=True,
):
    """
    Compute the regularization path for Composite Quantile Regression (CQR) with ridge penalty.

    This function fits a series of CQR models using different values of the regularization parameter `C`.
    It reuses a single estimator and modifies `C` in-place before refitting.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.

    y : ndarray of shape (n_samples,)
        Response vector.

    quantiles : list of float
        Quantile levels (e.g. [0.1, 0.5, 0.9]).

    eps : float, default=1e-5
        Log-scaled lower bound for generated `C` values (used if `Cs` is None).

    n_Cs : int, default=50
        Number of `C` values to generate.

    Cs : array-like or None, default=None
        Explicit values of regularization strength. If None, use `eps` and `n_Cs` to generate them.

    max_iter : int, default=5000
        Maximum number of solver iterations.

    tol : float, default=1e-4
        Solver convergence tolerance.

    verbose : int, default=0
        Verbosity level.

    shrink : float, default=1
        Shrinkage parameter passed to solver.

    warm_start : bool, default=False
        Use previous dual solution to initialize the next fit.

    return_time : bool, default=True
        Whether to return a list of fit durations.

    Returns
    -------
    Cs : ndarray
        List of regularization strengths.

    models : list
        List of fitted model objects.

    coefs : ndarray of shape (n_Cs, n_quantiles, n_features)
        Coefficient matrices per quantile and `C`.

    intercepts : ndarray of shape (n_Cs, n_quantiles)
        Intercepts per quantile and `C`.

    fit_times : list of float, optional
        Elapsed fit times (if `return_time=True`).

        
    Example
    -------
    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.preprocessing import StandardScaler
    >>> import numpy as np
    >>> from rehline import CQR_Ridge_path_sol

    >>> # Generate the data
    >>> X, y = make_friedman1(n_samples=500, n_features=6, noise=1.0, random_state=42)
    >>> X = StandardScaler().fit_transform(X)
    >>> y = y / y.std()

    >>> # Set quantiles and Cs
    >>> quantiles = [0.1, 0.5, 0.9]
    >>> Cs = np.logspace(-5, 0, 30)

    >>> # Fit CQR path
    >>> Cs, models, coefs, intercepts, fit_times = CQR_Ridge_path_sol(
    ...     X, y,
    ...     quantiles=quantiles,
    ...     Cs=Cs,
    ...     max_iter=100000,
    ...     tol=1e-4,
    ...     verbose=1,
    ...     warm_start=True,
    ...     return_time=True
    ... )
    
    """

    if Cs is None:
        log_Cs = np.linspace(np.log10(eps), np.log10(10), n_Cs)
        Cs = np.power(10.0, log_Cs)
    else:
        Cs = np.array(Cs)

    models = []
    fit_times = []
    coefs = []
    intercepts = []

    clf = CQR_Ridge(
        quantiles=quantiles,
        C=Cs[0],
        max_iter=max_iter,
        tol=tol,
        shrink=shrink,
        verbose=verbose,
        warm_start=warm_start,
    )

    for i, C in enumerate(Cs):
        clf.C = C  

        if return_time:
            start = time.time()

        clf.fit(X, y)

        d = X.shape[1]
        n_qt = len(quantiles)

        coef_matrix = np.tile(clf.coef_, (n_qt, 1))
        intercept_vector = clf.intercept_

        models.append(clf)
        coefs.append(coef_matrix)
        intercepts.append(intercept_vector)

        if return_time:
            elapsed = time.time() - start
            fit_times.append(elapsed)
            if verbose >= 1:
                print(f"[OK] C={C:.3e}, time={elapsed:.3f}s")

    coefs = np.array(coefs)       # (n_Cs, n_quantiles, n_features)
    intercepts = np.array(intercepts)  # (n_Cs, n_quantiles)

    if return_time:
        return Cs, models, coefs, intercepts, fit_times
    else:
        return Cs, models, coefs, intercepts
