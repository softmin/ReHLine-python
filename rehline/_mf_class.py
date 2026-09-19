"""Matrix Factorization Optimization with Various Loss Functions Based on ReHLine"""

import warnings
from copy import deepcopy
from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_is_fitted

from ._base import (
    ReHLine_solver,
    _BaseReHLine,
    _cast_sample_bias,
    _cast_sample_weight,
    _fit_transaction,
    _make_constraint_rehline_param,
    _make_loss_rehline_param,
)
from ._loss import ReHLoss
from ._validation import model_options, numeric_array, positive_real, sample_weights


class plqMF_Ridge(_BaseReHLine, BaseEstimator):
    r"""Matrix Factorization (MF) with a piecewise linear-quadratic objective and ridge penalty.

    .. math::
        \min_{\substack{
            \mathbf{P} \in \mathbb{R}^{n \times k}\
            \pmb{\alpha} \in \mathbb{R}^n \\
            \mathbf{Q} \in \mathbb{R}^{m \times k}\
            \pmb{\beta} \in \mathbb{R}^m
        }}
        \left[
            \sum_{(u,i)\in \Omega} C \cdot \text{PLQ}(r_{ui}, \ \mathbf{p}_u^T \mathbf{q}_i + \alpha_u + \beta_i)
        \right]
        +
        \left[
            \frac{\rho}{n}\sum_{u=1}^n(\|\mathbf{p}_u\|_2^2 + \alpha_u^2)
            + \frac{1-\rho}{m}\sum_{i=1}^m(\|\mathbf{q}_i\|_2^2 + \beta_i^2)
        \right]

    .. math::
        \ \text{ s.t. } \
        \mathbf{A}_{\text{user}} \begin{pmatrix} \alpha_u \\ \mathbf{p}_u \end{pmatrix} + \mathbf{b}_{\text{user}} \geq \mathbf{0},\ u = 1,\dots,n
        \quad \text{and} \quad
        \mathbf{A}_{\text{item}} \begin{pmatrix} \beta_i \\ \mathbf{q}_i \end{pmatrix} + \mathbf{b}_{\text{item}} \geq \mathbf{0},\ i = 1,\dots,m

    The function supports various loss functions, including:
        - 'hinge', 'svm' or 'SVM'
        - 'MAE' or 'mae' or 'mean absolute error'
        - 'squared hinge' or 'squared svm' or 'squared SVM'
        - 'MSE' or 'mse' or 'mean squared error'

    The following constraint types are supported:
        * 'nonnegative' or '>=0': A non-negativity constraint.
        * 'fair' or 'fairness': A fairness constraint.
        * 'custom': A custom constraint, where the user must provide the constraint matrix 'A' and vector 'b'.

    Parameters
    ----------
    n_users : int
        Number of unique users in the dataset (or number of rows in target sparse matrix).

    n_items : int
        Number of unique items in the dataset (or number of columns in target sparse matrix).

    loss : dict
        A dictionary specifying the loss function parameters.

    constraint_user : list of dict
        A list of dictionaries, where each dictionary represents a constraint to user side parameters.
        Each dictionary must contain a 'name' key, which specifies the type of constraint.

    constraint_item : list of dict
        A list of dictionaries, where each dictionary represents a constraint to item side parameters.
        Each dictionary must contain a 'name' key, which specifies the type of constraint.

    biased : bool, default=True
            Whether to include user and item bias terms in the model.

    rank : int, default=10
        Dimensionality of the latent factor vectors (number of factors).

    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to `C`. Must be strictly positive.
        `C` will be absorbed by the ReHLine parameters when `_cast_sample_weight()` is conducted.

    rho : float, default=0.5
        Regularization strength ratio between user and item factors. Must be within the range of (0,1).

    init_mean : float, default=0.0
        Mean of the Gaussian distribution for initializing latent factors.

    init_sd : float, default=0.1
        Standard deviation of the Gaussian distribution for initializing latent factors.

    random_state : int, RandomState or Generator, default=None
        Random seed for reproducible initialization of latent factors.

    max_iter : int, default=10000
        The maximum number of iterations to be run for the ReHLine solver.

    tol : float, default=1e-4
        Convergence tolerance for each ReHLine block solve and the final
        row-normalized factor constraint violation.

    shrink : float, default=1
        The shrinkage of dual variables for the ReHLine solver.

    trace_freq : int, default=100
        The frequency at which to print the optimization trace for the ReHLine solver.

    max_iter_CD : int, default=10
        Maximum number of iterations for coordinate descent steps.

    tol_CD : float, default=1e-4
        The tolerance for the stopping criterion for coordinate descent steps.

    verbose : int, default=0
        Verbosity level.
          0: No output
          1: CD iteration progress information
          2: ReHLine solver optimization information
          3: All information of CD and ReHLine

    Attributes
    ----------
    n_users : int
        Number of unique users in the dataset (or number of rows in target sparse matrix).

    n_items : int
        Number of unique items in the dataset (or number of columns in target sparse matrix).

    n_ratings : int
        Number of ratings in the training data. Available after fitting.

    P : ndarray of shape (n_users, rank)
        User latent factor matrix. Learned during fitting.

    Q : ndarray of shape (n_items, rank)
        Item latent factor matrix. Learned during fitting.

    bu : ndarray of shape (n_users,) or None
        User bias terms. Learned during fitting. Only available if `biased=True`.

    bi : ndarray of shape (n_items,) or None
        Item bias terms. Learned during fitting. Only available if `biased=True`.

    Iu : list of ndarray
        List where each element contains indices of items rated by the corresponding user.
        Available after fitting.

    Ui : list of ndarray
        List where each element contains indices of users who rated the corresponding item.
        Available after fitting.

    history : ndarray of shape (max_iter_CD + 1, 2)
        Optimization history containing loss and objective values at each coordinate descent iteration.
        First column: weighted cumulative loss. Second column: weighted objective including the penalty.
        Unused rows after early stopping are NaN.

    objective_ : float
        Final training objective, including sample weights and both penalties.

    n_iter_ : int
        Number of completed outer coordinate-descent sweeps.

    converged_ : bool
        Whether the weighted objective stopping criterion, inner convergence and
        final feasibility checks passed. This does not certify a global MF optimum.
        False when the outer iteration budget is exhausted without convergence.
        An exhausted outer budget emits ConvergenceWarning mentioning max_iter_CD.

    inner_converged_ : bool
        Whether all block solves in the last sweep converged.

    constraint_violation_ : float
        Maximum violation of the final user/item constraints in original units.

    scaled_constraint_violation_ : float
        Maximum violation after each constraint row (A, b) is divided by
        max(abs(A)). A zero row uses scale 1. This diagnostic is compared
        with tol for outer convergence and constraint warnings; it is
        invariant to positive row rescaling within floating-point accuracy.

    sample_weight : ndarray of shape (n_ratings,)
        Sample weights used during fitting. Available after fitting.

    Methods
    -------
    fit(X, y, sample_weight=None)
        Fit the model based on the given training data.

    decision_function(X)
        The decision function evaluated on the given dataset.

    obj(X, y)
        Compute the values of loss term and objective function.

    Notes
    -----
    The `plqMF_Ridge` class is a subclass of `_BaseReHLine` and `BaseEstimator`, which suggests that it is part of a larger framework for implementing ReHLine algorithms.

    """

    def __init__(
        self,
        n_users,
        n_items,
        loss,
        biased=True,
        constraint_user=None,
        constraint_item=None,
        rank=10,
        C=1.0,
        rho=0.5,
        init_mean=0.0,
        init_sd=0.1,
        random_state=None,
        max_iter=10000,
        tol=1e-4,
        shrink=1,
        trace_freq=100,
        max_iter_CD=10,
        tol_CD=1e-4,
        verbose=0,
    ):
        # parameter initialization
        ## -----------------------------basic parameters-----------------------------
        self.n_users = n_users
        self.n_items = n_items
        self.loss = loss
        self.constraint_user = constraint_user if constraint_user is not None else []
        self.constraint_item = constraint_item if constraint_item is not None else []
        self.biased = biased
        ## -----------------------------hyper parameters-----------------------------
        self.rank = rank
        self.C = C
        self.rho = rho
        ## -------------------------initialization parameters------------------------
        self.init_mean = init_mean
        self.init_sd = init_sd
        self.random_state = random_state
        ## ----------------------------fitting parameters----------------------------
        self.max_iter_CD = max_iter_CD
        self.tol_CD = tol_CD
        self.verbose = verbose
        self.tol = tol
        self.max_iter = max_iter
        self.shrink = shrink
        self.trace_freq = trace_freq

    def __sklearn_is_fitted__(self):
        return hasattr(self, "P") and hasattr(self, "Q")

    def _fitted_param(self, name):
        return self._fit_params_[name] if hasattr(self, "_fit_params_") else getattr(self, name)

    def _validate_pairs(self, X, *, fitted=False):
        X = numeric_array(X, "X", ndim=2)
        if X.shape[1] != 2:
            raise ValueError("X must have shape (n_ratings, 2)")
        if np.any(X != np.floor(X)):
            raise ValueError("User and item IDs must be integers")
        n_users = self._fitted_param("n_users") if fitted else self.n_users
        n_items = self._fitted_param("n_items") if fitted else self.n_items
        if np.any(X[:, 0] < 0) or np.any(X[:, 0] >= n_users):
            raise ValueError("User IDs must be in [0, n_users)")
        if np.any(X[:, 1] < 0) or np.any(X[:, 1] >= n_items):
            raise ValueError("Item IDs must be in [0, n_items)")
        return X.astype(np.intp)

    def _block_constraints(self, constraint, design):
        # Empirical fairness statistics have no definition for an empty block.
        if len(design) == 0 and any(
            isinstance(c, dict) and c.get("name") in ("fair", "fairness")
            for c in ([] if constraint is None else constraint)
        ):
            raise ValueError("Fairness constraints require observations for every constrained user/item")
        return _make_constraint_rehline_param(constraint, design)

    def _solve_block(self, design, target, weight, bias, constraint, C, cache):
        """Solve one convex factor update, including blocks with no effective loss."""
        A, b = self._block_constraints(constraint, design)
        d = design.shape[1]
        no_loss = not np.any(weight > 0)
        if no_loss:
            if np.all(b >= 0):
                return np.zeros(d), True
            key = (A.shape, A.tobytes(), b.tobytes())
            if key in cache:
                return cache[key].copy(), True
            # No terms depend on this placeholder design row.
            design = np.zeros((1, d))
            U = V = S = T = Tau = np.empty((0, 1))
        else:
            U, V, Tau, S, T = _make_loss_rehline_param(self.loss, design, target)
            U, V, Tau, S, T = _cast_sample_bias(U, V, Tau, S, T, sample_bias=bias)
            U, V, Tau, S, T = _cast_sample_weight(U, V, Tau, S, T, C=C, sample_weight=weight)
        result = ReHLine_solver(
            X=design,
            U=U,
            V=V,
            S=S,
            T=T,
            Tau=Tau,
            A=A,
            b=b,
            max_iter=self.max_iter,
            tol=self.tol,
            shrink=self.shrink,
            verbose=int(self.verbose in (2, 3)),
            trace_freq=self.trace_freq,
        )
        if not result.converged:
            warnings.warn(
                "ReHLine failed to converge, increase the number of iterations: `max_iter`.",
                ConvergenceWarning,
                stacklevel=3,
            )
        elif no_loss:
            cache[key] = result.beta.copy()
        return result.beta.copy(), result.converged

    def _factor_constraint_violations(self, X):
        """Return original-unit and row-normalized violations of all factors."""
        violation = scaled_violation = 0.0
        for groups, column, opposite, factors, biases, constraints in (
            (self.Iu, 1, self.Q, self.P, self.bu, self.constraint_user),
            (self.Ui, 0, self.P, self.Q, self.bi, self.constraint_item),
        ):
            if constraints is None or len(constraints) == 0:
                continue
            for index, rows in enumerate(groups):
                design = opposite[X[rows, column]]
                z = factors[index]
                if self.biased:
                    design = np.column_stack((np.ones(len(rows)), design))
                    z = np.r_[biases[index], z]
                A, b = self._block_constraints(constraints, design)
                if len(b):
                    row_scale = np.max(abs(A), axis=1)
                    row_scale[row_scale == 0] = 1
                    # Normalize before the product: a huge original row can
                    # otherwise hide a small but representable normalized slack.
                    try:
                        with np.errstate(over="raise", invalid="raise", under="ignore"):
                            raw_slack = A @ z + b
                            scaled_slack = (A / row_scale[:, None]) @ z + b / row_scale
                    except FloatingPointError as exc:
                        raise OverflowError("MF constraint diagnostics exceed the floating-point range") from exc
                    if not np.isfinite(raw_slack).all() or not np.isfinite(scaled_slack).all():
                        raise OverflowError("MF constraint diagnostics exceed the floating-point range")
                    violation = max(violation, float(-raw_slack.min()))
                    scaled_violation = max(scaled_violation, float(-scaled_slack.min()))
        return violation, scaled_violation

    def _factor_constraint_violation(self, X):
        """Original-unit diagnostic retained for existing internal callers."""
        return self._factor_constraint_violations(X)[0]

    @_fit_transaction
    def fit(self, X, y, sample_weight=None):
        """Fit the model based on the given training data.

        Parameters
        ----------
        X : array-like of shape (n_ratings, 2)
            Input data where first column contains user ID and
            second column contains item ID.

        y : array-like of shape (n_ratings,)
            Target rating values.

        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        self : object
            An instance of the estimator.

        """
        model_options(self)
        positive_real(self.rho, "rho", allow_zero=True)
        if not 0 < self.rho < 1:
            raise ValueError("rho must be between 0 and 1")
        positive_real(self.tol_CD, "tol_CD")
        positive_real(self.init_sd, "init_sd", allow_zero=True)
        numeric_array(self.init_mean, "init_mean", ndim=0)
        for name in ("n_users", "n_items", "rank", "max_iter_CD"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
                or not 1 <= value <= np.iinfo(np.int32).max
            ):
                raise ValueError(f"{name} must be a positive integer fitting in int32")
        if not isinstance(self.biased, (bool, np.bool_)):
            raise ValueError("biased must be boolean")
        X = self._validate_pairs(X)
        y = numeric_array(y, "y", ndim=1)
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        if len(y) == 0:
            raise ValueError("At least one rating is required")
        self._fit_params_ = {
            name: deepcopy(getattr(self, name)) for name in ("n_users", "n_items", "biased", "C", "rho", "loss")
        }
        self.n_ratings = len(y)
        self.history = np.full((self.max_iter_CD + 1, 2), np.nan)
        self.sample_weight = sample_weights(sample_weight, len(y)).copy()
        rng = (
            self.random_state
            if isinstance(self.random_state, np.random.RandomState)
            else np.random.default_rng(self.random_state)
        )

        def groups(column, count):
            order = np.argsort(X[:, column], kind="stable")
            ids, counts = np.unique(X[order, column], return_counts=True)
            result = [np.empty(0, dtype=np.intp) for _ in range(count)]
            for index, rows in zip(ids, np.split(order, np.cumsum(counts)[:-1])):
                result[index] = rows
            return result

        self.Iu = groups(0, self.n_users)
        self.Ui = groups(1, self.n_items)
        C_user = self.C * self.n_users / self.rho / 2
        C_item = self.C * self.n_items / (1 - self.rho) / 2
        self.P = rng.normal(self.init_mean, self.init_sd, (self.n_users, self.rank))
        self.Q = rng.normal(self.init_mean, self.init_sd, (self.n_items, self.rank))
        self.bu = np.zeros(self.n_users) if self.biased else None
        self.bi = np.zeros(self.n_items) if self.biased else None
        if self.verbose in (1, 3):
            print(f"{'Iteration':<12} {'Average Loss(' + self.loss['name'] + ')':<20} Objective Function")
        self.history[0] = self.obj(X, y, sample_weight=self.sample_weight)
        self.converged_ = False
        self.inner_converged_ = False
        self.n_iter_ = 0
        cache = {}
        for iteration in range(self.max_iter_CD):
            self.inner_converged_ = True
            for groups_, column, opposite, opposite_bias, factors, biases, constraint, C in (
                (self.Iu, 1, self.Q, self.bi, self.P, self.bu, self.constraint_user, C_user),
                (self.Ui, 0, self.P, self.bu, self.Q, self.bi, self.constraint_item, C_item),
            ):
                for index, rows in enumerate(groups_):
                    other_ids = X[rows, column]
                    design = opposite[other_ids]
                    bias = opposite_bias[other_ids] if self.biased else None
                    if self.biased:
                        design = np.column_stack((np.ones(len(rows)), design))
                    z, converged = self._solve_block(
                        design,
                        y[rows],
                        self.sample_weight[rows],
                        bias,
                        constraint,
                        C,
                        cache,
                    )
                    self.inner_converged_ &= converged
                    if self.biased:
                        biases[index], factors[index] = z[0], z[1:]
                    else:
                        factors[index] = z
            self.n_iter_ = iteration + 1
            self.history[self.n_iter_] = self.obj(X, y, sample_weight=self.sample_weight)
            previous = self.history[iteration, 1]
            self.objective_ = self.history[self.n_iter_, 1]
            improvement = previous - self.objective_
            roundoff = 64 * np.finfo(float).eps * max(1, abs(previous), abs(self.objective_))
            if self.verbose in (1, 3):
                print(
                    f"{self.n_iter_:<12} {self.history[self.n_iter_, 0] / self.n_ratings:<20.6f} {self.objective_:.6f}"
                )
            if self.inner_converged_ and -roundoff <= improvement < self.tol_CD:
                if self._factor_constraint_violations(X)[1] <= self.tol:
                    self.converged_ = True
                    break
        self.constraint_violation_, self.scaled_constraint_violation_ = self._factor_constraint_violations(X)
        if self.scaled_constraint_violation_ > self.tol and self.inner_converged_:
            warnings.warn(
                "MF factors do not satisfy all final constraints within scaled tol.", ConvergenceWarning, stacklevel=2
            )
        elif not self.converged_ and self.inner_converged_:
            warnings.warn(
                "MF outer iterations failed to converge; increase `max_iter_CD`.",
                ConvergenceWarning,
                stacklevel=2,
            )
        return self

    def decision_function(self, X):
        """The decision function evaluated on the given dataset

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            Training data where first column contains user ID and
            second column contains item ID.

        Returns
        -------
        prediction : ndarray of shape (n_samples,)
            Predicted ratings for the input pairs.
        """
        check_is_fitted(self)
        X = self._validate_pairs(X, fitted=True)
        users = X[:, 0]
        items = X[:, 1]
        dot_products = np.einsum("ij,ij->i", self.P[users], self.Q[items])

        if self._fitted_param("biased"):
            user_biases = self.bu[users]
            item_biases = self.bi[items]
            return user_biases + item_biases + dot_products
        else:
            return dot_products

    def obj(self, X, y, sample_weight=None):
        """
        Compute the values of loss term and objective function.

        Parameters
        ----------
        X : array-like of shape (n_ratings, 2)
            User-item rating pairs.

        y : array-like of shape (n_ratings,)
            Actual rating values.

        sample_weight : array-like or float, default=None
            Evaluation weights; None means equal weights. Training weights are
            not implicitly reused for evaluation on another dataset.

        Returns
        -------
        loss_term : float
            The data fitting term (sum of loss values).

        objective_value : float
            The total objective value including regularization.

        """

        check_is_fitted(self)
        X = self._validate_pairs(X, fitted=True)
        y = numeric_array(y, "y", ndim=1)
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        weight = sample_weights(sample_weight, len(y), allow_all_zero=True)

        rho = self._fitted_param("rho")
        if self._fitted_param("biased"):
            user_penalty = (np.sum(self.P**2) + np.sum(self.bu**2)) * rho / self._fitted_param("n_users")
            item_penalty = (np.sum(self.Q**2) + np.sum(self.bi**2)) * (1 - rho) / self._fitted_param("n_items")
            penalty = user_penalty + item_penalty
        else:
            user_penalty = np.sum(self.P**2) * rho / self._fitted_param("n_users")
            item_penalty = np.sum(self.Q**2) * (1 - rho) / self._fitted_param("n_items")
            penalty = user_penalty + item_penalty

        X_dummy = np.ones((len(y), 1))  # not used in loss computation, only shape matters for loss param construction
        U, V, Tau, S, T = _make_loss_rehline_param(loss=self._fitted_param("loss"), X=X_dummy, y=y)
        loss = ReHLoss(U, V, S, T, Tau)
        y_pred = self.decision_function(X)
        active = weight > 0
        loss_term = float(weight[active] @ loss.values(y_pred)[active])

        return loss_term, self._fitted_param("C") * loss_term + penalty
