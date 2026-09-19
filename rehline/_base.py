"""Base functions for ReHLine."""

# Authors: Ben Dai <bendai@cuhk.edu.hk>
#          Yixuan Qiu <yixuanq@gmail.com>

# License: MIT License

import warnings
from abc import abstractmethod
from copy import copy
from functools import wraps
from numbers import Integral

import numpy as np
from scipy.special import huber
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted

from ._internal import rehline_cqr_internal, rehline_internal, rehline_result
from ._validation import numeric_array, positive_real, sample_weights, solver_options


def _fit_transaction(fit):
    """Publish fitted state only after success; parameter changes are retained.

    Decorated fits allocate new result arrays. Reused duals are copied/clipped
    at the solver boundary, so staging does not mutate the previous fitted state.
    """

    @wraps(fit)
    def staged_fit(self, *args, **kwargs):
        staged = copy(self)
        fit(staged, *args, **kwargs)
        self.__dict__ = staged.__dict__
        return self

    return staged_fit


class _BaseReHLine(BaseEstimator):
    r"""Base Class of ReHLine Formulation.

    .. math::

        \min_{\mathbf{\beta} \in \mathbb{R}^d} \sum_{i=1}^n \sum_{l=1}^L \text{ReLU}( u_{li} \mathbf{x}_i^\intercal \mathbf{\beta} + v_{li}) + \sum_{i=1}^n \sum_{h=1}^H {\text{ReHU}}_{\tau_{hi}}( s_{hi} \mathbf{x}_i^\intercal \mathbf{\beta} + t_{hi}) + \frac{1}{2} \| \mathbf{\beta} \|_2^2, \\ \text{ s.t. }
        \mathbf{A} \mathbf{\beta} + \mathbf{b} \geq \mathbf{0},

    where :math:`\mathbf{U} = (u_{li}),\mathbf{V} = (v_{li}) \in \mathbb{R}^{L \times n}`
    and :math:`\mathbf{S} = (s_{hi}),\mathbf{T} = (t_{hi}),\mathbf{\tau} = (\tau_{hi}) \in \mathbb{R}^{H \times n}`
    are the ReLU-ReHU loss parameters, and :math:`(\mathbf{A},\mathbf{b})` are the constraint parameters.

    Parameters
    ----------

    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.

    U, V: array of shape (L, n_samples), default=np.empty(shape=(0, 0))
        The parameters pertaining to the ReLU part in the loss function.

    Tau, S, T: array of shape (H, n_samples), default=np.empty(shape=(0, 0))
        The parameters pertaining to the ReHU part in the loss function.

    A: array of shape (K, n_features), default=np.empty(shape=(0, 0))
        The coefficient matrix in the linear constraint.

    b: array of shape (K, ), default=np.empty(shape=0)
        The intercept vector in the linear constraint.

    """

    def __init__(
        self,
        *,
        C=1.0,
        U=None,
        V=None,
        Tau=None,
        S=None,
        T=None,
        A=None,
        b=None,
    ):
        self.C = C
        self.U = U
        self.V = V
        self.S = S
        self.T = T
        self.Tau = Tau
        self.A = A
        self.b = b
        self._U = U if U is not None else np.empty(shape=(0, 0))
        self._V = V if V is not None else np.empty(shape=(0, 0))
        self._S = S if S is not None else np.empty(shape=(0, 0))
        self._T = T if T is not None else np.empty(shape=(0, 0))
        self._Tau = Tau if Tau is not None else np.empty(shape=(0, 0))
        self._A = A if A is not None else np.empty(shape=(0, 0))
        self._b = b if b is not None else np.empty(shape=(0))
        self.L = self._U.shape[0]
        self.H = self._S.shape[0]
        self.K = self._A.shape[0]

    def __sklearn_is_fitted__(self):
        return getattr(self, "coef_", None) is not None

    def auto_shape(self):
        """
        Automatically generate the shape of the parameters of the ReHLine loss function.
        """
        self.L = self._U.shape[0]
        self.H = self._S.shape[0]
        self.K = self._A.shape[0]

    def _prepare_warm_start(self, n_samples, rho=None):
        """Match cached dual dimensions and retain normalized constraint duals."""
        row_scale = np.max(abs(self._A), axis=1) if self._A.shape[0] else np.empty(0)
        row_scale[row_scale == 0] = 1
        previous_scale = getattr(self, "_xi_row_scale", None)
        shapes = {
            "_Lambda": (self._U.shape[0], n_samples),
            "_Gamma": (self._S.shape[0], n_samples),
            "_xi": (self._A.shape[0],),
        }
        if hasattr(self, "_mu"):
            shapes["_mu"] = (0 if rho is None else len(rho),)
        compatible = all(
            (np.size(getattr(self, name)) == 0 and np.prod(shape) == 0)
            or (np.shape(getattr(self, name)) == shape and np.isfinite(getattr(self, name)).all())
            for name, shape in shapes.items()
        )
        if not self.warm_start or not compatible:
            for name, shape in shapes.items():
                setattr(self, name, np.empty((0,) * len(shape)))
        elif previous_scale is not None and previous_scale.shape == row_scale.shape:
            changed = previous_scale != row_scale
            if np.any(changed):
                xi = self._xi.copy()
                with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                    xi[changed] = (xi[changed] * previous_scale[changed]) / row_scale[changed]
                # An unusable initial multiplier can restart at zero. The
                # solver still rejects unrepresentable final multipliers.
                xi[~np.isfinite(xi)] = 0
                self._xi = xi
        self._xi_row_scale = row_scale

    def cast_sample_weight(self, sample_weight=None):
        """
        Cast the sample weight to the ReHLine parameters.

        Parameters
        ----------
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        U_weight : array-like of shape (L, n_samples)
            Weighted ReLU coefficient matrix.

        V_weight : array-like of shape (L, n_samples)
            Weighted ReLU intercept vector.

        Tau_weight : array-like of shape (H, n_samples)
            Weighted ReHU cutpoint matrix.

        S_weight : array-like of shape (H, n_samples)
            Weighted ReHU coefficient vector.

        T_weight : array-like of shape (H, n_samples)
            Weighted ReHU intercept vector.

        Notes
        -----
        This method casts the sample weight to the ReHLine parameters by multiplying
        the sample weight with the ReLU and ReHU parameters. If sample_weight is None,
        then the sample weight is set to the weight parameter C.
        """

        self.auto_shape()

        return _cast_sample_weight(
            self._U,
            self._V,
            self._Tau,
            self._S,
            self._T,
            C=self.C,
            sample_weight=sample_weight,
        )

    def call_ReLHLoss(self, score):
        """
        Return the value of the ReHLine loss of the `score`.

        Parameters
        ----------
        score : ndarray of shape (n_samples, )
            The input score that will be evaluated through the ReHLine loss.

        Returns
        -------
        float
            ReHLine loss evaluation of the given score.
        """
        n = len(score)
        relu_input = np.zeros((self.L, n))
        rehu_input = np.zeros((self.H, n))
        if self.L > 0:
            relu_input = (self._U.T * score[:, np.newaxis]).T + self._V
        if self.H > 0:
            rehu_input = (self._S.T * score[:, np.newaxis]).T + self._T
        return np.sum(_relu(relu_input), 0) + np.sum(_rehu(rehu_input, self._Tau), 0)

    @abstractmethod
    def fit(self, X, y, sample_weight):
        """Fit model."""

    @abstractmethod
    def decision_function(self, X):
        """The decision function evaluated on the given dataset

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        ndarray of shape (n_samples, )
            Returns the decision function of the samples.
        """
        # Check if fit has been called
        check_is_fitted(self)

        X = check_array(X)


def _relu(x):
    """
    Evaluation of ReLU given a vector.

    Parameters
    ----------

    x: {array-like} of shape (n_samples, )
    Training vector, where `n_samples` is the number of samples


    Returns
    -------
    array of shape (n_samples, )
        An array with ReLU applied, i.e., all negative values are replaced with 0.

    """
    return np.maximum(x, 0)


def _rehu(x, cut=1):
    """
    Evaluation of ReHU given a vector.

    Parameters
    ----------

    x: {array-like} of shape (n_samples, )
        Training vector, where `n_samples` is the number of samples

    cut: {array-like} of shape (n_samples, )
        Cutpoints of ReHU, where `n_samples` is the number of samples

    Returns
    -------
    array of shape (n_samples, )
        The result of the ReHU function.

    """
    cut = cut * np.ones_like(x)

    u = np.maximum(x, 0)
    return huber(cut, u)


def _check_relu(relu_coef, relu_intercept):
    if relu_coef.shape != relu_intercept.shape:
        raise ValueError("`relu_coef` and `relu_intercept` should be the same shape!")


def _check_rehu(rehu_coef, rehu_intercept, rehu_cut):
    if rehu_coef.shape != rehu_intercept.shape:
        raise ValueError("`rehu_coef` and `rehu_intercept` should be the same shape!")
    if len(rehu_coef) > 0 and not (rehu_cut >= 0.0).all():
        raise ValueError("`rehu_cut` must be non-negative!")


def ReHLine_solver(
    X,
    U,
    V,
    Tau=None,
    S=None,
    T=None,
    A=None,
    b=None,
    rho=None,
    Lambda=None,
    Gamma=None,
    xi=None,
    mu=None,
    max_iter=1000,
    tol=1e-4,
    shrink=1,
    verbose=1,
    trace_freq=100,
    *,
    _quantile_count=0,
):
    solver_options(max_iter, tol, shrink, verbose, trace_freq)
    X = check_array(X, dtype=np.float64, order="C")
    n, d = X.shape
    if (
        isinstance(_quantile_count, (bool, np.bool_))
        or not isinstance(_quantile_count, Integral)
        or _quantile_count < 0
        or n * max(1, int(_quantile_count)) > np.iinfo(np.int32).max
        or d + int(_quantile_count) > np.iinfo(np.int32).max
    ):
        raise ValueError("Invalid composite quantile dimensions")
    _quantile_count = int(_quantile_count)
    n *= max(1, _quantile_count)
    d += _quantile_count

    def matrix(value, name, columns, *, allow_inf=False):
        if value is None:
            return np.empty((0, columns))
        value = numeric_array(value, name, ndim=2, allow_inf=allow_inf)
        if value.shape[0] == 0 and value.shape[1] in (0, columns):
            return np.empty((0, columns))
        if value.shape[1] != columns:
            raise ValueError(f"{name} must have {columns} columns")
        return value

    U, V = matrix(U, "U", n), matrix(V, "V", n)
    S, T = matrix(S, "S", n), matrix(T, "T", n)
    Tau = matrix(Tau, "Tau", n, allow_inf=True)
    A = matrix(A, "A", d)
    b = np.empty(0) if b is None else numeric_array(b, "b", ndim=1)
    rho = np.empty(0) if rho is None else numeric_array(rho, "rho", ndim=1)
    if U.shape != V.shape:
        raise ValueError("U and V must have the same shape")
    if S.shape != T.shape or S.shape != Tau.shape or np.any(Tau < 0):
        raise ValueError("S, T and Tau must have the same shape and Tau must be non-negative")
    if b.shape != (A.shape[0],):
        raise ValueError("b must have one entry per row of A")
    if rho.shape not in ((0,), (d,)) or np.any(rho < 0):
        raise ValueError("rho must be empty or a non-negative vector of length n_features")
    if np.any(np.all(A == 0, axis=1) & (b < 0)):
        raise ValueError("A zero constraint row with negative b is infeasible")

    result = rehline_result()
    for name, value, shape, upper in (
        ("Lambda", Lambda, U.shape, 1.0),
        ("Gamma", Gamma, S.shape, Tau),
        ("xi", xi, b.shape, np.inf),
        ("mu", mu, rho.shape, rho),
    ):
        if value is None or np.size(value) == 0:
            continue
        value = numeric_array(value, name, ndim=len(shape))
        if value.shape != shape:
            raise ValueError(f"{name} warm start must have shape {shape}")
        setattr(result, name, np.clip(value, 0, upper))
    native = rehline_cqr_internal if _quantile_count else rehline_internal
    options = (_quantile_count,) if _quantile_count else ()
    native(
        result,
        X,
        A,
        b,
        rho,
        U,
        V,
        S,
        T,
        Tau,
        *options,
        max_iter,
        tol,
        shrink,
        verbose,
        trace_freq,
    )
    return result


def _make_loss_rehline_param(loss, X, y):
    """The `_make_loss_rehline_param` function generates parameters for the ReHLine solver, based on the provided training data.

    The function supports various loss functions, including:
        - 'hinge' or 'svm' or 'SVM'
        - 'squared hinge' or 'squared svm' or 'squared SVM'
        - 'mae' or 'MAE' or 'mean absolute error'
        - 'check' or 'quantile' or 'quantile regression' or 'QR'
        - 'sSVM' or 'smooth SVM' or 'smooth hinge'
        - 'TV'
        - 'huber' or 'Huber'
        - 'SVR' or 'svr'
        - Custom loss functions (manual setup required)

    Parameters
    ----------
    loss : dict
        A dictionary containing the loss function parameters.

        Keys:
            - 'name' : str, the name of the loss function (e.g. 'hinge', 'svm', 'QR', etc.)
            - 'loss_kwargs': more keys and values for loss parameters

    X : ndarray of shape (n_samples, n_features)
        The generated samples.

    y : ndarray of shape (n_samples,)
        Numeric targets, or +/-1 labels for classification losses. Converted
        to float64 before loss arithmetic, including for integer inputs.
    """

    if not isinstance(loss, dict) or not isinstance(loss.get("name"), str):
        raise ValueError("loss must be a dictionary with a string 'name'")
    required = []
    if loss["name"] in ("check", "quantile", "quantile regression", "QR", "check_eps", "QR_eps", "quantile_eps"):
        required.append("qt")
    if loss["name"] in ("SVR", "svr", "check_eps", "QR_eps", "quantile_eps"):
        required.append("epsilon")
    for key in required:
        if key not in loss:
            raise ValueError(f"loss requires parameter {key!r}")
    for key in ("qt", "tau", "epsilon"):
        if key in loss:
            positive_real(loss[key], f"loss[{key!r}]", allow_zero=(key != "qt"))
    if "qt" in loss and loss["qt"] >= 1:
        raise ValueError("loss['qt'] must be in (0, 1)")

    # Convert before negation/addition: integer arithmetic (especially unsigned
    # targets) can wrap before the native boundary ever sees the loss arrays.
    y = numeric_array(y, "y", ndim=1)
    # n, d = X.shape
    n = len(y)

    ## initialization of ReHLine params
    U = np.empty(shape=(0, 0))
    V = np.empty(shape=(0, 0))
    Tau = np.empty(shape=(0, 0))
    S = np.empty(shape=(0, 0))
    T = np.empty(shape=(0, 0))

    # _dummy_X = False

    if (loss["name"] == "hinge") or (loss["name"] == "svm") or (loss["name"] == "SVM"):
        U = -y.reshape(1, -1)
        V = (np.array(np.ones(n))).reshape(1, -1)

    elif (
        (loss["name"] == "check")
        or (loss["name"] == "quantile")
        or (loss["name"] == "quantile regression")
        or (loss["name"] == "QR")
    ):
        qt = loss["qt"]

        U = np.ones((2, n))
        V = np.ones((2, n))

        U[0] = -qt * U[0]
        U[1] = (1 - qt) * U[1]
        V[0] = qt * V[0] * y
        V[1] = -(1 - qt) * V[1] * y

    # elif (loss['name'] == 'CQR') \

    #     n_qt = len(loss['qt'])
    #     U = np.ones((2, n*n_qt))
    #     V = np.ones((2, n*n_qt))
    #     X_fake = np.zeros((n*n_qt, d+n_qt))

    #     for l,qt_tmp in enumerate(loss['qt']):
    #         U[0,l*n:(l+1)*n] = - (qt_tmp*U[0,l*n:(l+1)*n])
    #         U[1,l*n:(l+1)*n] = ((1.-qt_tmp)*U[1,l*n:(l+1)*n])

    #         V[0,l*n:(l+1)*n] = qt_tmp*V[0,l*n:(l+1)*n]*y
    #         V[1,l*n:(l+1)*n] = - (1.-qt_tmp)*V[1,l*n:(l+1)*n]*y

    #         X_fake[l*n:(l+1)*n,:d] = X
    #         X_fake[l*n:(l+1)*n,d+l] = 1.

    elif (loss["name"] == "sSVM") or (loss["name"] == "smooth SVM") or (loss["name"] == "smooth hinge"):
        S = np.ones((1, n))
        T = np.ones((1, n))
        Tau = np.ones((1, n))
        S[0] = -y

    elif loss["name"] == "TV":
        U = np.ones((2, n))
        V = np.ones((2, n))
        U[1] = -U[1]

        V[0] = -X.dot(y)
        V[1] = X.dot(y)

    elif (loss["name"] == "huber") or (loss["name"] == "Huber"):
        S = np.ones((2, n))
        T = np.ones((2, n))
        tau_tmp = loss.get("tau", 1.0)
        Tau = tau_tmp * np.ones((2, n))

        S[0] = -S[0]
        T[0] = y
        T[1] = -y

    elif loss["name"] in ["SVR", "svr"]:
        U = np.ones((2, n))
        V = np.ones((2, n))
        U[1] = -U[1]

        V[0] = -(y + loss["epsilon"])
        V[1] = y - loss["epsilon"]

    elif loss["name"] in ["check_eps", "QR_eps", "quantile_eps"]:
        # Check Loss with epsilon-tolerance: (rho_kappa(r) - epsilon)_+
        qt = loss["qt"]  # kappa (quantile level)
        epsilon = loss["epsilon"]

        U = np.zeros((2, n))
        V = np.zeros((2, n))

        U[0] = -qt
        U[1] = 1 - qt
        V[0] = qt * y - epsilon
        V[1] = -(1 - qt) * y - epsilon

    elif (loss["name"] == "MAE") or (loss["name"] == "mae") or (loss["name"] == "mean absolute error"):
        U = np.array([[1.0] * n, [-1.0] * n])
        V = np.array([-y, y])

    elif (loss["name"] == "squared SVM") or (loss["name"] == "squared svm") or (loss["name"] == "squared hinge"):
        Tau = np.inf * np.ones((1, n))
        S = -np.sqrt(2) * y.reshape(1, -1)
        T = np.sqrt(2) * np.ones((1, n))

    elif (loss["name"] == "MSE") or (loss["name"] == "mse") or (loss["name"] == "mean squared error"):
        Tau = np.inf * np.ones((2, n))
        S = np.array([[np.sqrt(2)] * n, [-np.sqrt(2)] * n])
        T = np.array([-np.sqrt(2) * y, np.sqrt(2) * y])

    else:
        raise ValueError(
            "Sorry, ReHLine currently does not support this loss function, "
            "but you can manually set ReHLine params to solve the problem via `ReHLine` class."
        )

    return U, V, Tau, S, T


def _combined_constraints(constraint, A=None, b=None, *, warn=True):
    """Combine constraint lists: every supplied constraint must hold."""
    constraints = list([] if constraint is None else constraint)
    if A is not None or b is not None:
        if A is None or b is None:
            raise ValueError("A and b must be supplied together")
        A = numeric_array(A, "A", ndim=2)
        b = numeric_array(b, "b", ndim=1)
        if b.shape != (A.shape[0],):
            raise ValueError("b must have one entry per row of A")
        # Legacy empty defaults do not specify any constraints.
        if A.shape != (0, 0):
            if warn and constraints and A.shape[0]:
                warnings.warn(
                    "Both A/b and constraint were supplied; combining them so all constraints are enforced.",
                    UserWarning,
                    stacklevel=3,
                )
            constraints.append({"name": "custom", "A": A, "b": b})
    return constraints


def _make_constraint_rehline_param(constraint, X, y=None):
    """The `_make_constraint_rehline_param` function generates constraint parameters for the ReHLine solver.

    Parameters
    ----------
    constraint : list of dict
        A list of dictionaries, where each dictionary represents a constraint.
        Each dictionary must contain a 'name' key, which specifies the type of constraint.
        The following constraint types are supported:
            * 'nonnegative' or '>=0': A non-negativity constraint.
            * 'fair' or 'fairness': A fairness constraint using 'sen_idx' and 'tol_sen'.
            * 'custom': A custom constraint, where the user must provide the constraint matrix 'A' and vector 'b'.

    X : array-like of shape (n_samples, n_features)
        The design matrix.

    y : array-like of shape (n_samples,), default=None
        The target variable. Not used in this function.

    Returns
    -------
    A : array-like of shape (n_constraints, n_features)
        The constraint matrix.

    b : array-like of shape (n_constraints,)
        The constraint vector.
    """

    n, d = X.shape

    ## initialization
    A = np.empty(shape=(0, 0))
    b = np.empty(shape=(0))

    for constr_tmp in [] if constraint is None else constraint:
        if not isinstance(constr_tmp, dict) or "name" not in constr_tmp:
            raise ValueError("Each constraint must be a dictionary with a name")
        if (constr_tmp["name"] == "nonnegative") or (constr_tmp["name"] == ">=0"):
            A_tmp = np.identity(d)
            b_tmp = np.zeros(d)

        elif (constr_tmp["name"] == "fair") or (constr_tmp["name"] == "fairness"):
            if n == 0:
                raise ValueError("Fairness constraints require observations")
            sen_idx = np.atleast_1d(constr_tmp["sen_idx"])
            if (
                sen_idx.ndim != 1
                or sen_idx.size == 0
                or not np.issubdtype(sen_idx.dtype, np.integer)
                or np.any((sen_idx < -d) | (sen_idx >= d))
            ):
                raise ValueError("sen_idx must contain valid integer feature indices")
            tol_sen = numeric_array(np.atleast_1d(constr_tmp["tol_sen"]), "tol_sen", ndim=1)
            if np.any(tol_sen < 0):
                raise ValueError("tol_sen must be non-negative")

            # Shift before averaging to avoid subtracting two large means.
            # A genuinely constant column becomes exactly zero, including
            # decimal constants whose arithmetic mean may round differently.
            # Only this temporary covariance copy is centered, never solver X.
            features = numeric_array(X, "X", ndim=2)
            centered_X = features - features[0]
            centered_X -= centered_X.mean(axis=0)
            X_sen = centered_X[:, sen_idx]

            if X_sen.shape[1] != len(tol_sen):
                raise ValueError("dim of X_sen and len of tol_sen must be equal")

            A_tmp = np.repeat(X_sen.T @ centered_X, repeats=[2], axis=0) / n
            A_tmp[::2] = -A_tmp[::2]
            b_tmp = np.repeat(tol_sen, repeats=[2], axis=0)

        elif (constr_tmp["name"] == "monotonic") or (constr_tmp["name"] == "monotonicity"):
            decreasing = constr_tmp.get("decreasing", False)
            idx = np.arange(d - 1)
            A_tmp = np.zeros((d - 1, d))
            if decreasing:
                A_tmp[idx, idx] = 1.0
                A_tmp[idx, idx + 1] = -1.0
            else:
                A_tmp[idx, idx] = -1.0
                A_tmp[idx, idx + 1] = 1.0
            b_tmp = np.zeros(d - 1)

        elif constr_tmp["name"] == "custom":
            A_tmp = constr_tmp["A"]
            b_tmp = constr_tmp["b"]

        else:
            raise ValueError(
                "Sorry, ReHLine currently does not support this constraint, "
                "but you can add it by manually setting A and b via {'name': 'custom', 'A': A, 'b': b}"
            )

        A_tmp = numeric_array(A_tmp, "A", ndim=2)
        b_tmp = numeric_array(b_tmp, "b", ndim=1)
        if A_tmp.shape[1] != d or b_tmp.shape != (A_tmp.shape[0],):
            raise ValueError(f"Constraint A must have {d} columns and b one entry per row")
        A = np.vstack([A, A_tmp]) if A.size else A_tmp
        b = np.hstack([b, b_tmp]) if b.size else b_tmp

    return A, b


def _make_penalty_rehline_param(penalty=None, X=None):
    """The `_make_penalty_rehline_param` function generates penalty parameters for the ReHLine solver."""
    raise NotImplementedError("Sorry, `_make_penalty_rehline_param` feature is currently under development.")


def _cast_sample_bias(U, V, Tau, S, T, sample_bias=None):
    """Cast sample bias to ReHLine parameters by injecting bias into V and T.

    This function modifies the ReHLine parameters to incorporate individual
    sample biases through linear transformations of the intercept parameters.

    Parameters
    ----------
    U : array-like of shape (L, n_samples)
        ReLU coefficient matrix.

    V : array-like of shape (L, n_samples)
        ReLU intercept vector.

    Tau : array-like of shape (H, n_samples)
        ReHU cutpoint matrix.

    S : array-like of shape (H, n_samples)
        ReHU coefficient vector.

    T : array-like of shape (H, n_samples)
        ReHU intercept vector.

    sample_bias : array-like of shape (n_samples, 1)
        Individual sample bias vector. If None, parameters are returned unchanged.

    Returns
    -------
    U_bias : array-like of shape (L, n_samples)
        Biased coefficient matrix, actually doesn't change

    V_bias : array-like of shape (L, n_samples)
        Biased ReLU intercept vector: V + U * sample_bias

    Tau_bias : array-like of shape (H, n_samples)
        Biased ReHU cutpoint matrix, actually doesn't change

    S_bias : array-like of shape (H, n_samples)
        Biased ReHU coefficient vector, actually doesn't change

    T_bias : array-like of shape (H, n_samples)
        Biased ReHU intercept vector: T + S * sample_bias

    Notes
    -----
    The transformation applies the sample bias through:
    - V_bias = V + U ⊙ sample_bias
    - T_bias = T + S ⊙ sample_bias

    where ⊙ denotes element-wise multiplication with broadcasting.
    """
    if sample_bias is None:
        return U, V, Tau, S, T

    else:
        sample_bias = sample_bias.reshape(1, -1)
        U_bias = U
        V_bias = V + (U * sample_bias if U.shape[0] > 0 else 0)
        Tau_bias = Tau
        S_bias = S
        T_bias = T + (S * sample_bias if S.shape[0] > 0 else 0)

        return U_bias, V_bias, Tau_bias, S_bias, T_bias


def _cast_sample_weight(U, V, Tau, S, T, C=1.0, sample_weight=None):
    """Apply sample weights and regularization to ReHLine parameters.

    Parameters
    ----------
    U : array-like of shape (L, n_samples)
        ReLU coefficient matrix.

    V : array-like of shape (L, n_samples)
        ReLU intercept vector.

    Tau : array-like of shape (H, n_samples)
        ReHU cutpoint matrix.

    S : array-like of shape (H, n_samples)
        ReHU coefficient vector.

    T : array-like of shape (H, n_samples)
        ReHU intercept vector.

    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.

    sample_weight : array-like of shape (n_samples,), default=None
        Individual sample weight. If None, then samples are equally weighted.

    Returns
    -------
    U_weight : array-like of shape (L, n_samples)
        Weighted ReLU coefficient matrix.

    V_weight : array-like of shape (L, n_samples)
        Weighted ReLU intercept vector.

    Tau_weight : array-like of shape (H, n_samples)
        Weighted ReHU cutpoint matrix.

    S_weight : array-like of shape (H, n_samples)
        Weighted ReHU coefficient vector.

    T_weight : array-like of shape (H, n_samples)
        Weighted ReHU intercept vector.

    Notes
    -----
    This function casts the sample weight to the ReHLine parameters by multiplying
    the sample weight with the ReLU and ReHU parameters. If sample_weight is None,
    then the sample weight is set to the regularization parameter C.
    """
    positive_real(C, "C")
    n = max(U.shape[1], S.shape[1], np.size(sample_weight) if sample_weight is not None else 0)
    sample_weight = C * sample_weights(sample_weight, n)

    if U.shape[0] > 0:
        U_weight = U * sample_weight
        V_weight = V * sample_weight
    else:
        U_weight = U
        V_weight = V

    if S.shape[0] > 0:
        sqrt_sample_weight = np.sqrt(sample_weight)
        # A zero-weight term is identically zero even when Tau is infinite.
        Tau_weight = np.zeros_like(Tau, dtype=np.float64)
        np.multiply(Tau, sqrt_sample_weight, out=Tau_weight, where=sqrt_sample_weight > 0)
        S_weight = S * sqrt_sample_weight
        T_weight = T * sqrt_sample_weight
    else:
        Tau_weight = Tau
        S_weight = S
        T_weight = T

    return U_weight, V_weight, Tau_weight, S_weight, T_weight


# def append_l1(self, X, l1_pen=1.0):
#     r"""
#     This function appends the l1 penalty to the ReHLine problem. The formulation becomes:

#     .. math::

#         \min_{\mathbf{\beta} \in \mathbb{R}^d} \sum_{i=1}^n \sum_{l=1}^L \text{ReLU}( u_{li} \mathbf{x}_i^\intercal \mathbf{\beta} + v_{li}) + \sum_{i=1}^n \sum_{h=1}^H {\text{ReHU}}_{\tau_{hi}}( s_{hi} \mathbf{x}_i^\intercal \mathbf{\beta} + t_{hi}) + \frac{1}{2} \| \mathbf{\beta} \|_2^2 + \lambda_1 \| \mathbf{\beta} \|_1, \\ \text{ s.t. }
#         \mathbf{A} \mathbf{\beta} + \mathbf{b} \geq \mathbf{0},

#     where :math:`\lambda_1` is associated with `l1_pen`.

#     Parameters
#     ----------

#     X : ndarray of shape (n_samples, n_features)
#         The generated samples.

#     l1_pen : float, default=1.0
#         The l1 penalty level, which controls the complexity or sparsity of the resulting model.

#     Returns
#     -------

#     X_fake: ndarray of shape (n_samples+n_features, n_features)
#         The manipulated data matrix. It has been padded with
#         identity matrix, allowing the correctly structured data to be input
#         into `self.fit` or other modelling processes.

#     Examples
#     --------

#     >>> import numpy as np
#     >>> from rehline import ReHLine

#     >>> # simulate classification dataset
#     >>> n, d, C, lam1 = 1000, 3, 0.5, 1.0
#     >>> np.random.seed(1024)
#     >>> X = np.random.randn(1000, 3)
#     >>> beta0 = np.random.randn(3)
#     >>> y = np.sign(X.dot(beta0) + np.random.randn(n))

#     >>> clf = ReHLine(loss={'name': 'svm'}, C=C)
#     >>> clf.make_ReLHLoss(X=X, y=y, loss={'name': 'svm'})
#     >>> # save and fit with the manipulated data matrix
#     >>> X_fake = clf.append_l1(X, l1_pen=lam1)
#     >>> clf.fit(X=X_fake)
#     >>> print('sol provided by rehline: %s' %clf.coef_)
#     >>> sol provided by rehline: [ 7.17796629e-01 -1.87075728e-06  2.61965622e+00] #sparse sol
#     >>> print(clf.decision_function([[.1,.2,.3]]))
#     >>> [0.85767616]
#     """

#     n, d = X.shape
#     l1_pen = l1_pen*np.ones(d)
#     U_new = np.zeros((self.L+2, n+d))
#     V_new = np.zeros((self.L+2, n+d))
#     ## Block 1
#     if len(self._U):
#         U_new[:self.L, :n] = self._U
#         V_new[:self.L, :n] = self._V
#     ## Block 2
#     U_new[-2,n:] = l1_pen
#     U_new[-1,n:] = -l1_pen

#     if len(self._S):
#         S_new = np.zeros((self.H, n+d))
#         T_new = np.zeros((self.H, n+d))
#         Tau_new = np.zeros((self.H, n+d))

#         S_new[:,:n] = self._S
#         T_new[:,:n] = self._T
#         Tau_new[:,:n] = self._Tau

#         self._S = S_new
#         self._T = T_new
#         self._Tau = Tau_new

#     ## fake X
#     X_fake = np.zeros((n+d, d))
#     X_fake[:n,:] = X
#     X_fake[n:,:] = np.identity(d)

#     self._U = U_new
#     self._V = V_new
#     self.auto_shape()
#     return X_fake
