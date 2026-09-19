from copy import copy, deepcopy
from itertools import combinations

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _check_sample_weight, check_is_fitted, validate_data

from ._base import _combined_constraints, _make_constraint_rehline_param
from ._class import plqERM_ElasticNet, plqERM_Ridge
from ._validation import (
    balanced_sample_weights,
    model_options,
    named_loss_parameters,
    numeric_array,
    positive_real,
    sample_weights,
)


class _SklearnReHLine(BaseEstimator):
    """Common data preparation, constraints and state handling for sklearn models."""

    def get_params(self, deep=True):
        return BaseEstimator.get_params(self, deep=deep)

    def _fit_model(self, X, y, weight, previous=None):
        n, d = X.shape
        X_aug = np.column_stack((X, np.full(n, self.intercept_scaling))) if self.fit_intercept else X
        matrices, offsets = [], []
        for constraint in _combined_constraints(self.constraint, self.A, self.b, warn=False):
            if (
                self.fit_intercept
                and isinstance(constraint, dict)
                and constraint.get("name") == "custom"
                and np.shape(constraint.get("A"))[1:] == (d + 1,)
            ):
                # An explicit final column constrains the actual intercept.
                A = numeric_array(constraint["A"], "A", ndim=2).copy()
                b = numeric_array(constraint["b"], "b", ndim=1)
                A[:, -1] *= self.intercept_scaling
                if b.shape != (A.shape[0],):
                    raise ValueError("b must have one entry per row of A")
            else:
                A, b = _make_constraint_rehline_param([constraint], X, y)
                if self.fit_intercept:
                    A = np.column_stack((A, np.zeros(A.shape[0])))
            matrices.append(A)
            offsets.append(b)
        constraint_params = []
        if matrices:
            constraint_params = [{"name": "custom", "A": np.vstack(matrices), "b": np.concatenate(offsets)}]
        kwargs = dict(
            loss=deepcopy(self.loss) if self.loss is not None else {"name": "QR", "qt": 0.5},
            constraint=constraint_params,
            C=self.C,
            max_iter=self.max_iter,
            tol=self.tol,
            shrink=self.shrink,
            warm_start=self.warm_start,
            verbose=self.verbose,
            trace_freq=self.trace_freq,
        )
        if hasattr(self, "l1_ratio"):
            omega = np.empty(0) if self.omega is None else numeric_array(self.omega, "omega", ndim=1)
            if omega.size not in (0, d) or np.any(omega < 0):
                raise ValueError(f"omega must be non-negative and empty or have {d} entries")
            if self.fit_intercept and omega.size:
                omega = np.append(omega, 1.0)
            kwargs.update(l1_ratio=self.l1_ratio, omega=omega)
            model = plqERM_ElasticNet(**kwargs)
        else:
            model = plqERM_Ridge(**kwargs)
        if self.warm_start and previous is not None:
            # Reuse duals only when the loss structure and dimensions agree.
            if (
                previous.loss == kwargs["loss"]
                and bool(getattr(previous, "l1_ratio", 0)) == bool(getattr(self, "l1_ratio", 0))
                and previous.n_features_in_ == X_aug.shape[1]
                and previous._U.shape[1] in (0, n)
                and previous._S.shape[1] in (0, n)
                and previous._A.shape[0] == sum(len(b) for b in offsets)
            ):
                for name in ("_Lambda", "_Gamma", "_xi", "_mu", "_xi_row_scale"):
                    if hasattr(previous, name):
                        setattr(model, name, getattr(previous, name).copy())
        model.fit(X_aug, y, sample_weight=weight)
        coef = model.coef_[:-1].copy() if self.fit_intercept else model.coef_.copy()
        intercept = float(model.coef_[-1] * self.intercept_scaling) if self.fit_intercept else 0.0
        return model, coef, intercept

    def fit(self, X, y, sample_weight=None):
        """Fit dense features and targets with optional non-negative sample weights.

        Parameters are validated at fit time. Built-in constraints act on feature
        coefficients; a custom final column can explicitly constrain the intercept.
        Returns this estimator, with final optimization diagnostics available.

        Fitted state is committed only after a successful fit. If fitting raises,
        the last successful fitted state is retained; constructor parameters
        changed with set_params are not rolled back.
        """
        # Fitting builds fresh inner models and copies reused dual arrays. A
        # shallow staging copy therefore shares no state that fitting mutates.
        staged = copy(self)
        staged._fit_inplace(X, y, sample_weight)
        self.__dict__ = staged.__dict__
        return self

    def _fit_inplace(self, X, y, sample_weight):
        model_options(self)
        named_loss_parameters(self)
        # Validate and announce combined constraints once per public fit, including
        # multiclass fits. Worker tasks construct their matrices without warnings.
        _combined_constraints(self.constraint, self.A, self.b)
        positive_real(self.intercept_scaling, "intercept_scaling")
        if not isinstance(self.fit_intercept, (bool, np.bool_)):
            raise ValueError("fit_intercept must be boolean")
        X, y = validate_data(self, X, y, accept_sparse=False, dtype=np.float64, order="C")
        weight = sample_weights(_check_sample_weight(sample_weight, X, dtype=np.float64), X.shape[0])
        # Removing zero-weight rows also keeps class labels and constraints consistent.
        active = weight > 0
        if not np.all(active):
            X, y, weight = X[active], y[active], weight[active]
        classifier = isinstance(self, ClassifierMixin)
        previous = getattr(self, "_model_", None)
        if classifier:
            check_classification_targets(y)
            old_classes = getattr(self, "classes_", None)
            self._label_encoder = LabelEncoder().fit(y)
            self.classes_ = self._label_encoder.classes_
            if self.classes_.size < 2:
                raise ValueError("Classifier requires at least 2 classes; got 1 class")
            if self.class_weight is not None:
                encoded = self._label_encoder.transform(y)
                if isinstance(self.class_weight, str) and self.class_weight == "balanced":
                    weight = balanced_sample_weights(encoded, weight, len(self.classes_))
                else:
                    class_weights = compute_class_weight(self.class_weight, classes=self.classes_, y=y)
                    weight = weight * class_weights[encoded]
                weight = sample_weights(weight, len(y))
                active = weight > 0
                if not np.all(active):
                    X, y, weight = X[active], y[active], weight[active]
                self._label_encoder = LabelEncoder().fit(y)
                self.classes_ = self._label_encoder.classes_
                if self.classes_.size < 2:
                    raise ValueError("Classifier requires at least 2 classes with positive weight; got 1 class")
            if old_classes is not None and not np.array_equal(old_classes, self.classes_):
                previous = None
            self.multi_class_ = "ovr" if self.multi_class is None or self.multi_class == [] else self.multi_class
            if self.multi_class_ not in ("ovr", "ovo"):
                raise ValueError("multi_class must be 'ovr' or 'ovo'")
            self._validate_decision_function_shape()
            if self.classes_.size > 2:
                self._fit_multiclass(X, y, weight)
                return self
            y = 2 * self._label_encoder.transform(y) - 1
        elif not np.issubdtype(y.dtype, np.number):
            y = y.astype(np.float64)
        self._model_, self.coef_, self.intercept_ = self._fit_model(X, y, weight, previous)
        for name in (
            "n_iter_",
            "dual_obj_",
            "primal_obj_",
            "objective_",
            "dual_objective_",
            "dual_gap_",
            "constraint_violation_",
            "scaled_constraint_violation_",
            "kkt_residual_",
            "converged_",
            "_Lambda",
            "_Gamma",
            "_xi",
            "_mu",
            "_U",
            "_V",
            "_S",
            "_T",
            "_Tau",
            "_A",
            "_b",
        ):
            if hasattr(self._model_, name):
                setattr(self, name, getattr(self._model_, name))
        for name in ("estimators_", "_models_", "_model_keys_", "_multiclass_signature_"):
            self.__dict__.pop(name, None)
        return self

    def _fit_multiclass_task(self, X, y, weight, key, rows, previous):
        """Create temporary subsets inside workers, preserving input row order."""
        if rows is not None:
            selected = np.sort(np.concatenate(rows))
            X, y, weight = X[selected], y[selected], weight[selected]
        target = np.where(y == key[-1], 1.0, -1.0)
        return self._fit_model(X, target, weight, previous)

    def _fit_multiclass(self, X, y, weight):
        signature = (self.multi_class_, tuple(self.classes_))
        previous = {}
        if self.warm_start and getattr(self, "_multiclass_signature_", None) == signature:
            previous = dict(zip(self._model_keys_, self._models_))
        if self.multi_class_ == "ovr":
            pairs = None
            keys = [(c,) for c in self.classes_]
            class_rows = None
        else:
            pairs = list(combinations(self.classes_, 2))
            keys = pairs
            class_rows = {c: np.flatnonzero(y == c) for c in self.classes_}
        results = Parallel(n_jobs=self.n_jobs, prefer="threads", pre_dispatch="n_jobs", batch_size=1)(
            delayed(self._fit_multiclass_task)(
                X,
                y,
                weight,
                key,
                None if class_rows is None else (class_rows[key[0]], class_rows[key[1]]),
                previous.get(key),
            )
            for key in keys
        )
        self._models_ = [model for model, _, _ in results]
        self._model_keys_ = keys
        self._multiclass_signature_ = signature
        self.coef_ = np.array([coef for _, coef, _ in results])
        self.intercept_ = np.array([intercept for _, _, intercept in results])
        self.estimators_ = [
            (coef, intercept) if pairs is None else (coef, intercept, *pairs[k])
            for k, (_, coef, intercept) in enumerate(results)
        ]
        for name in (
            "n_iter_",
            "objective_",
            "dual_objective_",
            "dual_gap_",
            "constraint_violation_",
            "scaled_constraint_violation_",
            "kkt_residual_",
            "converged_",
        ):
            setattr(self, name, np.array([getattr(model, name) for model in self._models_]))
        for name in (
            "_model_",
            "primal_obj_",
            "dual_obj_",
            "_U",
            "_V",
            "_S",
            "_T",
            "_Tau",
            "_A",
            "_b",
            "_Lambda",
            "_Gamma",
            "_xi",
            "_mu",
        ):
            self.__dict__.pop(name, None)

    def _decision_function(self, X):
        check_is_fitted(self, ["coef_", "intercept_"])
        X = validate_data(self, X, reset=False, accept_sparse=False, dtype=np.float64, order="C")
        return X @ self.coef_.T + self.intercept_

    def to_inference(self):
        """Return an independent prediction snapshot without training caches.

        Coefficients, feature/class metadata, score format and final diagnostics
        are copied. The snapshot supports predict, classifier decision_function
        and pickle/joblib serialization. It cannot be fitted or warm-started;
        the original estimator remains trainable.
        """
        from ._inference import _sklearn_snapshot

        return _sklearn_snapshot(self)

    def predict(self, X):
        check_is_fitted(self, ["coef_", "intercept_"])
        if isinstance(self, ClassifierMixin) and self.classes_.size > 2 and self.multi_class_ == "ovo":
            return self.classes_[self._ovo_class_scores(X).argmax(axis=1)]
        scores = self._decision_function(X)
        if not isinstance(self, ClassifierMixin):
            return scores
        if self.classes_.size == 2:
            return self._label_encoder.inverse_transform((scores > 0).astype(int))
        return self.classes_[self._class_scores(scores).argmax(axis=1)]

    def _class_scores(self, scores):
        """Aggregate OvO margins in classes_ order using the prediction rule."""
        if self.classes_.size == 2 or self.multi_class_ == "ovr":
            return scores
        votes = np.zeros((len(scores), len(self.classes_)))
        confidence = np.zeros_like(votes)
        for k, (_, _, a, b) in enumerate(self.estimators_):
            i, j = np.searchsorted(self.classes_, [a, b])
            positive = scores[:, k] > 0  # Positive score favors the second class.
            votes[:, j] += positive
            votes[:, i] += ~positive
            confidence[:, j] += scores[:, k]
            confidence[:, i] -= scores[:, k]
        confidence /= 3 * (np.abs(confidence) + 1)
        return votes + confidence

    def _ovo_class_scores(self, X):
        """Aggregate pair margins in bounded blocks, preserving pair/tie order."""
        X = validate_data(self, X, reset=False, accept_sparse=False, dtype=np.float64, order="C")
        votes = np.zeros((len(X), len(self.classes_)))
        confidence = np.zeros_like(votes)
        for start in range(0, len(self.coef_), 64):
            stop = start + 64
            scores = X @ self.coef_[start:stop].T + self.intercept_[start:stop]
            for column, (_, _, a, b) in enumerate(self.estimators_[start:stop]):
                i, j = np.searchsorted(self.classes_, [a, b])
                margin = scores[:, column]
                positive = margin > 0
                votes[:, j] += positive
                votes[:, i] += ~positive
                confidence[:, j] += margin
                confidence[:, i] -= margin
        confidence /= 3 * (np.abs(confidence) + 1)
        return votes + confidence

    def call_ReLHLoss(self, score):
        check_is_fitted(self, "_model_")
        return self._model_.call_ReLHLoss(score)


class _ReHLineClassifier(ClassifierMixin, _SklearnReHLine):
    def _validate_decision_function_shape(self):
        if not isinstance(self.decision_function_shape, str) or self.decision_function_shape not in ("ovr", "ovo"):
            raise ValueError("decision_function_shape must be 'ovr' or 'ovo'")
        if self.classes_.size > 2 and self.decision_function_shape == "ovo" and self.multi_class_ != "ovo":
            raise ValueError("decision_function_shape='ovo' requires multi_class='ovo' for multiclass models")

    def decision_function(self, X):
        """Return decision scores in the requested sklearn-style output format.

        Binary output has shape (n_samples,), positive for ``classes_[1]``.
        Binary ``predict`` selects ``classes_[0]`` at exactly zero, matching
        sklearn linear classifiers and the zero-margin vote in native OvO.
        With ``decision_function_shape='ovr'`` (default), multiclass output has
        shape (n_samples, n_classes), in ``classes_`` order. OvO training combines
        pair votes with bounded confidence scores to break vote ties.

        With ``decision_function_shape='ovo'``, multiclass OvO output has shape
        (n_samples, n_classes * (n_classes - 1) / 2). Columns follow sorted class
        pairs, positive for the FIRST class, like SVC. These margins equal
        ``-(X @ coef_.T + intercept_)``: stored coefficients retain their training
        orientation, positive for the second class. Multiclass OvR training
        cannot provide this output and raises ValueError.

        Changing the output format after fitting requires no refit and does not
        change ``predict``, coefficients, constraints or objective values.
        """
        check_is_fitted(self, ["coef_", "intercept_"])
        self._validate_decision_function_shape()
        if self.classes_.size > 2 and self.multi_class_ == "ovo" and self.decision_function_shape == "ovr":
            return self._ovo_class_scores(X)
        scores = self._decision_function(X)
        if self.classes_.size > 2 and self.decision_function_shape == "ovo":
            # SVC's multiclass pair scores favor the first class. Keep the
            # training orientation unchanged, including asymmetric constraints.
            return -scores
        return self._class_scores(scores)


class plq_Ridge_Classifier(_ReHLineClassifier):
    """
    Empirical Risk Minimization (ERM) Classifier with a Piecewise Linear-Quadratic (PLQ) loss
    and ridge penalty, compatible with the scikit-learn API.

    This wrapper makes ``plqERM_Ridge`` behave as a classifier:
        - Accepts arbitrary binary labels in the original label space.
        - Computes class weights on original labels (if ``class_weight`` is set).
        - Encodes labels with ``LabelEncoder`` into {0,1}, then maps to {-1,+1} for training.
        - Supports optional intercept fitting (via an augmented constant feature).
        - Provides standard methods ``fit``, ``predict``, and ``decision_function``.
        - Integrates with scikit-learn ecosystem (e.g., GridSearchCV, Pipeline).
        - Supports multiclass classification via OvR or OvO method.

    Parameters
    ----------
    loss : dict
        Dictionary specifying the loss function parameters. Examples include:
        - {'name': 'svm'}
        - {'name': 'sSVM'}
        - {'name': 'huber'}
        and other PLQ losses supported by ``plqERM_Ridge``.

    constraint : list of dict or None, default=None
        Optional constraints. Each dictionary must include a ``'name'`` key.
        Examples: {'name': 'nonnegative'}, {'name': 'fair'}, {'name': 'custom'}.

    C : float, default=1.0
        Inverse regularization strength.

    U, V, S, T, Tau : None or empty array, default=None
        Legacy constructor parameters. Nonempty values raise ValueError at fit
        time because these matrices are generated from ``loss``. Use ReHLine
        or ReHLine_solver for manually specified loss matrices.

    _A : ndarray of shape (K, n_features), default empty
        Linear-constraint coefficient matrix.

    _b : ndarray of shape (K,), default empty
        Linear-constraint intercept vector.

    max_iter : int, default=1000
        Maximum number of iterations for the ReHLine solver.

    tol : float, default=1e-4
        Convergence tolerance.

    shrink : int, default=1
        Shrinkage parameter for the solver.

    warm_start : int, default=0
        Whether to reuse the previous solution for initialization.

    verbose : int, default=0
        Verbosity level for the solver.

    trace_freq : int, default=100
        Frequency (in iterations) at which solver progress is traced
        when ``verbose > 0``.

    fit_intercept : bool, default=True
        Whether to fit an intercept term. If True, a constant feature column is added
        to ``X`` during training. The last learned coefficient is extracted as
        ``intercept_``.

    intercept_scaling : float, default=1.0
        Value used for the constant feature column when ``fit_intercept=True``.
        Matches the convention used in scikit-learn's ``LinearSVC``.

    class_weight : dict, 'balanced', or None, default=None
        'balanced' gives sample i effective weight
        w_i * sum(w) / (n_classes * sum(w[y == y_i])). Balancing uses
        retained original classes before binary/OvR/OvO decomposition.
        Zero-weight rows are excluded. This matches LinearSVC >= 1.7 and
        has the same definition with every supported sklearn version.
        A dict maps original labels to multipliers of sample_weight.

    multi_class : str or None, default=None
        Method for multiclass classification. Options:
        - 'ovo': One-vs-One, trains K*(K-1)/2 binary classifiers.
        - 'ovr': One-vs-Rest, trains K binary classifiers.
        - None selects OvR; binary problems use one estimator.

    decision_function_shape : {'ovr', 'ovo'}, default='ovr'
        Score output format, independent of the training strategy. 'ovr' returns
        one score per class; 'ovo' requires OvO training and returns one margin
        per sorted class pair, positive for the first class (like SVC). Binary
        output is always 1D, positive for classes_[1]. Changing this parameter
        after fitting does not change predictions or the fitted optimization.

    n_jobs : int or None, default=None
        Number of parallel jobs for multiclass fitting.
        None means 1 (serial). -1 means use all available CPUs.
        Passed directly to joblib.Parallel.


    Attributes
    ----------
    ``coef_``: ndarray of shape (n_features,) for binary, (n_estimators, n_features) for multiclass
        Coefficients of all fitted classifiers, excluding the intercept.
        OvO training coefficients favor the second class of each sorted pair;
        raw multiclass decision scores use the opposite sign, like SVC.

    ``intercept_``: float for binary, ndarray of shape (n_estimators,) for multiclass
        Intercept term(s). 0.0 if ``fit_intercept=False``.

    ``classes_`` : ndarray of shape (n_classes,)
        Unique class labels in the original label space.

    ``estimators_`` : list, only present for multiclass
        For OvR: list of (coef, intercept) tuples, length K.
        For OvO: list of (coef, intercept, cls_i, cls_j) tuples, length K*(K-1)/2.

    _label_encoder : LabelEncoder
        Encodes original labels into {0,1} for internal training.
    """

    def __init__(
        self,
        loss,
        constraint=None,
        C=1.0,
        U=None,
        V=None,
        Tau=None,
        S=None,
        T=None,
        A=None,
        b=None,
        max_iter=1000,
        tol=1e-4,
        shrink=1,
        warm_start=0,
        verbose=0,
        trace_freq=100,
        fit_intercept=True,
        intercept_scaling=1.0,
        class_weight=None,
        multi_class=None,
        n_jobs=None,
        decision_function_shape="ovr",
    ):
        self.loss = loss
        self.constraint = constraint
        self.C = C
        self.U = U
        self.V = V
        self.Tau = Tau
        self.S = S
        self.T = T
        self.A = A
        self.b = b
        self.max_iter = max_iter
        self.tol = tol
        self.shrink = shrink
        self.warm_start = warm_start
        self.verbose = verbose
        self.trace_freq = trace_freq
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.multi_class = multi_class
        self.n_jobs = n_jobs
        self.decision_function_shape = decision_function_shape


class plq_Ridge_Regressor(RegressorMixin, _SklearnReHLine):
    """
    Empirical Risk Minimization (ERM) regressor with a Piecewise Linear-Quadratic (PLQ) loss
    and a ridge penalty, implemented as a scikit-learn compatible estimator.

    This wrapper adds standard sklearn conveniences while delegating loss/constraint construction
    to :class:`plqERM_Ridge` (via `_make_loss_rehline_param` / `_make_constraint_rehline_param`).

    Notes
    -----
    - **Intercept handling**: if ``fit_intercept=True``, a constant column (value = ``intercept_scaling``)
      is appended to the right of the design matrix before calling the base solver. The last learned
      coefficient is multiplied by ``intercept_scaling`` to obtain ``intercept_``.
      → The column indices of the original features remain; therefore, ``sen_idx`` in the constraint ``fair`` follow the original index.
    - **Constraint handling**: built-in constraints act on original feature coefficients.
      Custom ``A`` accepts ``n_features`` columns, or ``n_features + 1`` columns
      to explicitly constrain the actual intercept.

    Parameters
    ----------
    loss : dict or None, default=None
        None selects {'name': 'QR', 'qt': 0.5} at fit time.
        PLQ loss configuration (e.g., median Quantile Regression). Examples:
        ``{'name': 'QR', 'qt': 0.5}``, ``{'name': 'huber', 'tau': 1.0}``,
        ``{'name': 'SVR', 'epsilon': 0.1}``.
        Required keys depend on the chosen loss and are consumed by the underlying solver.
    constraint : list of dict or None, default=None
        Constraint specifications. Supported by your updated `_make_constraint_rehline_param`:
          - ``{'name': 'nonnegative'}`` or ``{'name': '>=0'}``
          - ``{'name': 'fair', 'sen_idx': list[int], 'tol_sen': list[float]}``
          - ``{'name': 'custom', 'A': ndarray[K, d], 'b': ndarray[K]}``

        Note: when ``fit_intercept=True``, a constant column is appended **as the last column**;
        since you index sensitive columns by ``sen_idx`` on the *original* features, indices stay valid.
    C : float, default=1.0
        Regularization parameter (absorbed by ReHLine parameters inside the solver).
    U, V, S, T, Tau : None or empty array, default=None
        Legacy constructor parameters. Nonempty values raise ValueError at fit
        time because these matrices are generated from ``loss``. Use ReHLine
        or ReHLine_solver for manually specified loss matrices.
    A, b : ndarray or None, default=None
        Additional linear constraints, combined with every entry in ``constraint``.
    max_iter : int, default=1000
        Maximum iterations for the ReHLine solver.
    tol : float, default=1e-4
        Convergence tolerance for the ReHLine solver.
    shrink : int, default=1
        Shrink parameter passed to the solver (see solver docs).
    warm_start : int, default=0
        Warm start flag passed to the solver (see solver docs).
    verbose : int, default=0
        Verbosity for the solver (0: silent).
    trace_freq : int, default=100
        Iteration frequency to trace solver internals (if ``verbose`` is enabled).
    fit_intercept : bool, default=True
        If ``True``, append a constant column (value = ``intercept_scaling``) to the design matrix
        before calling the solver. The learned last coefficient is then split as ``intercept_``.
    intercept_scaling : float, default=1.0
        Scaling applied to the appended constant column when ``fit_intercept=True``.

    Attributes
    ----------
    ``coef_`` : ndarray of shape (n_features,)
        Learned linear coefficients (excluding the intercept term).
    ``intercept_`` : float
        Intercept term extracted from the last coefficient when ``fit_intercept=True``, otherwise 0.0.
    ``n_features_in_`` : int
        Number of input features seen during :meth:`fit` (before intercept augmentation).

    Notes
    -----
    This estimator **does not support sparse input**. If you need sparse support, convert inputs to dense
    or wrap this estimator in a scikit-learn :class:`~sklearn.pipeline.Pipeline` with a transformer that
    densifies data (at the cost of memory).
    """

    def __init__(
        self,
        loss=None,
        constraint=None,
        C=1.0,
        U=None,
        V=None,
        Tau=None,
        S=None,
        T=None,
        A=None,
        b=None,
        max_iter=1000,
        tol=1e-4,
        shrink=1,
        warm_start=0,
        verbose=0,
        trace_freq=100,
        fit_intercept=True,
        intercept_scaling=1.0,
    ):
        self.loss = loss
        self.constraint = constraint
        self.C = C
        self.U = U
        self.V = V
        self.Tau = Tau
        self.S = S
        self.T = T
        self.A = A
        self.b = b
        self.max_iter = max_iter
        self.tol = tol
        self.shrink = shrink
        self.warm_start = warm_start
        self.verbose = verbose
        self.trace_freq = trace_freq
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling


class plq_ElasticNet_Classifier(_ReHLineClassifier):
    """
    Empirical Risk Minimization (ERM) Classifier with a Piecewise Linear-Quadratic (PLQ) loss
    and elastic net penalty, compatible with the scikit-learn API.

    This wrapper makes ``plqERM_ElasticNet`` behave as a classifier:
        - Accepts arbitrary binary labels in the original label space.
        - Computes class weights on original labels (if ``class_weight`` is set).
        - Encodes labels with ``LabelEncoder`` into {0,1}, then maps to {-1,+1} for training.
        - Supports optional intercept fitting (via an augmented constant feature).
        - Provides standard methods ``fit``, ``predict``, and ``decision_function``.
        - Integrates with scikit-learn ecosystem (e.g., GridSearchCV, Pipeline).
        - Supports multiclass classification via OvR or OvO method.

    Parameters
    ----------
    loss : dict
        Dictionary specifying the loss function parameters. Examples include:
        - {'name': 'svm'}
        - {'name': 'sSVM'}
        - {'name': 'huber'}
        and other PLQ losses supported by ``plqERM_ElasticNet``.

    constraint : list of dict or None, default=None
        Optional constraints. Each dictionary must include a ``'name'`` key.

    C : float, default=1.0
        Inverse regularization strength (scales the loss term).

    l1_ratio : float, default=0.5
        The ElasticNet mixing parameter, 0 <= l1_ratio < 1.
        - l1_ratio = 0  → pure Ridge (equivalent to plq_Ridge_Classifier)
        - 0 < l1_ratio < 1 → combined L1 + L2 penalty
        Must be strictly less than 1.0 to avoid division by zero in rho/C_eff.

    omega : array of shape (n_features, ), default=np.empty(shape=(0, 0))
        Non-negative weight coefficients for adaptive lasso. If not provided, all non-intercept coefficients
        receive the same L1 penalty controlled by ``l1_ratio``. The penalty for the intercept
        can be scaled via ``intercept_scaling``.

    fit_intercept : bool, default=True
        Whether to fit an intercept term via an augmented constant feature column.

    intercept_scaling : float, default=1.0
        Value of the constant feature column when ``fit_intercept=True``.

    class_weight : dict, 'balanced', or None, default=None
        'balanced' gives sample i effective weight
        w_i * sum(w) / (n_classes * sum(w[y == y_i])), computed on retained
        original classes before binary/OvR/OvO decomposition. Zero-weight
        rows are excluded. The definition matches LinearSVC >= 1.7 on
        every supported sklearn version. A dict maps original labels to
        multipliers of sample_weight.

    multi_class : str or None, default=None
        Method for multiclass classification:
        - 'ovr': One-vs-Rest
        - 'ovo': One-vs-One
        - None selects OvR; binary problems use one estimator.

    decision_function_shape : {'ovr', 'ovo'}, default='ovr'
        Score output format, independent of the training strategy. 'ovr' returns
        one score per class; 'ovo' requires OvO training and returns one margin
        per sorted class pair, positive for the first class (like SVC). Binary
        output is always 1D, positive for classes_[1]. Changing this parameter
        after fitting does not change predictions or the fitted optimization.

    n_jobs : int or None, default=None
        Number of parallel jobs for multiclass fitting.

    max_iter : int, default=1000
    tol : float, default=1e-4
    shrink : int, default=1
    warm_start : int, default=0
    verbose : int, default=0
    trace_freq : int, default=100

    Attributes
    ----------
    ``coef_`` : ndarray of shape (n_features,) for binary, (n_estimators, n_features) for multiclass
        OvO training coefficients favor the second class of each sorted pair;
        raw multiclass decision scores use the opposite sign, like SVC.
    ``intercept_`` : float for binary, ndarray of shape (n_estimators,) for multiclass
    ``classes_`` : ndarray of shape (n_classes,)
    ``estimators_`` : list, only present for multiclass
    _label_encoder : LabelEncoder
    """

    def __init__(
        self,
        loss,
        constraint=None,
        C=1.0,
        l1_ratio=0.5,
        omega=None,
        U=None,
        V=None,
        Tau=None,
        S=None,
        T=None,
        A=None,
        b=None,
        max_iter=1000,
        tol=1e-4,
        shrink=1,
        warm_start=0,
        verbose=0,
        trace_freq=100,
        fit_intercept=True,
        intercept_scaling=1.0,
        class_weight=None,
        multi_class=None,
        n_jobs=None,
        decision_function_shape="ovr",
    ):
        self.loss = loss
        self.constraint = constraint
        self.C = C
        self.l1_ratio = l1_ratio
        self.omega = omega
        self.U = U
        self.V = V
        self.Tau = Tau
        self.S = S
        self.T = T
        self.A = A
        self.b = b
        self.max_iter = max_iter
        self.tol = tol
        self.shrink = shrink
        self.warm_start = warm_start
        self.verbose = verbose
        self.trace_freq = trace_freq
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.multi_class = multi_class
        self.n_jobs = n_jobs
        self.decision_function_shape = decision_function_shape


class plq_ElasticNet_Regressor(RegressorMixin, _SklearnReHLine):
    """
    Empirical Risk Minimization (ERM) regressor with a Piecewise Linear-Quadratic (PLQ) loss
    and an elastic net penalty, implemented as a scikit-learn compatible estimator.

    This wrapper makes ``plqERM_ElasticNet`` behave as a regressor:
        - Supports optional intercept fitting via an augmented constant feature column.
        - Provides standard methods ``fit``, ``predict``, and ``score``.
        - Integrates with the scikit-learn ecosystem (e.g., GridSearchCV, Pipeline).

    Notes
    -----
    - **Intercept handling**: if ``fit_intercept=True``, a constant column
      (value = ``intercept_scaling``) is appended to the right of the design
      matrix before calling the base solver. The last learned coefficient is
      then multiplied by ``intercept_scaling`` to obtain ``intercept_``.
      Original feature indices are therefore unaffected; ``sen_idx`` in a
      ``'fair'`` constraint continues to reference the original columns.
    - **Sparse input**: not supported. Convert to dense before fitting.

    Parameters
    ----------
    loss : dict or None, default=None
        None selects {'name': 'QR', 'qt': 0.5} at fit time.
        PLQ loss configuration. Examples:
        ``{'name': 'QR', 'qt': 0.5}``, ``{'name': 'huber', 'tau': 1.0}``,
        ``{'name': 'SVR', 'epsilon': 0.1}``.

    constraint : list of dict or None, default=None
        Constraint specifications:
          - ``{'name': 'nonnegative'}`` or ``{'name': '>=0'}``
          - ``{'name': 'fair', 'sen_idx': list[int], 'tol_sen': list[float]}``
          - ``{'name': 'custom', 'A': ndarray[K, d], 'b': ndarray[K]}``

    C : float, default=1.0
        Regularization parameter (scales the loss term).

    l1_ratio : float, default=0.5
        The ElasticNet mixing parameter, 0 <= l1_ratio < 1.
        - l1_ratio = 0  → pure Ridge (equivalent to plq_Ridge_Regressor)
        - 0 < l1_ratio < 1 → combined L1 + L2 penalty
        Must be strictly less than 1.0 to avoid division by zero in rho/C_eff.

    omega : array of shape (n_features, ), default=np.empty(shape=(0, 0))
            Non-negative weight coefficients for adaptive lasso. If not provided, all non-intercept coefficients
            receive the same L1 penalty controlled by ``l1_ratio``. The penalty for the intercept
            can be scaled via ``intercept_scaling``.

    fit_intercept : bool, default=True
        If True, append a constant column (value = ``intercept_scaling``) to
        the design matrix before solving. The last learned coefficient is then
        extracted as ``intercept_``.

    intercept_scaling : float, default=1.0
        Scaling applied to the appended constant column when
        ``fit_intercept=True``.

    max_iter : int, default=1000
    tol : float, default=1e-4
    shrink : int, default=1
    warm_start : int, default=0
    verbose : int, default=0
    trace_freq : int, default=100

    Attributes
    ----------
    ``coef_`` : ndarray of shape (n_features,)
        Learned linear coefficients (excluding the intercept term).
    ``intercept_`` : float
        Intercept term. 0.0 if ``fit_intercept=False``.
    ``n_features_in_`` : int
        Number of input features seen during :meth:`fit` (before intercept
        augmentation).
    """

    def __init__(
        self,
        loss=None,
        constraint=None,
        C=1.0,
        l1_ratio=0.5,
        omega=None,
        U=None,
        V=None,
        Tau=None,
        S=None,
        T=None,
        A=None,
        b=None,
        max_iter=1000,
        tol=1e-4,
        shrink=1,
        warm_start=0,
        verbose=0,
        trace_freq=100,
        fit_intercept=True,
        intercept_scaling=1.0,
    ):
        self.loss = loss
        self.constraint = constraint
        self.C = C
        self.l1_ratio = l1_ratio
        self.omega = omega
        self.U = U
        self.V = V
        self.Tau = Tau
        self.S = S
        self.T = T
        self.A = A
        self.b = b
        self.max_iter = max_iter
        self.tol = tol
        self.shrink = shrink
        self.warm_start = warm_start
        self.verbose = verbose
        self.trace_freq = trace_freq
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
