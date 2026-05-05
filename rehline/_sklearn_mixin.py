from itertools import combinations

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils._tags import ClassifierTags, RegressorTags
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._class import plqERM_ElasticNet, plqERM_Ridge


class plq_Ridge_Classifier(plqERM_Ridge, ClassifierMixin):
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

    constraint : list of dict, default=[]
        Optional constraints. Each dictionary must include a ``'name'`` key.
        Examples: {'name': 'nonnegative'}, {'name': 'fair'}, {'name': 'custom'}.

    C : float, default=1.0
        Inverse regularization strength.

    _U, _V, _Tau, _S, _T : ndarray, default empty
        Parameters for the PLQ representation of the loss function.
        Typically built internally by helper functions.

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
        Class weights applied like in LinearSVC:
        - 'balanced' uses n_samples / (n_classes * n_j).
        - dict maps label -> weight in the ORIGINAL label space.

    multi_class : str or list, default=[]
        Method for multiclass classification. Options:
        - 'ovo': One-vs-One, trains K*(K-1)/2 binary classifiers.
        - 'ovr': One-vs-Rest, trains K binary classifiers.
        - [ ] or ignored when only 2 classes are present.

    n_jobs : int or None, default=None
        Number of parallel jobs for multiclass fitting.
        None means 1 (serial). -1 means use all available CPUs.
        Passed directly to joblib.Parallel.


    Attributes
    ----------
    ``coef_``: ndarray of shape (n_features,) for binary, (n_estimators, n_features) for multiclass
        Coefficients of all fitted classifiers, excluding the intercept.

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
    ):
        self.loss = loss
        self.constraint = constraint if constraint is not None else []
        self.C = C
        self._U = U if U is not None else np.empty((0, 0))
        self._V = V if V is not None else np.empty((0, 0))
        self._S = S if S is not None else np.empty((0, 0))
        self._T = T if T is not None else np.empty((0, 0))
        self._Tau = Tau if Tau is not None else np.empty((0, 0))
        self._A = A if A is not None else np.empty((0, 0))
        self._b = b if b is not None else np.empty((0,))
        self.L = self._U.shape[0]
        self.H = self._S.shape[0]
        self.K = self._A.shape[0]
        self.max_iter = max_iter
        self.tol = tol
        self.shrink = shrink
        self.warm_start = warm_start
        self.verbose = verbose
        self.trace_freq = trace_freq
        self._Lambda = np.empty((0, 0))
        self._Gamma = np.empty((0, 0))
        self._xi = np.empty((0, 0))
        self.coef_ = None

        self.fit_intercept = fit_intercept
        self.intercept_scaling = float(intercept_scaling)
        self.class_weight = class_weight

        self._label_encoder = None
        self.classes_ = None
        self.multi_class = multi_class if multi_class is not None else []
        self.n_jobs = n_jobs

    @staticmethod
    def _fit_subproblem(estimator, X_aug, y_pm, sample_weight, fit_intercept):
        """
        Train a plqERM_Ridge instance on a single multiclass subproblem.

        Directly constructs plqERM_Ridge from estimator's hyperparameters,
        bypassing plq_Ridge_Classifier.fit() preprocessing (LabelEncoder,
        intercept augmentation) since X_aug and y_pm are already preprocessed.

        Parameters
        ----------
        estimator : plq_Ridge_Classifier
            Source estimator from which hyperparameters are extracted.
            Only used to read parameters, never fitted directly.

        X_aug : ndarray of shape (n_samples, n_features[+1])
            Feature matrix, possibly already augmented with intercept column.
            Passed directly to plqERM_Ridge.fit() without further preprocessing.

        y_pm : ndarray of shape (n_samples,)
            Binary labels already encoded in {-1, +1}.
            Passed directly to plqERM_Ridge.fit() without further preprocessing.

        sample_weight : ndarray of shape (n_samples,) or None
            Per-sample weights.

        fit_intercept : bool
            Whether to extract the last coefficient as intercept.
            Should match estimator.fit_intercept.

        Returns
        -------
        ``coef_``: ndarray of shape (n_features,)
            Fitted coefficients excluding the intercept column.

        ``intercept``: float
            Fitted intercept. 0.0 if fit_intercept is False.
        """

        clf = plqERM_Ridge(
            loss=estimator.loss,
            constraint=estimator.constraint,
            C=estimator.C,
            max_iter=estimator.max_iter,
            tol=estimator.tol,
            shrink=estimator.shrink,
            warm_start=estimator.warm_start,
            verbose=estimator.verbose,
            trace_freq=estimator.trace_freq,
        )
        clf.fit(X_aug, y_pm, sample_weight=sample_weight)
        if fit_intercept:
            coef = clf.coef_[:-1].copy()
            intercept = float(clf.coef_[-1])
        else:
            coef = clf.coef_.copy()
            intercept = 0.0
        return coef, intercept

    def fit(self, X, y, sample_weight=None):
        """
        Fit the classifier to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.

        y : array-like of shape (n_samples,)
            Target labels.

        sample_weight : array-like of shape (n_samples,), default=None
            Per-sample weights. If None, uniform weights are used.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate input (dense only) and set n_features_in_
        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            order="C",
        )
        self.n_features_in_ = X.shape[1]

        check_classification_targets(y)

        # Establish classes_ on ORIGINAL labels
        self.classes_ = np.unique(y)
        if self.classes_.size < 2:
            raise ValueError(
                f"plqERMClassifier requires at least 2 classes, "
                f"but received {self.classes_.size} classes: {self.classes_}."
            )

        # Compute class weights on original labels
        if self.class_weight is not None:
            cw_vec = compute_class_weight(
                class_weight=self.class_weight,
                classes=self.classes_,
                y=y,
            )
            cw_map = {c: w for c, w in zip(self.classes_, cw_vec)}
            sw_cw = np.asarray([cw_map[yi] for yi in y], dtype=np.float64)
            sample_weight = sw_cw if sample_weight is None else (np.asarray(sample_weight) * sw_cw)

        # Encode -> {0,1} -> {-1,+1}
        le = LabelEncoder().fit(self.classes_)
        self._label_encoder = le

        # Add constant column for intercept
        X_aug = X
        if self.fit_intercept:
            col = np.full((X.shape[0], 1), self.intercept_scaling, dtype=X.dtype)
            X_aug = np.hstack([X, col])

        if self.classes_.size == 2:
            y01 = le.transform(y)
            y_pm = 2 * y01 - 1

            super().fit(X_aug, y_pm, sample_weight=sample_weight)

            # Split intercept
            if self.fit_intercept:
                self.intercept_ = float(self.coef_[-1])
                self.coef_ = self.coef_[:-1].copy()
            else:
                self.intercept_ = 0.0

        else:
            # Multiclass classification
            if self.multi_class not in ("ovr", "ovo"):
                raise ValueError(
                    f"multi_class must be 'ovr' or 'ovo' for multiclass problems, got '{self.multi_class}'."
                )
            self._fit_multiclass(X_aug, y, sample_weight)

        return self

    def _fit_multiclass(self, X_aug, y, sample_weight=None):
        """
        Fit multiple binary classifiers for multiclass classification.

        For OvR, trains K binary classifiers (one per class vs. all others).
        For OvO, trains K*(K-1)/2 binary classifiers (one per pair of classes).

        Each binary subproblem is fully independent and dispatched in parallel
        via joblib.Parallel. Results are collected and stacked into ``coef_``
        and ``intercept_`` matrices.

        Parameters
        ----------
        X_aug : ndarray of shape (n_samples, n_features[+1])
            Feature matrix, possibly augmented with intercept column.

        y : ndarray of shape (n_samples,)
            Original (non-encoded) target labels.

        sample_weight : ndarray of shape (n_samples,) or None
            Per-sample weights.
        """
        if self.multi_class == "ovr":
            # Build one task per class: positive=cls, negative=all others
            tasks = [(X_aug, np.where(y == cls, 1, -1).astype(np.float64), sample_weight) for cls in self.classes_]
            class_pairs = None

        elif self.multi_class == "ovo":
            # Build one task per pair of classes
            tasks = []
            class_pairs = []
            for cls_i, cls_j in combinations(self.classes_, 2):
                mask = np.isin(y, [cls_i, cls_j])
                y_pm = np.where(y[mask] == cls_j, 1, -1).astype(np.float64)
                sw_sub = sample_weight[mask] if sample_weight is not None else None
                tasks.append((X_aug[mask], y_pm, sw_sub))
                class_pairs.append((cls_i, cls_j))

        # Dispatch all binary subproblems in parallel
        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self._fit_subproblem)(self, X_sub, y_pm, sw, self.fit_intercept) for X_sub, y_pm, sw in tasks
        )

        # Collect results into estimators_
        if self.multi_class == "ovr":
            self.estimators_ = [(coef, intercept) for coef, intercept in results]
        elif self.multi_class == "ovo":
            self.estimators_ = [
                (coef, intercept, cls_i, cls_j) for (coef, intercept), (cls_i, cls_j) in zip(results, class_pairs)
            ]

        # Stack into matrices for efficient decision_function computation
        # OvR: coef_ shape (K, n_features), intercept_ shape (K,)
        # OvO: coef_ shape (K*(K-1)/2, n_features), intercept_ shape (K*(K-1)/2,)
        self.coef_ = np.array([e[0] for e in self.estimators_])
        self.intercept_ = np.array([e[1] for e in self.estimators_])

    def decision_function(self, X):
        """
        Compute the decision function for samples in X.

        For binary classification, returns a 1D array of scores.
        For OvR multiclass, returns a 2D array of shape (n_samples, K).
        For OvO multiclass, returns a 2D array of shape (n_samples, K*(K-1)/2).

        Using ``coef_.T`` works uniformly for both binary (n_features,) and
        multiclass (n_estimators, n_features) shapes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        ndarray of shape (n_samples,) or (n_samples, n_estimators)
            Continuous scores for each sample.
        """
        check_is_fitted(self, attributes=["coef_", "intercept_", "_label_encoder", "classes_"])
        X = check_array(X, accept_sparse=False, dtype=np.float64, order="C")
        return X @ self.coef_.T + self.intercept_

    def predict(self, X):
        """
        Predict class labels for samples in X.
        For binary classification, thresholds the decision score at 0.
        For OvR, takes the argmax across K classifiers.
        For OvO, uses majority voting across K*(K-1)/2 classifiers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels in the original label space.
        """
        scores = self.decision_function(X)

        if self.classes_.size == 2:
            pred01 = (scores >= 0).astype(int)
            return self._label_encoder.inverse_transform(pred01)

        elif self.multi_class == "ovr":
            # OvR: class with highest decision score wins
            idx = np.argmax(scores, axis=1)
            return self.classes_[idx]

        elif self.multi_class == "ovo":
            # OvO: votes + normalized confidences to break ties
            # Note: score > 0 favors cls_i (first class in pair),
            n_samples = X.shape[0]
            n_classes = len(self.classes_)
            votes = np.zeros((n_samples, n_classes), dtype=np.float64)
            sum_of_confidences = np.zeros((n_samples, n_classes), dtype=np.float64)

            for k, (_, _, cls_i, cls_j) in enumerate(self.estimators_):
                i = np.where(self.classes_ == cls_i)[0][0]
                j = np.where(self.classes_ == cls_j)[0][0]

                # discrete vote: score > 0 favors cls_i, score <= 0 favors cls_j
                pred = (scores[:, k] > 0).astype(int)
                votes[:, j] += pred
                votes[:, i] += 1 - pred

                # continuous confidence: score > 0 means cls_i is more confident
                sum_of_confidences[:, j] += scores[:, k]
                sum_of_confidences[:, i] -= scores[:, k]

            # Monotonically transform to (-1/3, 1/3) to break ties without
            # overriding any decision made by a difference of >= 1 vote
            transformed_confidences = sum_of_confidences / (3 * (np.abs(sum_of_confidences) + 1))

            return self.classes_[np.argmax(votes + transformed_confidences, axis=1)]

    def __sklearn_tags__(self):
        """
        Return scikit-learn estimator tags for compatibility.
        """
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags()
        tags.target_tags.required = True
        tags.input_tags.sparse = False
        return tags


class plq_Ridge_Regressor(plqERM_Ridge, RegressorMixin):
    """
    Empirical Risk Minimization (ERM) regressor with a Piecewise Linear-Quadratic (PLQ) loss
    and a ridge penalty, implemented as a scikit-learn compatible estimator.

    This wrapper adds standard sklearn conveniences while delegating loss/constraint construction
    to :class:`plqERM_Ridge` (via `_make_loss_rehline_param` / `_make_constraint_rehline_param`).

    Notes
    -----
    - **Intercept handling**: if ``fit_intercept=True``, a constant column (value = ``intercept_scaling``)
      is appended to the right of the design matrix before calling the base solver. The last learned
      coefficient is then split out as ``intercept_``.
      → The column indices of the original features remain; therefore, ``sen_idx`` in the constraint ``fair`` follow the original index.
    - **Constraint handling**: constraints are passed through unchanged; the base class will call
      ``_make_constraint_rehline_param(constraint, X, y)`` on the matrix given to `fit`.
      With your updated implementation, ``fair`` must be specified as
      ``{'name': 'fair', 'sen_idx': list[int], 'tol_sen': list[float]}``.

    Parameters
    ----------
    loss : dict, default={'name': 'QR', 'qt': 0.5}
        PLQ loss configuration (e.g., median Quantile Regression). Examples:
        ``{'name': 'QR', 'qt': 0.5}``, ``{'name': 'huber', 'tau': 1.0}``,
        ``{'name': 'SVR', 'epsilon': 0.1}``.
        Required keys depend on the chosen loss and are consumed by the underlying solver.
    constraint : list of dict, default=[]
        Constraint specifications. Supported by your updated `_make_constraint_rehline_param`:
          - ``{'name': 'nonnegative'}`` or ``{'name': '>=0'}``
          - ``{'name': 'fair', 'sen_idx': list[int], 'tol_sen': list[float]}``
          - ``{'name': 'custom', 'A': ndarray[K, d], 'b': ndarray[K]}``

        Note: when ``fit_intercept=True``, a constant column is appended **as the last column**;
        since you index sensitive columns by ``sen_idx`` on the *original* features, indices stay valid.
    C : float, default=1.0
        Regularization parameter (absorbed by ReHLine parameters inside the solver).
    _U, _V, _Tau, _S, _T : ndarray, default empty
        Advanced PLQ parameters for the underlying ReHLine formulation (usually left as defaults).
    _A, _b : ndarray, default empty
        Optional linear constraint matrices (used only if ``constraint`` contains ``{'name': 'custom'}``).
        (Your `_make_constraint_rehline_param` is responsible for validating their shapes.)
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
        self.loss = loss if loss is not None else {"name": "QR", "qt": 0.5}
        self.constraint = constraint if constraint is not None else []
        self.C = C
        self._U = U if U is not None else np.empty((0, 0))
        self._V = V if V is not None else np.empty((0, 0))
        self._S = S if S is not None else np.empty((0, 0))
        self._T = T if T is not None else np.empty((0, 0))
        self._Tau = Tau if Tau is not None else np.empty((0, 0))
        self._A = A if A is not None else np.empty((0, 0))
        self._b = b if b is not None else np.empty((0,))
        self.L = self._U.shape[0]
        self.H = self._S.shape[0]
        self.K = self._A.shape[0]
        self.max_iter = max_iter
        self.tol = tol
        self.shrink = shrink
        self.warm_start = warm_start
        self.verbose = verbose
        self.trace_freq = trace_freq
        self._Lambda = np.empty((0, 0))
        self._Gamma = np.empty((0, 0))
        self._xi = np.empty((0, 0))
        self.coef_ = None

        self.fit_intercept = fit_intercept
        self.intercept_scaling = float(intercept_scaling)

    def fit(self, X, y, sample_weight=None):
        """
        If ``fit_intercept=True``, a constant column (value = ``intercept_scaling``) is appended
        to the **right** of ``X`` before calling the base solver. The base class
        (:class:`plqERM_Ridge`) will construct the loss and constraints via its internal helpers
        on the matrix passed here. After solving, the last coefficient is split as
        ``intercept_`` and removed from ``coef_``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training design matrix (dense). Sparse inputs are not supported.
        y : ndarray of shape (n_samples,)
            Target values.
        sample_weight : ndarray of shape (n_samples,), default=None
            Optional per-sample weights; forwarded to the underlying solver.

        Returns
        -------
        self : object
        Fitted estimator.

        """

        # Dense-only validation
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64, order="C")
        self.n_features_in_ = X.shape[1]

        # Intercept augmentation (append as last column so original indices stay the same)
        X_aug = X
        if self.fit_intercept:
            col = np.full((X.shape[0], 1), self.intercept_scaling, dtype=X.dtype)
            X_aug = np.hstack([X, col])

        super().fit(X_aug, y, sample_weight=sample_weight)

        # Split intercept from coefficients to match sklearn's linear model API
        if self.fit_intercept:
            self.intercept_ = float(self.coef_[-1])
            self.coef_ = self.coef_[:-1].copy()
        else:
            self.intercept_ = 0.0

        return self

    def decision_function(self, X):
        """Compute f(X) = X @ ``coef_`` + ``intercept_``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data (dense). Must have the same number of features as seen in :meth:`fit`.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Predicted real-valued scores.
        """
        check_is_fitted(self, attributes=["coef_", "intercept_"])
        X = check_array(X, accept_sparse=False, dtype=np.float64, order="C")
        return X @ self.coef_ + self.intercept_

    def predict(self, X):
        """
        Predict targets as the linear decision function.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data (dense).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values (real-valued).
        """
        return self.decision_function(X)

    def __sklearn_tags__(self):
        """
        Return scikit-learn estimator tags for compatibility.
        """

        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        tags.regressor_tags = RegressorTags()
        tags.input_tags.sparse = False
        tags.target_tags.required = True
        return tags


class plq_ElasticNet_Classifier(plqERM_ElasticNet, ClassifierMixin):
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

    constraint : list of dict, default=[]
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
        Class weights applied like in LinearSVC.

    multi_class : str or list, default=[]
        Method for multiclass classification:
        - 'ovr': One-vs-Rest
        - 'ovo': One-vs-One
        - [] or ignored when only 2 classes are present.

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
    ):
        if not (0.0 <= l1_ratio < 1.0):
            raise ValueError(
                f"l1_ratio must be in [0, 1), got {l1_ratio}. "
                f"Use l1_ratio=0 for pure Ridge, or plq_Ridge_Classifier directly."
            )

        constraint = [] if constraint is None else constraint
        omega = np.empty((0,)) if omega is None else omega
        U = np.empty((0, 0)) if U is None else U
        V = np.empty((0, 0)) if V is None else V
        Tau = np.empty((0, 0)) if Tau is None else Tau
        S = np.empty((0, 0)) if S is None else S
        T = np.empty((0, 0)) if T is None else T
        A = np.empty((0, 0)) if A is None else A
        b = np.empty((0,)) if b is None else b
        multi_class = [] if multi_class is None else multi_class

        super().__init__(
            loss=loss,
            constraint=constraint,
            C=C,
            l1_ratio=l1_ratio,
            omega=omega,
            U=U,
            V=V,
            Tau=Tau,
            S=S,
            T=T,
            A=A,
            b=b,
            max_iter=max_iter,
            tol=tol,
            shrink=shrink,
            warm_start=warm_start,
            verbose=verbose,
            trace_freq=trace_freq,
        )
        self.fit_intercept = fit_intercept
        self.intercept_scaling = float(intercept_scaling)
        self.class_weight = class_weight
        self._label_encoder = None
        self.classes_ = None
        self.multi_class = multi_class
        self.n_jobs = n_jobs

    @staticmethod
    def _fit_subproblem(estimator, X_aug, y_pm, sample_weight, fit_intercept):
        """
        Train a plqERM_ElasticNet instance on a single multiclass subproblem.

        Directly constructs plqERM_ElasticNet from estimator's hyperparameters,
        bypassing plq_ElasticNet_Classifier.fit() preprocessing (LabelEncoder,
        intercept augmentation) since X_aug and y_pm are already preprocessed.

        Parameters
        ----------
        estimator : plq_ElasticNet_Classifier
            Source estimator from which hyperparameters are extracted.

        X_aug : ndarray of shape (n_samples, n_features[+1])
            Already preprocessed feature matrix (intercept column included if needed).

        y_pm : ndarray of shape (n_samples,)
            Binary labels already in {-1, +1}.

        sample_weight : ndarray or None

        fit_intercept : bool

        Returns
        -------
        coef : ndarray of shape (n_features,)
        intercept : float
        """
        clf = plqERM_ElasticNet(
            loss=estimator.loss,
            constraint=estimator.constraint,
            C=estimator.C,
            l1_ratio=estimator.l1_ratio,
            omega=estimator.omega,
            max_iter=estimator.max_iter,
            tol=estimator.tol,
            shrink=estimator.shrink,
            warm_start=estimator.warm_start,
            verbose=estimator.verbose,
            trace_freq=estimator.trace_freq,
        )
        clf.fit(X_aug, y_pm, sample_weight=sample_weight)
        if fit_intercept:
            coef = clf.coef_[:-1].copy()
            intercept = float(clf.coef_[-1])
        else:
            coef = clf.coef_.copy()
            intercept = 0.0
        return coef, intercept

    def fit(self, X, y, sample_weight=None):
        """
        Fit the classifier to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        sample_weight : array-like of shape (n_samples,), default=None

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64, order="C")
        self.n_features_in_ = X.shape[1]

        check_classification_targets(y)

        self.classes_ = np.unique(y)
        if self.classes_.size < 2:
            raise ValueError(
                f"plq_ElasticNet_Classifier requires at least 2 classes, "
                f"but received {self.classes_.size} class(es): {self.classes_}."
            )

        # Compute class weights on original labels
        if self.class_weight is not None:
            cw_vec = compute_class_weight(
                class_weight=self.class_weight,
                classes=self.classes_,
                y=y,
            )
            cw_map = {c: w for c, w in zip(self.classes_, cw_vec)}
            sw_cw = np.asarray([cw_map[yi] for yi in y], dtype=np.float64)
            sample_weight = sw_cw if sample_weight is None else (np.asarray(sample_weight) * sw_cw)

        le = LabelEncoder().fit(self.classes_)
        self._label_encoder = le

        # Intercept augmentation
        X_aug = X
        omega_copy = self.omega.copy()
        if self.fit_intercept:
            col = np.full((X.shape[0], 1), self.intercept_scaling, dtype=X.dtype)
            X_aug = np.hstack([X, col])
            self.omega = np.append(self.omega, 1) if self.omega.size > 0 else self.omega

        if self.classes_.size == 2:
            y01 = le.transform(y)
            y_pm = 2 * y01 - 1

            # super() resolves to plqERM_ElasticNet.fit()
            super().fit(X_aug, y_pm, sample_weight=sample_weight)
            self.omega = omega_copy
            if self.fit_intercept:
                self.intercept_ = float(self.coef_[-1])
                self.coef_ = self.coef_[:-1].copy()
            else:
                self.intercept_ = 0.0

        else:
            if self.multi_class not in ("ovr", "ovo"):
                raise ValueError(
                    f"multi_class must be 'ovr' or 'ovo' for multiclass problems, got '{self.multi_class}'."
                )
            self._fit_multiclass(X_aug, y, sample_weight)
            self.omega = omega_copy

        return self

    def _fit_multiclass(self, X_aug, y, sample_weight=None):
        """
        Fit multiple binary classifiers for multiclass classification.
        Identical logic to plq_Ridge_Classifier._fit_multiclass; dispatches
        to self._fit_subproblem which uses plqERM_ElasticNet internally.
        """
        if self.multi_class == "ovr":
            tasks = [(X_aug, np.where(y == cls, 1, -1).astype(np.float64), sample_weight) for cls in self.classes_]
            class_pairs = None

        elif self.multi_class == "ovo":
            tasks = []
            class_pairs = []
            for cls_i, cls_j in combinations(self.classes_, 2):
                mask = np.isin(y, [cls_i, cls_j])
                y_pm = np.where(y[mask] == cls_j, 1, -1).astype(np.float64)
                sw_sub = sample_weight[mask] if sample_weight is not None else None
                tasks.append((X_aug[mask], y_pm, sw_sub))
                class_pairs.append((cls_i, cls_j))

        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self._fit_subproblem)(self, X_sub, y_pm, sw, self.fit_intercept) for X_sub, y_pm, sw in tasks
        )

        if self.multi_class == "ovr":
            self.estimators_ = [(coef, intercept) for coef, intercept in results]
        elif self.multi_class == "ovo":
            self.estimators_ = [
                (coef, intercept, cls_i, cls_j) for (coef, intercept), (cls_i, cls_j) in zip(results, class_pairs)
            ]

        self.coef_ = np.array([e[0] for e in self.estimators_])
        self.intercept_ = np.array([e[1] for e in self.estimators_])

    def decision_function(self, X):
        """
        Compute the decision function for samples in X.

        For binary: 1D array of shape (n_samples,).
        For OvR/OvO multiclass: 2D array of shape (n_samples, n_estimators).
        """
        check_is_fitted(self, attributes=["coef_", "intercept_", "_label_encoder", "classes_"])
        X = check_array(X, accept_sparse=False, dtype=np.float64, order="C")
        return X @ self.coef_.T + self.intercept_

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Binary: threshold at 0.
        OvR: argmax across K classifiers.
        OvO: majority vote + normalized confidence tie-breaking.
        """
        scores = self.decision_function(X)

        if self.classes_.size == 2:
            pred01 = (scores >= 0).astype(int)
            return self._label_encoder.inverse_transform(pred01)

        elif self.multi_class == "ovr":
            idx = np.argmax(scores, axis=1)
            return self.classes_[idx]

        elif self.multi_class == "ovo":
            n_samples = X.shape[0]
            n_classes = len(self.classes_)
            votes = np.zeros((n_samples, n_classes), dtype=np.float64)
            sum_of_confidences = np.zeros((n_samples, n_classes), dtype=np.float64)

            for k, (_, _, cls_i, cls_j) in enumerate(self.estimators_):
                i = np.where(self.classes_ == cls_i)[0][0]
                j = np.where(self.classes_ == cls_j)[0][0]

                pred = (scores[:, k] > 0).astype(int)
                votes[:, j] += pred
                votes[:, i] += 1 - pred

                sum_of_confidences[:, j] += scores[:, k]
                sum_of_confidences[:, i] -= scores[:, k]

            transformed_confidences = sum_of_confidences / (3 * (np.abs(sum_of_confidences) + 1))
            return self.classes_[np.argmax(votes + transformed_confidences, axis=1)]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags()
        tags.target_tags.required = True
        tags.input_tags.sparse = False
        return tags


class plq_ElasticNet_Regressor(plqERM_ElasticNet, RegressorMixin):
    """
    Empirical Risk Minimization (ERM) regressor with a Piecewise Linear-Quadratic (PLQ) loss
    and an elastic net penalty, implemented as a scikit-learn compatible estimator.

    This wrapper makes ``plqERM_ElasticNet`` behave as a regressor:
        - Supports optional intercept fitting via an augmented constant feature column.
        - Provides standard methods ``fit``, ``predict``, and ``decision_function``.
        - Integrates with the scikit-learn ecosystem (e.g., GridSearchCV, Pipeline).

    Notes
    -----
    - **Intercept handling**: if ``fit_intercept=True``, a constant column
      (value = ``intercept_scaling``) is appended to the right of the design
      matrix before calling the base solver. The last learned coefficient is
      then split out as ``intercept_``.
      Original feature indices are therefore unaffected; ``sen_idx`` in a
      ``'fair'`` constraint continues to reference the original columns.
    - **Sparse input**: not supported. Convert to dense before fitting.

    Parameters
    ----------
    loss : dict, default={'name': 'QR', 'qt': 0.5}
        PLQ loss configuration. Examples:
        ``{'name': 'QR', 'qt': 0.5}``, ``{'name': 'huber', 'tau': 1.0}``,
        ``{'name': 'SVR', 'epsilon': 0.1}``.

    constraint : list of dict, default=[]
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
        if not (0.0 <= l1_ratio < 1.0):
            raise ValueError(
                f"l1_ratio must be in [0, 1), got {l1_ratio}. "
                f"Use l1_ratio=0 for pure Ridge, or plq_Ridge_Regressor directly."
            )

        loss = {"name": "QR", "qt": 0.5} if loss is None else loss
        constraint = [] if constraint is None else constraint
        omega = np.empty((0,)) if omega is None else omega
        U = np.empty((0, 0)) if U is None else U
        V = np.empty((0, 0)) if V is None else V
        Tau = np.empty((0, 0)) if Tau is None else Tau
        S = np.empty((0, 0)) if S is None else S
        T = np.empty((0, 0)) if T is None else T
        A = np.empty((0, 0)) if A is None else A
        b = np.empty((0,)) if b is None else b

        super().__init__(
            loss=loss,
            constraint=constraint,
            C=C,
            l1_ratio=l1_ratio,
            omega=omega,
            U=U,
            V=V,
            Tau=Tau,
            S=S,
            T=T,
            A=A,
            b=b,
            max_iter=max_iter,
            tol=tol,
            shrink=shrink,
            warm_start=warm_start,
            verbose=verbose,
            trace_freq=trace_freq,
        )
        self.fit_intercept = fit_intercept
        self.intercept_scaling = float(intercept_scaling)

    def fit(self, X, y, sample_weight=None):
        """
        Fit the regressor to training data.

        If ``fit_intercept=True``, a constant column (value =
        ``intercept_scaling``) is appended to the right of ``X`` before
        calling the base solver (``plqERM_ElasticNet.fit``). After solving,
        the last coefficient is split as ``intercept_`` and removed from
        ``coef_``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training design matrix (dense). Sparse inputs are not supported.
        y : ndarray of shape (n_samples,)
            Target values.
        sample_weight : ndarray of shape (n_samples,), default=None
            Optional per-sample weights; forwarded to the underlying solver.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64, order="C")
        self.n_features_in_ = X.shape[1]

        X_aug = X
        omega_copy = self.omega.copy()
        if self.fit_intercept:
            col = np.full((X.shape[0], 1), self.intercept_scaling, dtype=X.dtype)
            X_aug = np.hstack([X, col])
            self.omega = np.append(self.omega, 1) if self.omega.size > 0 else self.omega

        # MRO resolves super() to plqERM_ElasticNet.fit()
        super().fit(X_aug, y, sample_weight=sample_weight)
        self.omega = omega_copy

        if self.fit_intercept:
            self.intercept_ = float(self.coef_[-1])
            self.coef_ = self.coef_[:-1].copy()
        else:
            self.intercept_ = 0.0

        return self

    def decision_function(self, X):
        """
        Compute f(X) = X @ ``coef_`` + ``intercept_``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data (dense).

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Predicted real-valued scores.
        """
        check_is_fitted(self, attributes=["coef_", "intercept_"])
        X = check_array(X, accept_sparse=False, dtype=np.float64, order="C")
        return X @ self.coef_ + self.intercept_

    def predict(self, X):
        """
        Predict target values as the linear decision function.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data (dense).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values (real-valued).
        """
        return self.decision_function(X)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        tags.regressor_tags = RegressorTags()
        tags.input_tags.sparse = False
        tags.target_tags.required = True
        return tags
