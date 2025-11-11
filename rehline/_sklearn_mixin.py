import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils._tags import ClassifierTags, RegressorTags

from ._class import plqERM_Ridge



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

    Attributes
    ----------
    coef\_ : ndarray of shape (n_features,)
        Coefficients excluding the intercept.

    intercept\_ : float
        Intercept term. 0.0 if ``fit_intercept=False``.

    classes\_ : ndarray of shape (2,)
        Unique class labels in the original label space.

    _label_encoder : LabelEncoder
        Encodes original labels into {0,1} for internal training.
    """

    def __init__(self, 
                 loss,
                 constraint=[],
                 C=1.,
                 U=np.empty((0, 0)), V=np.empty((0, 0)),
                 Tau=np.empty((0, 0)), S=np.empty((0, 0)), T=np.empty((0, 0)),
                 A=np.empty((0, 0)), b=np.empty((0,)),
                 max_iter=1000, tol=1e-4, shrink=1, warm_start=0,
                 verbose=0, trace_freq=100,
                 fit_intercept=True,
                 intercept_scaling=1.0,
                 class_weight=None):
        
        self.loss = loss
        self.constraint = constraint
        self.C = C
        self._U = U
        self._V = V
        self._S = S
        self._T = T
        self._Tau = Tau
        self._A = A
        self._b = b
        self.L = U.shape[0]
        self.H = S.shape[0]
        self.K = A.shape[0]
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
            X, y,
            accept_sparse=False,
            dtype=np.float64,
            order="C",
        )
        self.n_features_in_ = X.shape[1]

        check_classification_targets(y)

        # Establish classes_ on ORIGINAL labels
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError(
                f"plqERMClassifier currently supports only binary classification, "
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
        y01 = le.transform(y)
        y_pm = 2 * y01 - 1

        # Add constant column for intercept
        X_aug = X
        if self.fit_intercept:
            col = np.full((X.shape[0], 1), self.intercept_scaling, dtype=X.dtype)
            X_aug = np.hstack([X, col])

        super().fit(X_aug, y_pm, sample_weight=sample_weight)

        # Split intercept
        if self.fit_intercept:
            self.intercept_ = float(self.coef_[-1])
            self.coef_ = self.coef_[:-1].copy()
        else:
            self.intercept_ = 0.0

        return self

    def decision_function(self, X):
        """
        Compute the decision function for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Continuous scores for each sample.
        """
        check_is_fitted(self, attributes=["coef_", "intercept_", "_label_encoder", "classes_"])
        X = check_array(X, accept_sparse=False, dtype=np.float64, order="C")
        return X @ self.coef_ + self.intercept_

    def predict(self, X):
        """
        Predict class labels for samples in X.

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
        pred01 = (scores >= 0).astype(int)
        return self._label_encoder.inverse_transform(pred01)

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

    Key behavior
    ------------
    - **Intercept handling**: if ``fit_intercept=True``, a constant column (value = ``intercept_scaling``)
      is appended to the right of the design matrix before calling the base solver. The last learned
      coefficient is then split out as ``intercept_``.
      → The column indices of the original features reamin; therefore, ``sen_idx`` in the constraint ``fair`` follow the original index.
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
    coef\_ : ndarray of shape (n_features,)
        Learned linear coefficients (excluding the intercept term).
    intercept\_ : float
        Intercept term extracted from the last coefficient when ``fit_intercept=True``, otherwise 0.0.
    n_features_in\_ : int
        Number of input features seen during :meth:`fit` (before intercept augmentation).

    Notes
    -----
    This estimator **does not support sparse input**. If you need sparse support, convert inputs to dense
    or wrap this estimator in a scikit-learn :class:`~sklearn.pipeline.Pipeline` with a transformer that
    densifies data (at the cost of memory).
    """

    def __init__(self,
                 loss={'name': 'QR', 'qt': 0.5},
                 constraint=[],
                 C=1.,
                 U=np.empty((0, 0)), V=np.empty((0, 0)),
                 Tau=np.empty((0, 0)), S=np.empty((0, 0)), T=np.empty((0, 0)),
                 A=np.empty((0, 0)), b=np.empty((0,)),
                 max_iter=1000, tol=1e-4, shrink=1, warm_start=0,
                 verbose=0, trace_freq=100,
                 fit_intercept=True,
                 intercept_scaling=1.0):

        self.loss = loss
        self.constraint = constraint
        self.C = C
        self._U = U
        self._V = V
        self._S = S
        self._T = T
        self._Tau = Tau
        self._A = A
        self._b = b
        self.L = U.shape[0]
        self.H = S.shape[0]
        self.K = A.shape[0]
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
        """Compute f(X) = X @ coef\_ + intercept\_.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        Input data (dense). Must have the same number of features as seen in :meth:`fit`.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
        Predicted real-valued scores.、
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
        Estimator tags: regressor, requires y, dense-only.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        Input data (dense).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        Predicted target values (real-valued).
        """

        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        tags.regressor_tags = RegressorTags()
        tags.input_tags.sparse = False
        tags.target_tags.required = True
        return tags
