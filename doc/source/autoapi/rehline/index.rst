
rehline
=======

.. py:module:: rehline


Overview
--------

.. list-table:: Classes
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`CQR_Ridge <rehline.CQR_Ridge>`
     - Composite Quantile Regressor (CQR) with a ridge penalty.
   * - :py:obj:`ReHLine <rehline.ReHLine>`
     - ReHLine Minimization.
   * - :py:obj:`plqERM_Ridge <rehline.plqERM_Ridge>`
     - Empirical Risk Minimization (ERM) with a piecewise linear-quadratic (PLQ) objective with a ridge penalty.
   * - :py:obj:`plq_Ridge_Classifier <rehline.plq_Ridge_Classifier>`
     - Empirical Risk Minimization (ERM) Classifier with a Piecewise Linear-Quadratic (PLQ) loss
   * - :py:obj:`plq_Ridge_Regressor <rehline.plq_Ridge_Regressor>`
     - Empirical Risk Minimization (ERM) regressor with a Piecewise Linear-Quadratic (PLQ) loss


.. list-table:: Function
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`ReHLine_solver <rehline.ReHLine_solver>`\ (X, U, V, Tau, S, T, A, b, Lambda, Gamma, xi, max_iter, tol, shrink, verbose, trace_freq)
     - \-
   * - :py:obj:`plqERM_Ridge_path_sol <rehline.plqERM_Ridge_path_sol>`\ (X, y, \*None, loss, constraint, eps, n_Cs, Cs, max_iter, tol, verbose, shrink, warm_start, return_time)
     - Compute the PLQ Empirical Risk Minimization (ERM) path over a range of regularization parameters.



Classes
-------

.. py:class:: CQR_Ridge(quantiles, C=1.0, max_iter=1000, tol=0.0001, shrink=1, warm_start=0, verbose=0, trace_freq=100)

   Bases: :py:obj:`rehline._base._BaseReHLine`, :py:obj:`sklearn.base.BaseEstimator`

   Composite Quantile Regressor (CQR) with a ridge penalty.

   It allows for the fitting of a linear regression model that minimizes a composite quantile loss function.

   .. math::

       \min_{\mathbf{\beta} \in \mathbb{R}^d, \mathbf{\beta_0} \in \mathbb{R}^K} \sum_{k=1}^K \sum_{i=1}^n \text{PLQ}(y_i, \mathbf{x}_i^T \mathbf{\beta} + \mathbf{\beta_0k}) + \frac{1}{2} \| \mathbf{\beta} \|_2^2.


   Parameters
   ----------
   quantiles : list of float (n_quantiles,)
       The quantiles to be estimated.

   C : float, default=1.0
       Regularization parameter. The strength of the regularization is
       inversely proportional to C. Must be strictly positive. 
       `C` will be absorbed by the ReHLine parameters when `self.make_ReLHLoss` is conducted.

   verbose : int, default=0
       Enable verbose output. Note that this setting takes advantage of a
       per-process runtime setting in liblinear that, if enabled, may not work
       properly in a multithreaded context.

   max_iter : int, default=1000
       The maximum number of iterations to be run.

   tol : float, default=1e-4
       The tolerance for the stopping criterion.

   shrink : float, default=1
       The shrinkage of dual variables for the ReHLine algorithm.

   warm_start : bool, default=False
       Whether to use the given dual params as an initial guess for the
       optimization algorithm.

   trace_freq : int, default=100
       The frequency at which to print the optimization trace.
       
   Attributes
   ----------
   coef\_ : array-like
       The optimized model coefficients.

   intercept\_ : array-like
       The optimized model intercepts.

   quantiles\_: array-like
       The quantiles to be estimated.

   n_iter\_ : int
       The number of iterations performed by the ReHLine solver.

   opt_result\_ : object
       The optimization result object.

   dual_obj\_ : array-like
       The dual objective function values.

   primal_obj\_ : array-like
       The primal objective function values.

   Methods
   -------
   fit(X, y, sample_weight=None)
       Fit the model based on the given training data.

   predict(X)
       The prediction for the given dataset.


   Overview
   ========


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`fit <rehline.CQR_Ridge.fit>`\ (X, y, sample_weight)
        - Fit the model based on the given training data.
      * - :py:obj:`predict <rehline.CQR_Ridge.predict>`\ (X)
        - The decision function evaluated on the given dataset


   Members
   =======

   .. py:method:: fit(X, y, sample_weight=None)

      Fit the model based on the given training data.

      Parameters
      ----------

      X: {array-like} of shape (n_samples, n_features)
          Training vector, where `n_samples` is the number of samples and
          `n_features` is the number of features.

      y : array-like of shape (n_samples,)
          The target variable.

      sample_weight : array-like of shape (n_samples,), default=None
          Array of weights that are assigned to individual
          samples. If not provided, then each sample is given unit weight.

      Returns
      -------
      self : object
          An instance of the estimator.




   .. py:method:: predict(X)

      The decision function evaluated on the given dataset

      Parameters
      ----------
      X : array-like of shape (n_samples, n_features)
          The data matrix.

      Returns
      -------
      ndarray of shape (n_samples, n_quantiles)
          Returns the decision function of the samples.




.. py:class:: ReHLine(C=1.0, U=np.empty(shape=(0, 0)), V=np.empty(shape=(0, 0)), Tau=np.empty(shape=(0, 0)), S=np.empty(shape=(0, 0)), T=np.empty(shape=(0, 0)), A=np.empty(shape=(0, 0)), b=np.empty(shape=0), max_iter=1000, tol=0.0001, shrink=1, warm_start=0, verbose=0, trace_freq=100)

   Bases: :py:obj:`rehline._base._BaseReHLine`, :py:obj:`sklearn.base.BaseEstimator`

   ReHLine Minimization.

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

   _U, _V: array of shape (L, n_samples), default=np.empty(shape=(0, 0))
       The parameters pertaining to the ReLU part in the loss function.

   _Tau, _S, _T: array of shape (H, n_samples), default=np.empty(shape=(0, 0))
       The parameters pertaining to the ReHU part in the loss function.

   _A: array of shape (K, n_features), default=np.empty(shape=(0, 0))
       The coefficient matrix in the linear constraint.

   _b: array of shape (K, ), default=np.empty(shape=0)
       The intercept vector in the linear constraint.

   verbose : int, default=0
       Enable verbose output. 

   max_iter : int, default=1000
       The maximum number of iterations to be run.

   tol : float, default=1e-4
       The tolerance for the stopping criterion.

   shrink : float, default=1
       The shrinkage of dual variables for the ReHLine algorithm.

   warm_start : bool, default=False
       Whether to use the given dual params as an initial guess for the
       optimization algorithm.

   trace_freq : int, default=100
       The frequency at which to print the optimization trace.

   Attributes
   ----------
   coef\_ : array-like
       The optimized model coefficients.

   n_iter\_ : int
       The number of iterations performed by the ReHLine solver.

   opt_result\_ : object
       The optimization result object.

   dual_obj\_ : array-like
       The dual objective function values.

   primal_obj\_ : array-like
       The primal objective function values.

   _Lambda: array-like
       The optimized dual variables for ReLU parts.

   _Gamma: array-like
       The optimized dual variables for ReHU parts.

   _xi: array-like
       The optimized dual variables for linear constraints.

   Examples
   --------

   >>> ## test SVM on simulated dataset
   >>> import numpy as np
   >>> from rehline import ReHLine 

   >>> # simulate classification dataset
   >>> n, d, C = 1000, 3, 0.5
   >>> np.random.seed(1024)
   >>> X = np.random.randn(1000, 3)
   >>> beta0 = np.random.randn(3)
   >>> y = np.sign(X.dot(beta0) + np.random.randn(n))

   >>> # Usage of ReHLine
   >>> n, d = X.shape
   >>> U = -(C*y).reshape(1,-1)
   >>> L = U.shape[0]
   >>> V = (C*np.array(np.ones(n))).reshape(1,-1)
   >>> clf = ReHLine(C=C)
   >>> clf._U, clf._V = U, V
   >>> clf.fit(X=X)
   >>> print('sol privided by rehline: %s' %clf.coef_)
   >>> sol privided by rehline: [ 0.7410154  -0.00615574  2.66990408]
   >>> print(clf.decision_function([[.1,.2,.3]]))
   >>> [0.87384162]

   References
   ----------
   .. [1] `Dai, B., Qiu, Y,. (2023). ReHLine: Regularized Composite ReLU-ReHU Loss Minimization with Linear Computation and Linear Convergence <https://openreview.net/pdf?id=3pEBW2UPAD>`_


   Overview
   ========


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`fit <rehline.ReHLine.fit>`\ (X, sample_weight)
        - Fit the model based on the given training data.
      * - :py:obj:`decision_function <rehline.ReHLine.decision_function>`\ (X)
        - The decision function evaluated on the given dataset


   Members
   =======

   .. py:method:: fit(X, sample_weight=None)

      Fit the model based on the given training data.

      Parameters
      ----------

      X: {array-like} of shape (n_samples, n_features)
          Training vector, where `n_samples` is the number of samples and
          `n_features` is the number of features.

      sample_weight : array-like of shape (n_samples,), default=None
          Array of weights that are assigned to individual
          samples. If not provided, then each sample is given unit weight.

      Returns
      -------
      self : object
          An instance of the estimator.


   .. py:method:: decision_function(X)

      The decision function evaluated on the given dataset

      Parameters
      ----------
      X : array-like of shape (n_samples, n_features)
          The data matrix.

      Returns
      -------
      ndarray of shape (n_samples, )
          Returns the decision function of the samples.




.. py:class:: plqERM_Ridge(loss, constraint=[], C=1.0, U=np.empty(shape=(0, 0)), V=np.empty(shape=(0, 0)), Tau=np.empty(shape=(0, 0)), S=np.empty(shape=(0, 0)), T=np.empty(shape=(0, 0)), A=np.empty(shape=(0, 0)), b=np.empty(shape=0), max_iter=1000, tol=0.0001, shrink=1, warm_start=0, verbose=0, trace_freq=100)

   Bases: :py:obj:`rehline._base._BaseReHLine`, :py:obj:`sklearn.base.BaseEstimator`

   Empirical Risk Minimization (ERM) with a piecewise linear-quadratic (PLQ) objective with a ridge penalty.

   .. math::

       \min_{\mathbf{\beta} \in \mathbb{R}^d} \sum_{i=1}^n \text{PLQ}(y_i, \mathbf{x}_i^T \mathbf{\beta}) + \frac{1}{2} \| \mathbf{\beta} \|_2^2, \ \text{ s.t. } \ 
       \mathbf{A} \mathbf{\beta} + \mathbf{b} \geq \mathbf{0},

   The function supports various loss functions, including:
       - 'hinge', 'svm' or 'SVM'
       - 'check' or 'quantile' or 'quantile regression' or 'QR'
       - 'sSVM' or 'smooth SVM' or 'smooth hinge'
       - 'TV'
       - 'huber' or 'Huber'
       - 'SVR' or 'svr'

   The following constraint types are supported:
       * 'nonnegative' or '>=0': A non-negativity constraint.
       * 'fair' or 'fairness': A fairness constraint.
       * 'custom': A custom constraint, where the user must provide the constraint matrix 'A' and vector 'b'.

   Parameters
   ----------
   loss : dict
       A dictionary specifying the loss function parameters. 

   constraint : list of dict
       A list of dictionaries, where each dictionary represents a constraint.
       Each dictionary must contain a 'name' key, which specifies the type of constraint.

   C : float, default=1.0
       Regularization parameter. The strength of the regularization is
       inversely proportional to C. Must be strictly positive. 
       `C` will be absorbed by the ReHLine parameters when `self.make_ReLHLoss` is conducted.

   verbose : int, default=0
       Enable verbose output. Note that this setting takes advantage of a
       per-process runtime setting in liblinear that, if enabled, may not work
       properly in a multithreaded context.

   max_iter : int, default=1000
       The maximum number of iterations to be run.

   _U, _V: array of shape (L, n_samples), default=np.empty(shape=(0, 0))
       The parameters pertaining to the ReLU part in the loss function.

   _Tau, _S, _T: array of shape (H, n_samples), default=np.empty(shape=(0, 0))
       The parameters pertaining to the ReHU part in the loss function.

   _A: array of shape (K, n_features), default=np.empty(shape=(0, 0))
       The coefficient matrix in the linear constraint.

   _b: array of shape (K, ), default=np.empty(shape=0)
       The intercept vector in the linear constraint.

   Attributes
   ----------
   coef\_ : array-like
       The optimized model coefficients.

   n_iter\_ : int
       The number of iterations performed by the ReHLine solver.

   opt_result\_ : object
       The optimization result object.

   dual_obj\_ : array-like
       The dual objective function values.

   primal_obj\_ : array-like
       The primal objective function values.

   Methods
   -------
   fit(X, y, sample_weight=None)
       Fit the model based on the given training data.

   decision_function(X)
       The decision function evaluated on the given dataset.

   Notes
   -----
   The `plqERM_Ridge` class is a subclass of `_BaseReHLine` and `BaseEstimator`, which suggests that it is part of a larger framework for implementing ReHLine algorithms.



   Overview
   ========


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`fit <rehline.plqERM_Ridge.fit>`\ (X, y, sample_weight)
        - Fit the model based on the given training data.
      * - :py:obj:`decision_function <rehline.plqERM_Ridge.decision_function>`\ (X)
        - The decision function evaluated on the given dataset


   Members
   =======

   .. py:method:: fit(X, y, sample_weight=None)

      Fit the model based on the given training data.

      Parameters
      ----------

      X: {array-like} of shape (n_samples, n_features)
          Training vector, where `n_samples` is the number of samples and
          `n_features` is the number of features.

      y : array-like of shape (n_samples,)
          The target variable.

      sample_weight : array-like of shape (n_samples,), default=None
          Array of weights that are assigned to individual
          samples. If not provided, then each sample is given unit weight.

      Returns
      -------
      self : object
          An instance of the estimator.




   .. py:method:: decision_function(X)

      The decision function evaluated on the given dataset

      Parameters
      ----------
      X : array-like of shape (n_samples, n_features)
          The data matrix.

      Returns
      -------
      ndarray of shape (n_samples, )
          Returns the decision function of the samples.




.. py:class:: plq_Ridge_Classifier(loss, constraint=[], C=1.0, U=np.empty((0, 0)), V=np.empty((0, 0)), Tau=np.empty((0, 0)), S=np.empty((0, 0)), T=np.empty((0, 0)), A=np.empty((0, 0)), b=np.empty((0, )), max_iter=1000, tol=0.0001, shrink=1, warm_start=0, verbose=0, trace_freq=100, fit_intercept=True, intercept_scaling=1.0, class_weight=None)

   Bases: :py:obj:`rehline._class.plqERM_Ridge`, :py:obj:`sklearn.base.ClassifierMixin`

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
   coef_ : ndarray of shape (n_features,)
       Coefficients excluding the intercept.

   intercept_ : float
       Intercept term. 0.0 if ``fit_intercept=False``.

   classes_ : ndarray of shape (2,)
       Unique class labels in the original label space.

   _label_encoder : LabelEncoder
       Encodes original labels into {0,1} for internal training.


   Overview
   ========


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`fit <rehline.plq_Ridge_Classifier.fit>`\ (X, y, sample_weight)
        - Fit the classifier to training data.
      * - :py:obj:`decision_function <rehline.plq_Ridge_Classifier.decision_function>`\ (X)
        - Compute the decision function for samples in X.
      * - :py:obj:`predict <rehline.plq_Ridge_Classifier.predict>`\ (X)
        - Predict class labels for samples in X.


   Members
   =======

   .. py:method:: fit(X, y, sample_weight=None)

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


   .. py:method:: decision_function(X)

      Compute the decision function for samples in X.

      Parameters
      ----------
      X : array-like of shape (n_samples, n_features)
          Input samples.

      Returns
      -------
      ndarray of shape (n_samples,)
          Continuous scores for each sample.


   .. py:method:: predict(X)

      Predict class labels for samples in X.

      Parameters
      ----------
      X : array-like of shape (n_samples, n_features)
          Input samples.

      Returns
      -------
      y_pred : ndarray of shape (n_samples,)
          Predicted class labels in the original label space.




.. py:class:: plq_Ridge_Regressor(loss={'name': 'QR', 'qt': 0.5}, constraint=[], C=1.0, U=np.empty((0, 0)), V=np.empty((0, 0)), Tau=np.empty((0, 0)), S=np.empty((0, 0)), T=np.empty((0, 0)), A=np.empty((0, 0)), b=np.empty((0, )), max_iter=1000, tol=0.0001, shrink=1, warm_start=0, verbose=0, trace_freq=100, fit_intercept=True, intercept_scaling=1.0)

   Bases: :py:obj:`rehline._class.plqERM_Ridge`, :py:obj:`sklearn.base.RegressorMixin`

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
   coef_ : ndarray of shape (n_features,)
       Learned linear coefficients (excluding the intercept term).
   intercept_ : float
       Intercept term extracted from the last coefficient when ``fit_intercept=True``, otherwise 0.0.
   n_features_in_ : int
       Number of input features seen during :meth:`fit` (before intercept augmentation).

   Notes
   -----
   This estimator **does not support sparse input**. If you need sparse support, convert inputs to dense
   or wrap this estimator in a scikit-learn :class:`~sklearn.pipeline.Pipeline` with a transformer that
   densifies data (at the cost of memory).


   Overview
   ========


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`fit <rehline.plq_Ridge_Regressor.fit>`\ (X, y, sample_weight)
        - If ``fit_intercept=True``, a constant column (value = ``intercept_scaling``) is appended
      * - :py:obj:`decision_function <rehline.plq_Ridge_Regressor.decision_function>`\ (X)
        - Compute f(X) = X @ coef_ + intercept_.
      * - :py:obj:`predict <rehline.plq_Ridge_Regressor.predict>`\ (X)
        - Predict targets as the linear decision function.


   Members
   =======

   .. py:method:: fit(X, y, sample_weight=None)

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



   .. py:method:: decision_function(X)

      Compute f(X) = X @ coef_ + intercept_.

      Parameters
      ----------
      X : ndarray of shape (n_samples, n_features)
      Input data (dense). Must have the same number of features as seen in :meth:`fit`.

      Returns
      -------
      scores : ndarray of shape (n_samples,)
      Predicted real-valued scores.、


   .. py:method:: predict(X)

      Predict targets as the linear decision function.
      Parameters
      ----------
      X : ndarray of shape (n_samples, n_features)
      Input data (dense).

      Returns
      -------
      y_pred : ndarray of shape (n_samples,)
      Predicted target values (real-valued).




Functions
---------
.. py:function:: ReHLine_solver(X, U, V, Tau=np.empty(shape=(0, 0)), S=np.empty(shape=(0, 0)), T=np.empty(shape=(0, 0)), A=np.empty(shape=(0, 0)), b=np.empty(shape=0), Lambda=np.empty(shape=(0, 0)), Gamma=np.empty(shape=(0, 0)), xi=np.empty(shape=(0, 0)), max_iter=1000, tol=0.0001, shrink=1, verbose=1, trace_freq=100)

.. py:function:: plqERM_Ridge_path_sol(X, y, *, loss, constraint=[], eps=0.001, n_Cs=100, Cs=None, max_iter=5000, tol=0.0001, verbose=0, shrink=1, warm_start=False, return_time=True)

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






