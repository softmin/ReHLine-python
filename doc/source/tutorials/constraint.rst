Constraint
**********

ReHLine allows you to impose various linear constraints on the model coefficients.

Numerical scaling
-----------------

For ``A beta + b >= 0``, multiplying a row of ``A`` and its entry of ``b`` by
the same positive number preserves the feasible set. ReHLine divides each
nonzero row and offset by ``max(abs(A[row]))`` internally. This does not rescale
features or change the objective. ``scaled_constraint_violation_`` reports the
maximum violation in these normalized units, which are used with ``tol`` for
convergence. ``constraint_violation_`` retains the original input units.

The returned constraint multipliers ``xi`` / ``_xi`` also retain the original
units. Estimator warm refits adjust cached multipliers when row scales change.
Direct ``ReHLine_solver`` callers must supply multipliers for the current rows:
multiplying a row by ``s > 0`` divides its equivalent multiplier by ``s``.
An actual all-zero row with a negative offset is infeasible. Tiny nonzero rows
are not treated as zero; computations or returned multipliers outside float64's
representable range raise ``OverflowError``.

Usage Pattern
-------------

Define constraints as a list of dictionaries:

.. code-block:: python

   # list of constraint dictionaries
   constraint = [{'name': <constraint_name>, **kwargs}, ...]


Supported Constraints
---------------------

Non-negative
^^^^^^^^^^^^
Constrains all coefficients to be non-negative (:math:`\beta_j \ge 0`) [1]_.

* **Names**: ``'nonnegative'``, ``'>=0'``
* **Parameters**: None

.. code-block:: python

   constraint = [{'name': '>=0'}]

**Related Example**

.. nblinkgallery::
   :name: nmf-gallery

   ../examples/NMF.ipynb

Fairness
^^^^^^^^
Bounds the **empirical covariance** between the linear score and each sensitive
attribute [2]_. It does not divide by standard deviations, and therefore is not
a bound on the Pearson correlation coefficient.

* **Names**: ``'fair'``, ``'fairness'``
* **Parameters**:
    * ``sen_idx`` (*list of int*): Column indices of sensitive attributes in ``X``.
    * ``tol_sen`` (*list of float*, or a scalar for one attribute): Finite,
      non-negative covariance bounds, one per sensitive attribute.

For the :math:`m` rows used to construct a constraint, let :math:`s_i` be a
sensitive feature and :math:`f_i = x_i^T\beta + a` the linear score. The constraint is

.. math::

   \left|\frac{1}{m}\sum_{i=1}^{m}(s_i-\bar{s})f_i\right|
   = \left|\frac{1}{m}\sum_{i=1}^{m}(s_i-\bar{s})(x_i-\bar{x})^T\beta\right|
   \leq \mathrm{tol\_sen}.

The builder computes the row vector :math:`c=(s-\bar{s})^T(X-\bar{X})/m` and
adds the two inequalities :math:`-c\beta+t\geq0` and :math:`c\beta+t\geq0`.
Temporary centered copies are used **only to compute these constraint
coefficients**. The input X is not modified, and the solver still receives the
original features. This adds no feature standardization or change to the loss,
intercept parameterization or regularization. The second centered factor improves
numerical stability; the constant intercept drops out of covariance.

Before taking the mean, the temporary copy subtracts its first row. Thus an
exactly constant column, including a decimal such as ``0.1``, has exactly zero
covariance even when ``tol_sen=0``. This avoids roundoff creating a spurious
constraint. No threshold discards small nonzero covariance coefficients.

For example, with ``s=[0, 0, 1, 1]`` and scores ``[10, 10, 10, 10]``, the
covariance is zero because both groups have identical scores. The uncentered
product mean would instead be 5. Previously, the builder used this uncentered
moment unless the input sensitive column already had mean zero. Centering fixes
the constraint definition and can therefore change the fitted coefficients and
objective value on uncentered data.

Reference rows and weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Covariance uses equal weight for each included row and denominator ``m`` (not
``m-1``). Positive ``sample_weight`` magnitudes affect the training loss, not
the covariance weights. The reference rows follow each existing fitting API:

* ``plqERM_Ridge`` / ``plqERM_ElasticNet``: all rows passed to ``fit``, including
  rows whose loss weight is zero.
* sklearn classifiers/regressors: rows remaining after zero effective weights
  are removed. Classifier effective weights include ``class_weight``. OvR uses
  the retained training population; each OvO constraint uses only that class
  pair's retained rows and recomputes its mean. Cross-validation uses training
  folds only. In a pipeline, the constraint is expressed in the transformed
  feature coordinates received by the estimator.
* ``plqMF_Ridge``: the current user/item block's observed design rows, including
  zero-loss-weight observations. Truly unobserved blocks cannot define empirical
  fairness statistics and raise an error. The constraint concerns that block's
  linear score; observation-specific offsets from the other factors are not
  included. Changing the opposite factors changes the block design and its
  empirical constraints; the final feasibility diagnostic remains relevant.

These row-selection policies are preserved by the centering fix. To use a
different reference population or weighted covariance, compute the desired
constraint matrix explicitly and pass a ``custom`` constraint or ``A/b``.
A constant sensitive column has zero covariance and imposes no restriction.
Scaling a sensitive feature scales its covariance and changes the interpretation
of ``tol_sen``; the builder does not normalize it to unit variance.

The bound concerns **linear scores**. It does not by itself guarantee equal
positive-prediction rates after thresholding, or fairness of aggregated OvO
votes, calibrated probabilities or predictions on a different population.

.. code-block:: python

   # Example: Constrain fairness w.r.t. feature at index 0 with tolerance 0.01
   constraint = [{'name': 'fair', 'sen_idx': [0], 'tol_sen': [0.01]}]

For a fitted linear model, check the statistic directly on its reference rows:

.. code-block:: python

   sensitive = X_reference[:, [0]]
   centered_sensitive = sensitive - sensitive[0]
   centered_sensitive -= centered_sensitive.mean(axis=0)
   score = X_reference @ coef + intercept
   covariance = centered_sensitive.T @ (score - score.mean()) / len(score)
   assert np.max(np.abs(covariance)) <= 0.01 + solver_tolerance

**Related Example**

.. nblinkgallery::
   :name: fair-gallery

   ../examples/FairSVM.ipynb

Monotonicity
^^^^^^^^^^^^
Constrains coefficients to be monotonically increasing or decreasing [3]_.
Increasing: :math:`\beta_i \le \beta_{i+1}`. Decreasing: :math:`\beta_i \ge \beta_{i+1}`.

* **Names**: ``'monotonic'``, ``'monotonicity'``
* **Parameters**:
    * ``decreasing`` (*bool*, default=False): If ``True``, enforces decreasing monotonicity.

.. code-block:: python

   # Monotonically increasing
   constraint = [{'name': 'monotonic'}]

   # Monotonically decreasing
   constraint = [{'name': 'monotonic', 'decreasing': True}]

**Related Example**

.. nblinkgallery::
   :name: monotonic-gallery

   ../examples/MonotonicSVM.ipynb

Custom Constraints
^^^^^^^^^^^^^^^^^^
Define arbitrary linear constraints of the form :math:`A\beta + b \ge 0`.

* **Names**: ``'custom'``
* **Parameters**:
    * ``A`` (*ndarray*): Coefficient matrix of shape (K, d).
    * ``b`` (*ndarray*): Intercept vector of shape (K,).

.. code-block:: python

   import numpy as np

   # Example: beta_0 + beta_1 >= 1
   A = np.zeros((1, d))
   A[0, 0] = 1
   A[0, 1] = 1
   b = np.array([-1.0])

   constraint = [{'name': 'custom', 'A': A, 'b': b}]

**Related Example**

.. nblinkgallery::
   :name: custom-gallery

   ../examples/CustomQR.ipynb

References
----------

.. [1] `Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. Nature, 401(6755), 788-791. <https://www.nature.com/articles/44565>`_
.. [2] `Zafar, M. B., Valera, I., Gomez Rodriguez, M., & Gummadi, K. P. (2019). Fairness Constraints: A Flexible Approach for Fair Classification. Journal of Machine Learning Research, 20(75), 1-42. <https://www.jmlr.org/papers/v20/18-262.html>`_
.. [3] `Nature Research Intelligence. Monotonicity Constraints in Machine Learning and Classification. <https://www.nature.com/research-intelligence/nri-topic-summaries/monotonicity-constraints-in-machine-learning-and-classification-micro-23773>`_
