Constraint
**********

ReHLine allows you to impose various linear constraints on the model coefficients.

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
Constrains the correlation between predictions and sensitive attributes to be within a tolerance [2]_.

* **Names**: ``'fair'``, ``'fairness'``
* **Parameters**:
    * ``sen_idx`` (*list of int*): Column indices of sensitive attributes in ``X``.
    * ``tol_sen`` (*list of float*): Tolerance thresholds for each sensitive attribute.

.. code-block:: python

   # Example: Constrain fairness w.r.t. feature at index 0 with tolerance 0.01
   constraint = [{'name': 'fair', 'sen_idx': [0], 'tol_sen': [0.01]}]

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


References
----------

.. [1] `Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. Nature, 401(6755), 788-791. <https://www.nature.com/articles/44565>`_
.. [2] `Zafar, M. B., Valera, I., Gomez Rodriguez, M., & Gummadi, K. P. (2019). Fairness Constraints: A Flexible Approach for Fair Classification. Journal of Machine Learning Research, 20(75), 1-42. <https://www.jmlr.org/papers/v20/18-262.html>`_
.. [3] `Nature Research Intelligence. Monotonicity Constraints in Machine Learning and Classification. <https://www.nature.com/research-intelligence/nri-topic-summaries/monotonicity-constraints-in-machine-learning-and-classification-micro-23773>`_
