Loss
****

ReHLine supports a variety of convex PLQ loss functions for both classification and regression tasks.

Usage Pattern
-------------

Define a loss function using a dictionary:

.. code-block:: python

   # name (str): name of the custom loss function
   # loss_kwargs: more keys and values for loss parameters
   loss = {'name': <loss_name>, **loss_kwargs}


Classification
--------------

SVM (Hinge Loss)
^^^^^^^^^^^^^^^^
Standard Support Vector Machine loss [1]_.

* **Names**: ``'hinge'``, ``'svm'``, ``'SVM'``
* **Parameters**: None

.. code-block:: python

   loss = {'name': 'SVM'}

**Related Example**

.. nblinkgallery::
   :name: svm-gallery

   ../examples/SVM.ipynb
   ../examples/SVM_new.ipynb

Smooth SVM
^^^^^^^^^^
A smoothed version of the Hinge loss (using ReHU) that is differentiable everywhere.

* **Names**: ``'sSVM'``, ``'smooth SVM'``, ``'smooth hinge'``
* **Parameters**: None

.. code-block:: python

   loss = {'name': 'sSVM'}

**Related Example**

.. nblinkgallery::
   :name: ssvm-gallery

   ../examples/SVM.ipynb
   ../examples/Smooth_SVM_new.ipynb

Squared SVM
^^^^^^^^^^^
Squared Hinge loss.

* **Names**: ``'squared SVM'``, ``'squared svm'``, ``'squared hinge'``
* **Parameters**: None

.. code-block:: python

   loss = {'name': 'squared SVM'}

**Related Example**

.. nblinkgallery::
   :name: squared-svm-gallery

   ../examples/SVM.ipynb
   ../examples/Squared_SVM_new.ipynb

Compare SVM Losses with GridSearchCV
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We compare three SVM-type losses: ``SVM``, ``Smooth SVM``, and ``Squared SVM`` with
``GridSearchCV`` and report the best-performing loss together with its optimal hyperparameter.

**Related Example**

.. nblinkgallery::
   :name: compare-svm-gallery

   ../examples/GridSearchCV_SVM_losses.ipynb


Regression
----------

Quantile Regression
^^^^^^^^^^^^^^^^^^^
Minimizes the check loss (pinball loss) for estimating conditional quantiles [2]_.

* **Names**: ``'check'``, ``'quantile'``, ``'QR'``
* **Parameters**:
    * ``qt`` (*float*): The target quantile (e.g., 0.5 for median).

.. code-block:: python

   loss = {'name': 'QR', 'qt': 0.25}

**Related Example**

.. nblinkgallery::
   :name: qr-gallery

   ../examples/QR.ipynb
   ../examples/QR_new.ipynb

Quantile Regression with Epsilon Tolerance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Minimizes the check loss with an epsilon-insensitive zone for robust conditional quantile estimation.

* **Names**: ``'check_eps'``, ``'quantile_eps'``, ``'QR_eps'``
* **Parameters**:
    * ``qt`` (*float*): The target quantile (e.g., 0.5 for median).
    * ``epsilon`` (*float*): The tolerance parameter defining the insensitive zone.

.. code-block:: python

   loss = {'name': 'check_eps', 'qt': 0.25, 'epsilon': 0.1}

**Related Example**

.. nblinkgallery::
   :name: qr-eps-gallery

   ../examples/QR_eps.ipynb
   ../examples/QR_eps_new.ipynb

Huber Regression
^^^^^^^^^^^^^^^^
Robust regression loss that is quadratic for small errors and linear for large errors [3]_.

* **Names**: ``'huber'``, ``'Huber'``
* **Parameters**:
    * ``tau`` (*float*, default=1.0): The threshold parameter controlling the transition from quadratic to linear.

.. code-block:: python

   loss = {'name': 'huber', 'tau': 1.0}

**Related Example**

.. nblinkgallery::
   :name: huber-gallery

   ../examples/Huber_new.ipynb


Support Vector Regression (SVR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Epsilon-insensitive loss [4]_.

* **Names**: ``'SVR'``, ``'svr'``
* **Parameters**:
    * ``epsilon`` (*float*): The epsilon-tube width.

.. code-block:: python

   loss = {'name': 'svr', 'epsilon': 0.1}

**Related Example**

.. nblinkgallery::
   :name: svr-gallery

   ../examples/SVR.ipynb
   ../examples/SVR_new.ipynb

Mean Absolute Error (MAE)
^^^^^^^^^^^^^^^^^^^^^^^^^
L1 loss, robust to outliers.

* **Names**: ``'MAE'``, ``'mae'``, ``'mean absolute error'``
* **Parameters**: None

.. code-block:: python

   loss = {'name': 'mae'}

**Related Example**

.. nblinkgallery::
   :name: mae-gallery

   ../examples/MAE_new.ipynb
   
Mean Squared Error (MSE)
^^^^^^^^^^^^^^^^^^^^^^^^
Standard L2 loss (Least Squares).

* **Names**: ``'MSE'``, ``'mse'``, ``'mean squared error'``
* **Parameters**: None

.. code-block:: python

   loss = {'name': 'mse'}

**Related Example**

.. nblinkgallery::
   :name: mse-gallery

   ../examples/MSE_new.ipynb

Compare Regression Losses with GridSearchCV
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We compare three regression losses: ``Huber``, ``MAE``, and ``MSE`` with 
``GridSearchCV`` and report the best-performing loss together with its optimal hyperparameter.

**Related Example**

.. nblinkgallery::
   :name: compare-reg-gallery

   ../examples/GridSearchCV_reg_losses.ipynb


References
----------

.. [1] `Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine learning, 20(3), 273-297. <https://link.springer.com/article/10.1007/BF00994018>`_
.. [2] `Koenker, R., & Bassett Jr, G. (1978). Regression quantiles. Econometrica: journal of the Econometric Society, 33-50. <https://www.jstor.org/stable/1913643>`_
.. [3] `Huber, P. J. (1964). Robust estimation of a location parameter. The Annals of Mathematical Statistics, 35(1), 73-101. <https://projecteuclid.org/euclid.aoms/1177703732>`_
.. [4] `Drucker, H., Burges, C. J., Kaufman, L., Smola, A., & Vapnik, V. (1997). Support vector regression machines. Advances in neural information processing systems, 9. <https://proceedings.neurips.cc/paper/1996/file/d38901788c533e8286cb6400b40b386d-Paper.pdf>`_
