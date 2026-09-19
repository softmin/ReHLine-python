ReHLine: Empirical Risk Minimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The objective function is given by the following PLQ formulation, where :math:`\phi` is a convex piecewise linear function and :math:`\lambda` is a positive regularization parameter.

.. math::

    \min_{\pmb{\beta} \in \mathbb{R}^d} \sum_{i=1}^n \text{PLQ}(y_i, \mathbf{x}_i^T \pmb{\beta}) + \frac{1}{2} \| \pmb{\beta} \|_2^2, \ \text{ s.t. } \
    \mathbf{A} \pmb{\beta} + \mathbf{b} \geq \mathbf{0},

where :math:`\text{PLQ}(\cdot, \cdot)` is a convex piecewise linear quadratic function, see `Loss <./loss.rst>`_ for built-in loss functions, and :math:`\mathbf{A}` is a :math:`K \times d` matrix, and :math:`\mathbf{b}` is a :math:`K`-dimensional vector for linear constraints, see `Constraints <./constraint.rst>`_ for more details.

For example, it supports the following loss functions and constraints.

.. image:: ../figs/tab.png

Named losses convert numeric targets to ``float64`` before negation or other
loss arithmetic. Integer and floating-point targets representing the same
values therefore define the same objective, subject to float64 precision.
In particular, unsigned regression targets are supported. Original target
arrays are not modified; sklearn classifiers encode class labels separately.

Composite quantile regression
-----------------------------

``CQR_Ridge`` jointly fits a shared slope vector and one intercept per quantile:

.. math::

   \min_{\beta,\alpha}\;
   C \sum_{i=1}^n w_i \sum_{k=1}^q
   \rho_{\tau_k}(y_i-x_i^\top\beta-\alpha_k)
   + \frac12\left(\|\beta\|_2^2+\|\alpha\|_2^2\right),
   \qquad \rho_\tau(r)=\max(\tau r,(\tau-1)r).

The loss is a weighted sum, without division by the number of samples or
quantiles. Both slopes and intercepts are regularized. The quantile order,
including repeated levels, is retained in ``quantiles_`` and prediction columns.

Training evaluates the virtual rows ``[X[i], e_k]`` directly from the original
feature matrix. It avoids a dense ``(n*q, d+q)`` matrix without changing the
joint optimization problem. Feature storage is ``O(n*d)``; loss and dual
arrays still require ``O(n*q)`` memory. This is not a streaming solver.
Prediction returns ``(n_samples, n_quantiles)``.

If fitting raises, CQR and the other public estimators retain their last
successful coefficients, fitted metadata and diagnostics. A failed first fit
remains unfitted. Parameters changed using ``set_params`` are retained and
should be corrected before retrying.

Fitted ``quantiles_`` is an independent copy of the levels used by the last
successful fit. Changing the constructor's ``quantiles`` array takes effect
only after another successful fit, so existing prediction columns keep their
original labels.

For a smaller stored model, ``model.to_inference()`` exports an independent
prediction snapshot with coefficients, intercepts, fitted quantiles and final
objective/convergence diagnostics. It omits loss matrices and dual variables
and supports pickle/joblib serialization. ``fit`` on a snapshot raises
``TypeError``; retain the original model if future refits are needed.

``CQR_Ridge_path_sol(..., compact=True)`` stores these snapshots for every C.
The default ``compact=False`` retains full fitted estimators. Both modes return
the same tuple and coefficient/intercept array shapes. The working estimator
continues to use ``warm_start`` between C values in either mode. Compact stored
results scale with path length and model dimensions instead of retaining
``O(n*q)`` loss/dual storage at every point; a working fit still needs that
training memory. Every snapshot retains its own final objective and diagnostics.

.. code-block:: python

   Cs, snapshots, coefs, intercepts = CQR_Ridge_path_sol(
       X, y, quantiles=[0.2, 0.5, 0.8], Cs=[0.01, 0.1, 1.0],
       compact=True, warm_start=True, return_time=False,
   )
   prediction = snapshots[-1].predict(X_test)
   training_objective = snapshots[-1].objective_

Example
-------

.. nblinkgallery::
   :caption: Empirical Risk Minimization
   :name: rst-link-gallery

   ../examples/QR.ipynb
   ../examples/CQR.ipynb
   ../examples/SVM.ipynb
   ../examples/FairSVM.ipynb
