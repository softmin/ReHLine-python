ReHLine: Empirical Risk Minimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The objective function is given by the following PLQ formulation, where :math:`\phi` is a convex piecewise linear function and :math:`\lambda` is a positive regularization parameter.

.. math::

    \min_{\pmb{\beta} \in \mathbb{R}^d} \sum_{i=1}^n \text{PLQ}(y_i, \mathbf{x}_i^T \pmb{\beta}) + \frac{1}{2} \| \pmb{\beta} \|_2^2, \ \text{ s.t. } \ 
    \mathbf{A} \pmb{\beta} + \mathbf{b} \geq \mathbf{0},

where :math:`\text{PLQ}(\cdot, \cdot)` is a convex piecewise linear quadratic function, see `Loss <./loss.rst>`_ for build-in loss functions, and :math:`\mathbf{A}` is a :math:`K \times d` matrix, and :math:`\mathbf{b}` is a :math:`K`-dimensional vector for linear constraints, see `Constraints <./constraint.rst>`_ for more details.

For example, it supports the following loss functions and constraints.

.. image:: ../figs/tab.png

Example
-------

.. nblinkgallery::
   :caption: Emprical Risk Minimization
   :name: rst-link-gallery

   ../examples/QR.ipynb
   ../examples/SVM.ipynb
   ../examples/FairSVM.ipynb
