Tutorials
---------

`ReHLine` is designed to address the regularized ReLU-ReHU minimization problem, named *ReHLine optimization*, of the following form:

.. math::

  \min_{\mathbf{\beta} \in \mathbb{R}^d} \sum_{i=1}^n \sum_{l=1}^L \text{ReLU}( u_{li} \mathbf{x}_i^\intercal \mathbf{\beta} + v_{li}) + \sum_{i=1}^n \sum_{h=1}^H {\text{ReHU}}_{\tau_{hi}}( s_{hi} \mathbf{x}_i^\intercal \mathbf{\beta} + t_{hi}) + \frac{1}{2} \| \mathbf{\beta} \|_2^2, \ \text{ s.t. } \mathbf{A} \mathbf{\beta} + \mathbf{b} \geq \mathbf{0},


where :math:`\mathbf{U} = (u_{li}),\mathbf{V} = (v_{li}) \in \mathbb{R}^{L \times n}` 
and :math:`\mathbf{S} = (s_{hi}),\mathbf{T} = (t_{hi}),\mathbf{\tau} = (\tau_{hi}) \in \mathbb{R}^{H \times n}` 
are the ReLU-ReHU loss parameters, and :math:`(\mathbf{A},\mathbf{b})` are the constraint parameters. 
This formulation has a wide range of applications spanning various fields, including statistics, 
machine learning, computational biology, and social studies. 
Some popular examples include SVMs with fairness constraints (FairSVM), 
elastic net regularized quantile regression (ElasticQR), 
and ridge regularized Huber minimization (RidgeHuber).

See `Manual ReHLine Formulation`_ documentation for more details and examples on converting your problem to ReHLine formulation.


Moreover, the following specific classes of formulations can be directly solved by `ReHLine`.

List of Tutorials
=================

.. list-table::
 :align: left
 :widths: 10 10 20
 :header-rows: 1

 * - tutorials
   - | API
   - | description
 * - `Manual ReHLine Formulation <./tutorials/ReHLine_manual.rst>`_
   - | `ReHLine <./autoapi/rehline/index.html#rehline.ReHLine>`_
   - | ReHLine minimization with manual parameter settings.

 * - `ReHLine: Empirical Risk Minimization <./tutorials/ReHLine_ERM.rst>`_
   - | `plqERM_Ridge <./autoapi/rehline/index.html#rehline.plqERM_Ridge>`_
   - | Empirical Risk Minimization (ERM) with a piecewise linear-quadratic (PLQ) objective with a ridge penalty.

 * - `ReHLine: Scikit-learn Compatible Estimators <./tutorials/ReHLine_sklearn.rst>`_
   - | `plq_Ridge_Classifier <./autoapi/rehline/index.html#rehline.plq_Ridge_Classifier>`_ `plq_Ridge_Regressor <./autoapi/rehline/index.html#rehline.plq_Ridge_Regressor>`_
   - | Scikit-learn compatible estimators framework for empirical risk minimization problem.

 * - `ReHLine: Ridge Composite Quantile Regression <./examples/CQR.ipynb>`_
   - | `CQR_Ridge <./autoapi/rehline/index.html#rehline.CQR_Ridge>`_
   - | Composite Quantile Regression (CQR) with a ridge penalty.

 * - `ReHLine: Matrix Factorization <./tutorials/ReHLine_MF.rst>`_
   - | `plqMF_Ridge <./autoapi/rehline/index.html#rehline.plqMF_Ridge>`_
   - | Matrix Factorization (MF) with a piecewise linear-quadratic (PLQ) objective with a ridge penalty.

.. toctree::
   :maxdepth: 2
   :hidden:

   ./tutorials/ReHLine_manual
   ./tutorials/ReHLine_ERM
   ./tutorials/loss
   ./tutorials/constraint
   ./tutorials/ReHLine_sklearn
   ./tutorials/warmstart
   ./tutorials/ReHLine_MF

