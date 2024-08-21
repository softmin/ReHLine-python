Tutorials
=========

`ReHLine` is designed to address the empirical regularized ReLU-ReHU minimization problem, named *ReHLine optimization*, of the following form:

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

.. image:: ./figs/tab.png

Solving Custom ERMs
-------------------