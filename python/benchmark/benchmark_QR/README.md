## Benchmark repository for elastic net regularized quantile regression

The elastic net regularized quantile regression (ElasticQR) solves the following optimization problem:

$$\min_{\beta \in \mathbb{R}^{d+1}} \frac{1}{n} \sum_{i=1}^n \rho_\kappa ( y_i - x^\intercal_i \beta_{1:d} - \beta_{d+1} ) + \lambda_1 \Vert \beta \Vert_1 + \frac{\lambda_2}{2} \Vert \beta \Vert_2^2,$$

where $\rho_\kappa(u) = u\cdot(\kappa - \mathbf{1}(u < 0))$ is the check loss,
$x_i \in \mathbb{R}^d$ is a feature vector, $y_i \in \mathbb{R}$ is the response variable,
and $\lambda_1, \lambda_2>0$ are weights of lasso and ridge penalties, respectively.
