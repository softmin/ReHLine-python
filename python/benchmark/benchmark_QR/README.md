## Benchmark repository for elastic net regularized quantile regression

The elastic net regularized quantile regression (ElasticQR) solves the following optimization problem:
```math
\min_{\beta \in \mathbb{R}^{d+1}} \frac{1}{n} \sum_{i=1}^n \rho_\kappa ( y_i - x^\intercal_i \beta_{1:d} - \beta_{d+1} ) + \lambda_1 \Vert \beta \Vert_1 + \frac{\lambda_2}{2} \Vert \beta \Vert_2^2,
```
where $\rho_\kappa(u) = u\cdot(\kappa - \mathbf{1}(u < 0))$ is the check loss,
$x_i \in \mathbb{R}^d$ is a feature vector, $y_i \in \mathbb{R}$ is the response variable,
and $\lambda_1, \lambda_2>0$ are weights of lasso and ridge penalties, respectively.

Then, the ElasticQR can be rewritten as a ReHLine optimization with
```math
\mathbf{U} \leftarrow
\begin{pmatrix}
-\frac{\kappa}{n\lambda_2} \mathbf{1}^\intercal_n & \mathbf{0}^\intercal_{d+1} \\
 \frac{1-\kappa}{n\lambda_2} \mathbf{1}^\intercal_n & \mathbf{0}^\intercal_{d+1} \\
 \mathbf{0}^\intercal_n & \frac{\lambda_1}{\lambda_2} \mathbf{1}^\intercal_{d+1} \\
 \mathbf{0}^\intercal_n & -\frac{\lambda_1}{\lambda_2} \mathbf{1}^\intercal_{d+1}
\end{pmatrix}, \quad
\mathbf{V} \leftarrow
\begin{pmatrix}
   \frac{\kappa}{n\lambda_2} \mathbf{y}^\intercal & \mathbf{0}^\intercal_{d+1} \\
   -\frac{1-\kappa}{n\lambda_2} \mathbf{y}^\intercal & \mathbf{0}^\intercal_{d+1} \\
   \mathbf{0}^\intercal_n & \mathbf{0}_{d+1}^\intercal \\
   \mathbf{0}^\intercal_n & \mathbf{0}_{d+1}^\intercal
  \end{pmatrix}, \quad
\mathbf{X} \leftarrow
\begin{pmatrix}
    \begin{matrix}
      \mathbf{X} & \mathbf{1}_n
    \end{matrix}
    \\
    \mathbf{I}_{d+1}
  \end{pmatrix},
```
where $\mathbf{I}_{d+1}$ is an identity matrix.

### Benchmarking solvers

The solvers can be benchmarked using the command below:

```bash
benchopt run . -d reg_data
```
