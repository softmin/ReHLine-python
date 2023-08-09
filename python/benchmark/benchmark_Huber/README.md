## Benchmark repository for elastic net regularized Huber minimization

The elastic net regularized Huber minimization (ElasticHuber) solves the following optimization problem:

```math
\min_{\mathbf{\beta}} \frac{1}{n} \sum_{i=1}^n H_\kappa( y_i - \mathbf{x}_i^\intercal \mathbf{\beta} ) + \lambda_1 \| \mathbf{\beta} \|_1 + \frac{\lambda_2}{2} \| \mathbf{\beta} \|_2^2,
```
where $H_\kappa(\cdot)$ is the Huber loss with a given parameter $\kappa$:

```math
\begin{equation*}
  H_\kappa(z) =
  \begin{cases}
  \ z^2/2,                  & 0 < |z| \leq \kappa, \\
  \ \kappa ( |z| - \kappa/2 ),   & |z| > \kappa.
  \end{cases}
\end{equation*}
```
In this case, the ElasticHuber can be rewritten as a ReHLine optimization with
```math
\mathbf{S} \leftarrow
\begin{pmatrix}
-\sqrt{\frac{1}{n\lambda_2}} \mathbf{1}^\intercal_n & \mathbf{0}^\intercal_d \\
\sqrt{\frac{1}{n\lambda_2}} \mathbf{1}^\intercal_n & \mathbf{0}^\intercal_d \\
\end{pmatrix}, \quad
\mathbf{T} \leftarrow
\begin{pmatrix}
  \sqrt{\frac{1}{n\lambda_2}} \mathbf{y}^\intercal & \mathbf{0}^\intercal_d \\
   -\sqrt{\frac{1}{n\lambda_2}} \mathbf{y}^\intercal & \mathbf{0}^\intercal_d \\
  \end{pmatrix}, \quad
\mathbf{\tau} \leftarrow
\begin{pmatrix}
  \kappa \sqrt{\frac{1}{n\lambda_2}} \mathbf{1}^\intercal_n & \mathbf{0}^\intercal_d \\
  \\
  \kappa \sqrt{\frac{1}{n\lambda_2}} \mathbf{1}^\intercal_n  & \mathbf{0}^\intercal_d \\
  \end{pmatrix},
```

```math
\mathbf{U} \leftarrow
\begin{pmatrix}
\mathbf{0}^\intercal_n & \frac{\lambda_1}{\lambda_2} \mathbf{1}_d^\intercal \\
\\
\mathbf{0}^\intercal_n & - \frac{\lambda_1}{\lambda_2} \mathbf{1}_d^\intercal \\
\end{pmatrix}, \quad
\mathbf{V} \leftarrow \mathbf{0}, \quad
\mathbf{X} \leftarrow
\begin{pmatrix}
    \begin{matrix}
      \mathbf{X}
    \end{matrix}
    \\
    \mathbf{I}_{d}
  \end{pmatrix}.
```

### Installation

Assuming the current working directory is the `benchmark_Huber`
folder, some preliminary steps need to be done before
running R solvers. First, enter the Conda environment
and install R:

```bash
conda install r-base rpy2 -c conda-forge
```

Then open R inside this Conda environment, and install necessary packages:

```r
# Install CRAN packages
install.packages(c("hqreg", "Rcpp", "RcppEigen"))

# Install rehline R package
# "../../.." is the relative path to the root folder of ReHLine
# Change this to absolute path if needed
install.packages("../../..", repos = NULL, type = "source")
```

### Benchmarking solvers

The solvers can be benchmarked using the command below:

```bash
benchopt run . -d reg_data
```
