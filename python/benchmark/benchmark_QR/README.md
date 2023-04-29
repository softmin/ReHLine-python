## Benchmark repository for elastic net regularized quantile regression

The elastic net regularized quantile regression (ElasticQR) solves the following optimization problem:

$$\min_{\beta \in \mathbb{R}^{d+1}} \frac{1}{n} \sum_{i=1}^n \rho_\kappa ( y_i - x^\intercal_i \beta_{1:d} - \beta_{d+1} ) + \lambda_1 \Vert \beta \Vert_1 + \frac{\lambda_2}{2} \Vert \beta \Vert_2^2,$$

where $\rho_\kappa(u) = u\cdot(\kappa - \mathbf{1}(u < 0))$ is the check loss,
$x_i \in \mathbb{R}^d$ is a feature vector, $y_i \in \mathbb{R}$ is the response variable,
and $\lambda_1, \lambda_2>0$ are weights of lasso and ridge penalties, respectively.

### Installation

Assuming the current working directory is the `benchmark_QR`
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
benchopt run . -s rehline -s rehline-r -s hqreg-r -d simulated
```
