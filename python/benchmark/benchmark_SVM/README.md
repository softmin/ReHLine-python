## Benchmark repository for SVMs

SVMs solve the following optimization problem:
```math
  \min_{\mathbf{\beta} \in \mathbb{R}^d} \frac{C}{n} \sum_{i=1}^n ( 1 - y_i \mathbf{\beta}^\intercal \mathbf{x}_i )_+ + \frac{1}{2} \| \mathbf{\beta} \|_2^2
```
where $\mathbf{x}_i \in \mathbb{R}^d$ is a feature vector, and $y_i \in \{-1, 1\}$ is a binary label. Note that the SVM can be rewritten as a ReHLine optimization with
```math
\mathbf{U} \leftarrow -C \mathbf{y}^\intercal/n, \quad
\mathbf{V} \leftarrow C \mathbf{1}^\intercal_n/n,
```
where $\mathbf{1}_n = (1, \cdots, 1)^\intercal$ is the $n$-length one vector, $\mathbf{X} \in \mathbb{R}^{n \times d}$ is the feature matrix, and $\mathbf{y} = (y_1, \cdots, y_n)^\intercal$ is the response vector.
### Benchmarking solvers

The solvers can be benchmarked using the command below:

```bash
benchopt run . -d classification_data
```
