## Benchmark repository for SVMs with fairness constraints

The SVM with fairness constraints (FairSVM) solves the following optimization problem:
```math
\begin{align}
  & \min_{\mathbf{\beta} \in \mathbb{R}^d} \frac{C}{n} \sum_{i=1}^n ( 1 - y_i \mathbf{\beta}^\intercal \mathbf{x}_i )_+ + \frac{1}{2} \| \mathbf{\beta} \|_2^2, \nonumber \\
  \text{subj. to } & \quad \frac{1}{n} \sum_{i=1}^n \mathbf{z}_i \mathbf{\beta}^\intercal \mathbf{x}_i \leq \mathbf{\rho}, \quad \frac{1}{n} \sum_{i=1}^n \mathbf{z}_i \mathbf{\beta}^\intercal \mathbf{x}_i \geq -\mathbf{\rho},
\end{align}
```
where $\mathbf{x}_i \in \mathbb{R}^d$ is a feature vector, and $y_i \in \{-1, 1\}$ is a binary label, $\mathbf{z}_i$ is a collection of centered sensitive features

$$\sum_{i=1}^n z_{ij} = 0,$$

such as gender and/or race. The constraints limit the correlation between the $d_0$-length sensitive features $\mathbf{z}_ i \in \mathbb{R}^{d_0}$ and the decision function $\mathbf{\beta}^\intercal \mathbf{x}$, and the constants $\mathbf{\rho} \in \mathbb{R}_+^{d_0}$ trade-offs predictive accuracy and fairness. Note that the FairSVM can be rewritten as a ReHLine optimization with
```math
\mathbf{U} \leftarrow -C \mathbf{y}^\intercal/n, \quad
\mathbf{V} \leftarrow C \mathbf{1}^\intercal_n/n, \quad
\mathbf{A} \leftarrow
\begin{pmatrix}
  \mathbf{Z}^\intercal \mathbf{X} / n \\
  -\mathbf{Z}^\intercal \mathbf{X} / n
  \end{pmatrix}, \quad
\mathbf{b} \leftarrow
\begin{pmatrix}
  \mathbf{\rho} \\
  \mathbf{\rho}
  \end{pmatrix}
```

### Benchmarking solvers

The solvers can be benchmarked using the command below:

```bash
benchopt run . -d classification_data
```
