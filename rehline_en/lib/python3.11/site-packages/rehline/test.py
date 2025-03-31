from ._path_sol import plqERM_path_sol
import numpy as np
import matplotlib.pyplot as plt
import time

# generate data
np.random.seed(42)
X = np.random.randn(100, 5)
true_beta = np.array([2, -1, 0, 0.5, 0])
y = X @ true_beta + np.random.randn(100) * 0.1

# define loss function
loss = {'name': 'svm'}
Cs = np.logspace(-3, 1, 30)

# calculate
Cs_values, times, n_iters, loss_values, L2_norms, coefs = plqERM_path_sol(
    X, y,
    loss=loss,
    Cs=Cs,
    max_iter=10000,
    tol=1e-4,
    verbose=1,
    warm_start=True,
    constraint=[{'name': 'nonnegative'}],
    return_time=True,
    plot_path=True
)

# # plot the path
# plt.figure(figsize=(8, 6))
# colors = plt.cm.viridis(np.linspace(0, 1, coefs.shape[0]))

# for i, (coef_path, color) in enumerate(zip(coefs, colors)):
#     plt.semilogx(Cs_values, coef_path, linestyle="-", color=color, label=f"Feature {i}")

# plt.xlabel("Regularization parameter (C)")
# plt.ylabel("Coefficients")
# plt.title("PLQ ERM Path of Coefficients (SVM)")
# plt.axvline(x=1, linestyle="--", color="black", label="C=1 Reference")
# plt.legend(loc="upper right", fontsize=8)
# plt.grid(True)
# plt.show()