import time

import matplotlib.pyplot as plt
import numpy as np
## load datasets
from sklearn.datasets import make_hastie_10_2

from rehline import plqERM_Ridge_path_sol

X, y = make_hastie_10_2()

# define loss function
loss = {'name': 'svm'}
Cs = np.logspace(-20, 10, 50, base=2)  # Define a range of C values for the path

# calculate with warm_start=False
# Cs_values_cold, times_cold, n_iters_cold, loss_values_cold, L2_norms_cold, coefs_cold = plqERM_Ridge_path_sol(
#     X, y,
#     loss=loss,
#     Cs=Cs,
#     max_iter=5000000,
#     tol=1e-4,
#     verbose=0,
#     warm_start=False,
#     constraint=[],
#     return_time=True,
#     plot_path=True
# )

# calculate with warm_start=True
Cs_values_warm, times_warm, n_iters_warm, loss_values_warm, L2_norms_warm, coefs_warm = plqERM_Ridge_path_sol(
    X, y,
    loss=loss,
    Cs=Cs,
    max_iter=1000000,
    tol=1e-4,
    verbose=1,
    warm_start=True,
    constraint=[],
    return_time=True,
    plot_path=True
)


# # Plot Cs vs times comparison
# plt.figure(figsize=(10, 6))
# plt.plot(Cs, times_warm, 'o-', label='Warm Start')
# plt.plot(Cs, times_cold, 's-', label='Cold Start')
# plt.xscale('log', base=2)
# plt.xlabel('C values')
# plt.ylabel('Time (seconds)')
# plt.title('Computation Time vs. C Parameter')
# plt.legend()
# plt.grid(True)
# plt.show()


# # Print table comparing number of iterations
# print("\nComparison of Number of Iterations:")
# print("-" * 50)
# print(f"{'C Value':^15} | {'Cold Start (iterations)':^20} | {'Warm Start (iterations)':^20}")
# print("-" * 50)
# for i, C in enumerate(Cs):
#     print(f"{C:^15.4f} | {n_iters_cold[i]:^20d} | {n_iters_warm[i]:^20d}")
