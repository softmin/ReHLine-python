## Test ElasticNet on simulated dataset
import time
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from rehline import ReHLine, plqERM_ElasticNet, plqERM_Ridge



# simulate classification dataset
n = 1000
C, l1_ratio = 0.00001, 0.5
X, y = make_regression(n_samples=n, n_features=6, noise=0.1, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



# comparison with sklearn
clf = ElasticNet(alpha=1 / (C * 2 * n), 
                 l1_ratio=l1_ratio,
                 max_iter=10000,
                 tol=0.001,
                 fit_intercept=False)

clf.fit(X_scaled, y)
sol_skl = clf.coef_.flatten()


clf = plqERM_ElasticNet(loss={'name': 'mse'}, 
                        C=C,
                        l1_ratio=l1_ratio,
                        max_iter=10000,
                        tol=0.001)

clf.fit(X_scaled, y)
sol_reh = clf.coef_.flatten()
sol_reh = np.where(np.abs(sol_reh) < 1e-8, 0, sol_reh)


print("=" * 60)
print(f"{'Index':^8} {'sklearn':^20} {'rehline':^20} {'diff':^10}")
print("=" * 60)

for i, (s, r) in enumerate(zip(sol_skl, sol_reh)):
    diff = s - r
    print(f"{i:^8d} {s:^20.8f} {r:^20.8f} {diff:^10.2e}")



# warmstart
n = 1000
X, y = make_regression(n_samples=n, n_features=50, noise=0.1, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



Cs = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]
l1_ratio = 0.2


clf = plqERM_ElasticNet(loss={'name': 'mae'}, 
                    C=1.0,
                    l1_ratio=l1_ratio,
                    max_iter=30000,
                    tol=0.0001, warm_start=False)
start = time.perf_counter()
for C_tmp in Cs:
    clf.C = C_tmp
    clf.fit(X_scaled, y)
end = time.perf_counter()
print(f"coldstart: {end - start:.4f} s")




clf = plqERM_ElasticNet(loss={'name': 'mae'}, 
                    C=1.0,
                    l1_ratio=l1_ratio,
                    max_iter=30000,
                    tol=0.0001, warm_start=True)
start = time.perf_counter()
for C_tmp in Cs:
    clf.C = C_tmp
    clf.fit(X_scaled, y)
end = time.perf_counter()
print(f"warmstart: {end - start:.4f} s")
