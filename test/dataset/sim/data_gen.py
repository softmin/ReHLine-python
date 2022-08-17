## Test dual QP (4) based on simulated dataset 

## simulate datasets
## Example 1: unconstrained QP
import numpy as np
K, n, d = 5, 1000, 10
U1 = np.random.randn(K,n,d)
v1 = np.random.randn(K,n)
np.savez('exp1', U=U1, v=v1)

## Example 1: constrained QP
import numpy as np
K, n, d, L = 5, 1000, 10, 10
U2 = np.random.randn(K,n,d)
v2 = np.random.randn(K,n)
A2 = np.random.randn(L,d)
b2 = np.random.randn(L)

np.savez('exp2', U=U2, v=v2, A=A2, b=b2)