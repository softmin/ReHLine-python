library(rehline)
library(reticulate)

py_config()
np = import("numpy")

# Utility function to read .npz files
read_npz = function(npz_file)
{
    npz = np$load(npz_file)
    files = npz$files
    nfiles = length(files)

    res = vector("list", nfiles)
    for(i in seq_along(files))
    {
        res[[i]] = npz$f[[files[i]]]
    }
    names(res) = files

    res
}

dat = read_npz("./dataset/sim/exp_svm.npz")
X = dat[["X"]]
U = dat[["U"]]
V = dat[["V"]]

print(dim(X))
print(dim(U))
print(dim(V))

set.seed(123)
res = rehline(X, U, V, max_iter = 1000, tol = 1e-6, verbose = 1)
print(res$beta)
#  [1]  0.40693474  0.26571104  0.04150807  0.95287610  0.78163841  0.05321983 -0.03329368  0.03607190
#  [9]  0.04388921 -0.17519971  0.71490548  0.16579140 -0.17195130 -0.18565053  0.41503904  0.09991364
# [17]  0.20471269  0.23528107  0.14956082 -0.40864135

# Add constraints
set.seed(123)
K = 5
d = ncol(X)
A = matrix(rnorm(K * d), K)
b = rnorm(K)
res = rehline(
    X, U, V, Amat = A, bvec = b, max_iter = 1000,
    tol = 1e-6, verbose = 1
)
# Test whether A * beta + b >= 0
print(c(A %*% res$beta) + b)
