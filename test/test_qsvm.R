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

dat = read_npz("./dataset/sim/exp_qsvm.npz")
X = dat[["X"]]
Smat = dat[["S"]]
Tmat = dat[["T"]]
tau = dat[["tau"]]

print(dim(X))
print(dim(Smat))
print(dim(Tmat))

set.seed(123)
res = rehline(
    X, Umat = NULL, V = NULL, Smat = Smat, Tmat = Tmat, Tau = tau,
    max_iter = 1000, tol = 1e-6, verbose = 1)
print(res$beta)
#  [1]  0.326476260  0.192968686 -0.019646593  0.630737002  0.504404859
#  [6] -0.026382578  0.015123459 -0.005368009  0.135625386 -0.194919709
# [11]  0.480970710  0.166156756 -0.073218441 -0.007768079  0.213031604
# [16]  0.038335470  0.180552815  0.141290067  0.192858360 -0.300795854

# Add constraints
set.seed(123)
K = 5
d = ncol(X)
A = matrix(rnorm(K * d), K)
b = rnorm(K)
res = rehline(
    X, Umat = NULL, V = NULL, Smat = Smat, Tmat = Tmat, Tau = tau,
    Amat = A, bvec = b,
    max_iter = 1000, tol = 1e-6, verbose = 1
)
# Test whether A * beta + b >= 0
print(c(A %*% res$beta) + b)
