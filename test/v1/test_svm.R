library(L3solver)
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
U = dat[["U"]]
v = dat[["v"]]

print(dim(U))
print(dim(v))

# Get dimensions
K = dim(U)[1]
n = dim(U)[2]
d = dim(U)[3]

# Reshaping for input
U = lapply(1:K, function(i) U[i, , ])
V = t(v)

set.seed(123)
res = l3solver(U, V, A = NULL, b = NULL, max_iter = 1000, tol = 1e-6, verbose = TRUE)
print(res$beta)
#  [1]  0.40693474  0.26571104  0.04150807  0.95287610  0.78163841  0.05321983 -0.03329368  0.03607190
#  [9]  0.04388921 -0.17519971  0.71490548  0.16579140 -0.17195130 -0.18565053  0.41503904  0.09991364
# [17]  0.20471269  0.23528107  0.14956082 -0.40864135
