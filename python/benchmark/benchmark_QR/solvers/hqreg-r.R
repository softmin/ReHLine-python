library(hqreg)

hqreg_r = function(x, y, kappa, lam1, lam2, max_iter, tol)
{
    xone = cbind(x, 1)
    res = hqreg_raw(xone, y, method = "quantile", tau = kappa,
        alpha = lam1 / (lam1 + lam2), lambda = c(1, lam1 + lam2),
        intercept = FALSE, max.iter = max_iter, eps = tol)
    as.numeric(res$beta[, 2])
}
