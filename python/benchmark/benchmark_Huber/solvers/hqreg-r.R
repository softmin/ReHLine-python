library(hqreg)

hqreg_r = function(x, y, gamma, lam1, lam2, max_iter, tol)
{
    res = hqreg_raw(x, y, method='huber', gamma = gamma,
        alpha = lam1 / (lam1 + lam2), lambda = c(1, lam1 + lam2),
        intercept = FALSE, max.iter = max_iter, eps = tol, screen='none')
    as.numeric(res$beta[, 2])
}
