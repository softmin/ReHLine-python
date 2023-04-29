elasticqr = function(x, y, kappa = 0.5, lam1 = 0.1, lam2 = 0.1,
    max_iter = 1000, tol = 1e-5, verbose = 0)
{
    n = nrow(x)
    d = ncol(x)
    c1 = kappa / (n * lam2)
    c2 = (1 - kappa) / (n * lam2)
    c3 = lam1 / lam2

    Xmat = rbind(cbind(x, 1), diag(rep(1, d + 1)))
    Ur1 = c(rep(-c1, n), rep(  0, d + 1))
    Ur2 = c(rep( c2, n), rep(  0, d + 1))
    Ur3 = c(rep(  0, n), rep( c3, d + 1))
    Ur4 = c(rep(  0, n), rep(-c3, d + 1))
    Umat = rbind(Ur1, Ur2, Ur3, Ur4)
    Vr1 = c( c1 * y, rep(0, d + 1))
    Vr2 = c(-c2 * y, rep(0, d + 1))
    Vr34 = matrix(0, 2, n + d + 1)
    Vmat = rbind(Vr1, Vr2, Vr34)

    res = rehline(Xmat, Umat, Vmat, max_iter = max_iter,
                  tol = tol, verbose = verbose)
    res$beta
}
