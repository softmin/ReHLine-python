##' Solving Elastic Net Regularized Quantile Regression
##'
##' @description
##' This function solves the elastic net regularized quantile regression
##' (ElasticQR) of the following form:
##' \deqn{
##' \min_{\beta}\ \frac{1}{n}\sum_{i=1}^n \rho_\kappa(y_i-x_i^T\beta_{1:d}-\beta_{d+1})+
##' \lambda_1\Vert\beta\Vert_1 + \frac{\lambda_2}{2}\Vert\beta\Vert_2^2
##' }
##' where \eqn{\rho_\kappa(u)=u\cdot(\kappa-I(u<0))} is the check loss,
##' \eqn{\beta\in\mathbb{R}^{d+1}} is a length-\eqn{(d+1)} vector,
##' \eqn{x_i\in\mathbb{R}^d} is the feature vector for the \eqn{i}-th observation,
##' \eqn{y_i\in\mathbb{R}} is the \eqn{i}-th response variable value,
##' and \eqn{\lambda_1,\lambda_2>0} are weights of lasso and ridge penalties,
##' respectively.
##'
##' @param x          The data matrix \eqn{X=(x_1,\ldots,x_n)^T} of size
##'                   \eqn{n\times d}, representing \eqn{n} observations
##'                   and \eqn{d} features.
##' @param y          The length-\eqn{n} response vector.
##' @param kappa      Parameter of the check loss.
##' @param lam1,lam2  Weights of lasso and ridge penalties, respectively.
##' @param max_iter   Maximum number of iterations.
##' @param tol        Tolerance parameter for convergence test.
##' @param shrink     Whether to use the shrinkage algorithm.
##' @param verbose    Level of verbosity.
##'
##' @return A list of the following components:
##' \item{beta}{Optimized value of the \eqn{\beta} vector.}
##' \item{xi,Lambda,Gamma}{Values of dual variables.}
##' \item{niter}{Number of iterations used.}
##' \item{dual_objfns}{Dual objective function values during the optimization process.}
##'
##' @author Yixuan Qiu \url{https://statr.me}
##'
##'         Ben Dai \url{https://bendai.org}
##'
elastic_qr = function(x, y, kappa = 0.5, lam1 = 0.1, lam2 = 0.1,
    max_iter = 1000, tol = 1e-5, shrink = TRUE, verbose = 0)
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
                  tol = tol, shrink = shrink, verbose = verbose)
    res$beta
}
