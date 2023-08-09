##' Solving Regularized Support Vector Machine
##'
##' @description
##' This function solves the regularized support vector machine
##' of the following form:
##' \deqn{
##' \min_{\beta}\ \frac{C}{n}\sum_{i=1}^n \max(1-y_i x_i^T\beta,0) +
##' \frac{1}{2}\Vert\beta\Vert_2^2
##' }
##' where \eqn{\beta\in\mathbb{R}^d} is a length-\eqn{d} vector,
##' \eqn{x_i\in\mathbb{R}^d} is the feature vector for the \eqn{i}-th observation,
##' \eqn{y_i\in\{-1,1\}} is a binary label,
##' and \eqn{C} is the cost parameter.
##'
##' @param x          The data matrix \eqn{X=(x_1,\ldots,x_n)^T} of size
##'                   \eqn{n\times d}, representing \eqn{n} observations
##'                   and \eqn{d} features.
##' @param y          The length-\eqn{n} response vector.
##' @param C          The cost parameter.
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
svm = function(x, y, C = 1, max_iter = 1000, tol = 1e-5, shrink = TRUE, verbose = 0)
{
    n = nrow(x)

    Umat = -C / n * matrix(y, nrow = 1)
    Vmat = matrix(C / n, 1, n)

    res = rehline(x, Umat, Vmat, max_iter = max_iter,
                  tol = tol, shrink = shrink, verbose = verbose)
    res$beta
}
