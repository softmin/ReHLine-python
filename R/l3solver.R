l3solver = function(
    Xmat, Umat, Vmat, Smat = NULL, Tmat = NULL, Tau = Inf,
    Amat = NULL, bvec = NULL,
    max_iter = 1000, tol = 1e-5, verbose = FALSE)
{
    n = nrow(Xmat)
    d = ncol(Xmat)

    if(is.null(Umat) || is.null(Vmat))
    {
        Umat = Vmat = matrix(0, 0, n)
    }
    if(is.null(Smat) || is.null(Tmat))
    {
        Smat = Tmat = matrix(0, 0, n)
    }
    if(is.null(Amat) || is.null(bvec))
    {
        Amat = matrix(0, 0, d)
        bvec = numeric(0)
    }

    # Expand Tau to a matrix
    if(length(Tau) == 1)
        Tau = matrix(Tau, nrow(Tmat), ncol(Tmat))

    l3solver_(
        Xmat, Amat, bvec, Umat, Vmat, Smat, Tmat, Tau,
        max_iter, tol, verbose
    )
}

l3cd = function(Umat, Vmat, Amat = NULL, bvec = NULL, max_iter = 1000, tol = 1e-5, verbose = FALSE)
{
    K = length(Umat)
    n = nrow(Umat[[1]])
    d = ncol(Umat[[1]])
    if(is.null(Amat) || is.null(bvec))
    {
        Amat = matrix(0, 0, d)
        bvec = numeric(0)
    }

    l3cd_(Umat, Vmat, Amat, bvec, max_iter, tol, verbose)
}
