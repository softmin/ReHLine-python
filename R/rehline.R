rehline = function(
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

    rehline_(
        Xmat, Amat, bvec, Umat, Vmat, Smat, Tmat, Tau,
        max_iter, tol, verbose
    )
}

rehline2 = function(
    Xmat, Umat, Vmat, Smat = NULL, Tmat = NULL, Tau = Inf,
    Amat = NULL, bvec = NULL,
    max_iter = 1000, tol = 1e-5, verbose = 0)
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

    rehline2_(
        Xmat, Amat, bvec, Umat, Vmat, Smat, Tmat, Tau,
        max_iter, tol, verbose
    )
}
