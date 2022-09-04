l3solver = function(Umat, Vmat, Amat = NULL, bvec = NULL, max_iter = 1000, tol = 1e-5, verbose = FALSE)
{
    K = length(Umat)
    n = nrow(Umat[[1]])
    d = ncol(Umat[[1]])
    if(is.null(Amat) || is.null(bvec))
    {
        Amat = matrix(0, 0, d)
        bvec = numeric(0)
    }

    l3solver_(Umat, Vmat, Amat, bvec, max_iter, tol, verbose)
}
