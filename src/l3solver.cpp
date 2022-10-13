#include <RcppEigen.h>
#include <vector>

using Rcpp::List;
using Rcpp::NumericVector;
using Rcpp::NumericMatrix;

using Matrix = Eigen::MatrixXd;
using MapMat = Eigen::Map<Matrix>;
using Vector = Eigen::VectorXd;
using MapVec = Eigen::Map<Vector>;

// Compute the Q matrix from U array
// U: [n x d] x K
// Q: [n x K]
inline Matrix compute_Q(const std::vector<MapMat>& U)
{
    const int K = U.size();
    const int n = U[0].rows();
    Matrix Q(n, K);
    for(int k = 0; k < K; k++)
    {
        Q.col(k).noalias() = U[k].rowwise().squaredNorm();
    }

    return Q;
}

// Compute the p vector from A
// A: [L x d]
// p: [L]
inline Vector compute_p(const MapMat& A)
{
    const int L = A.rows();
    if(L < 1)
        return Vector::Zero(0);
    return A.rowwise().squaredNorm();
}

inline Vector U_Lambda_prod(
    const std::vector<MapMat>& U, const Matrix& Lambda)
{
    const int K = U.size();
    const int d = U[0].cols();
    Vector res = Vector::Zero(d);
    for(int k = 0; k < K; k++)
    {
        res.noalias() += U[k].transpose() * Lambda.col(k);
    }
    return res;
}

// Initialize result matrices
inline void init_params(
    const std::vector<MapMat>& U, const MapMat& A,
    Matrix& Lambda, Vector& alpha, Vector& beta
)
{
    // Each element of Lambda satisfies 0 <= lambda_ik <= 1,
    // and we use 0.5 to initialize Lambda
    Lambda.fill(0.5);

    /*
    // so we use Unif(0, 1) to initialize Lambda
    const std::size_t nK = Lambda.size();
    double* Lptr = Lambda.data();
    const double* Lptr_end = Lptr + nK;
    for(; Lptr < Lptr_end; Lptr++)
        *Lptr = R::unif_rand();
    */

    // alpha >= 0, initialized to be 1
    const int L = A.rows();
    if(L > 0)
        alpha.fill(1.0);

    // beta = A' * alpha - U(3) * vec(Lambda)
    beta.noalias() = -U_Lambda_prod(U, Lambda);
    if(L > 0)
        beta.noalias() += A.transpose() * alpha;
}

inline void update_Lambda_beta(
    const std::vector<MapMat>& U, const MapMat& V, const Matrix& Q,
    Matrix& Lambda, Vector& beta
)
{
    const int K = U.size();
    const int n = U[0].rows();
    for(int k = 0; k < K; k++)
    {
        for(int i = 0; i < n; i++)
        {
            // Compute epsilon
            double eps = (V(i, k) + U[k].row(i).dot(beta)) / Q(i, k);
            const double lambda_ik = Lambda(i, k);
            eps = std::min(eps, 1.0 - lambda_ik);
            eps = std::max(eps, -lambda_ik);
            // Update Lambda and beta
            Lambda(i, k) += eps;
            beta.noalias() -= eps * U[k].row(i).transpose();
        }
    }
}

inline void update_alpha_beta(
    const MapMat& A, const MapVec& b, const Vector& p,
    Vector& alpha, Vector & beta
)
{
    const int L = A.rows();
    for(int l = 0; l < L; l++)
    {
        // Compute epsilon
        double eps = -(A.row(l).dot(beta) + b[l]) / p[l];
        eps = std::max(eps, -alpha[l]);
        // Update alpha and beta
        alpha[l] += eps;
        beta.noalias() += eps * A.row(l).transpose();
    }
}

inline double objfn(
    const std::vector<MapMat>& U, const MapMat& V,
    const MapMat& A, const MapVec& b,
    const Matrix& Lambda, const Vector& alpha, const Vector& beta
)
{
    Vector Aa = A.transpose() * alpha;
    Vector ULambda = U_Lambda_prod(U, Lambda);
    double obj = 0.5 * (Aa.squaredNorm() + ULambda.squaredNorm()) -
        Aa.dot(ULambda) + alpha.dot(b) - Lambda.cwiseProduct(V).sum();
    return obj;
}

// [[Rcpp::export(l3solver_)]]
List l3solver(List Umat, NumericMatrix Vmat, NumericMatrix Amat, NumericVector bvec,
              int max_iter, double tol, bool verbose = false)
{
    // Get dimensions
    // U: [n x d] x K
    // V: [n x K]
    // A: [L x d]
    // b: [L]
    const int K = Umat.length();
    const int n = Vmat.nrow();
    const int d = Amat.ncol();
    const int L = Amat.nrow();

    // Convert to Eigen matrix objects
    std::vector<MapMat> U;
    for(int k = 0; k < K; k++)
        U.emplace_back(Rcpp::as<MapMat>(Umat[k]));
    MapMat V = Rcpp::as<MapMat>(Vmat);
    MapMat A = Rcpp::as<MapMat>(Amat);
    MapVec b = Rcpp::as<MapVec>(bvec);

    // Store Q matrix and p vector
    Matrix Q = compute_Q(U);
    Vector p = compute_p(A);

    // Create and initialize result matrices
    Matrix Lambda(n, K);
    Vector alpha(L);
    Vector beta(d);
    init_params(U, A, Lambda, alpha, beta);

    // Main iterations
    std::vector<double> objfns;
    int i = 0;
    for(; i < max_iter; i++)
    {
        Vector old_alpha = alpha;
        Vector old_beta = beta;
        update_Lambda_beta(U, V, Q, Lambda, beta);
        update_alpha_beta(A, b, p, alpha, beta);

        // Compute difference of alpha and beta
        const double alpha_diff = (L > 0) ?
                                  (alpha - old_alpha).norm() :
                                  (0.0);
        const double beta_diff = (beta - old_beta).norm();

        // Print progress
        if(verbose && (i % 10 == 0))
        {
            double obj = objfn(U, V, A, b, Lambda, alpha, beta);
            objfns.push_back(obj);
            Rcpp::Rcout << "Iter " << i << ", objfn = " << obj <<
                ", alpha_diff = " << alpha_diff <<
                ", beta_diff = " << beta_diff << std::endl;
        }

        // Convergence test
        if(alpha_diff < tol && beta_diff < tol)
            break;
    }

    return List::create(
        Rcpp::Named("Lambda") = Lambda,
        Rcpp::Named("alpha") = alpha,
        Rcpp::Named("beta") = beta,
        Rcpp::Named("niter") = i,
        Rcpp::Named("objfn") = objfns
    );
}
