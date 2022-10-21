#include <vector>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using Matrix = Eigen::MatrixXd;
// using MapMat = Eigen::Map<Matrix>;
using MapMat = py::EigenDRef<Matrix>;
using Vector = Eigen::VectorXd;
// using MapVec = Eigen::Map<Vector>;
using MapVec = Eigen::Ref<Vector>;

// Dimensions of the matrices involved
// - Input
//   * X   : [n x d]
//   * U, V: [L x n]
//   * S, T: [H x n]
//   * A   : [K x d]
//   * b   : [K]
// - Pre-computed
//   * r: [n]
//   * p: [K]
// - Primal
//   * beta: [d]
// - Dual
//   * xi    : [K]
//   * Lambda: [L x n]
//   * Gamma : [H x n]
//   * Omega : [H x n]

// Pre-compute the r vector from X
inline Vector precompute_r(const MapMat& X)
{
    return X.rowwise().squaredNorm();
}

// Pre-compute the p vector from A
// A [K x d], K can be zero
inline Vector precompute_p(const MapMat& A)
{
    const int K = A.rows();
    if(K < 1)
        return Vector::Zero(0);
    return A.rowwise().squaredNorm();
}

// Compute the primal variable beta from dual variables
inline Vector get_primal(
    const MapMat& X, const MapMat& A, const MapMat& U, const MapMat& S,
    const Vector& xi, const Matrix& Lambda, const Matrix& Gamma
)
{
    // Get dimensions
    const int n = X.rows();
    const int d = X.cols();
    const int L = U.rows();
    const int H = S.rows();
    const int K = A.rows();

    Vector beta = Vector::Zero(d);
    if (K > 0)
        beta.noalias() = A.transpose() * xi;

    // [n x 1]
    Vector LHterm = Vector::Zero(n);
    if (L > 0)
        LHterm.noalias() = U.cwiseProduct(Lambda).colwise().sum().transpose();
    // [n x 1]
    if (H > 0)
        LHterm.noalias() += S.cwiseProduct(Gamma).colwise().sum().transpose();

    beta.noalias() -= X.transpose() * LHterm;
    return beta;
}

// Initialize result matrices
inline void init_params(
    const MapMat& X, const MapMat& A,
    const MapMat& U, const MapMat& S, double tau,
    Vector& xi, Matrix& Lambda, Matrix& Gamma, Matrix& Omega, Vector& beta
)
{
    // Get dimensions
    const int L = U.rows();
    const int H = S.rows();
    const int K = A.rows();

    // xi >= 0, initialized to be 1
    if (K > 0)
        xi.fill(1.0);

    // Each element of Lambda satisfies 0 <= lambda_li <= 1,
    // and we use 0.5 to initialize Lambda
    if (L > 0)
        Lambda.fill(0.5);

    // Each element of Gamma satisfies 0 <= gamma_hi <= tau,
    // and we use min(0.5 * tau, 1) to initialize (tau can be Inf)
    // Each element of Omega satisfies omega_hi >= 0, initialized to be 1
    if (H > 0)
    {
        Gamma.fill(std::min(1.0, 0.5 * tau));
        Omega.fill(1.0);
    }

    beta = get_primal(X, A, U, S, xi, Lambda, Gamma);
}

// Update Lambda and beta
inline void update_Lambda_beta(
    const MapMat& X, const MapMat& U, const MapMat& V, const Vector& r,
    Matrix& Lambda, Vector& beta
)
{
    const int n = X.rows();
    const int L = U.rows();
    for(int l = 0; l < L; l++)
    {
        for(int i = 0; i < n; i++)
        {
            // Compute epsilon
            const double u_li = U(l, i);
            double eps = (V(l, i) + u_li * X.row(i).dot(beta)) / r[i] / u_li / u_li;
            const double lambda_li = Lambda(l, i);
            eps = std::min(eps, 1.0 - lambda_li);
            eps = std::max(eps, -lambda_li);
            // Update Lambda and beta
            Lambda(l, i) += eps;
            beta.noalias() -= eps * u_li * X.row(i).transpose();
        }
    }
}

// Update Gamma, Omega, and beta
inline void update_Gamma_Omega_beta(
    const MapMat& X, const MapMat& S, const MapMat& T,
    double tau, const Vector& r,
    Matrix& Gamma, Matrix& Omega, Vector& beta
)
{
    const int n = X.rows();
    const int H = S.rows();
    for(int h = 0; h < H; h++)
    {
        for(int i = 0; i < n; i++)
        {
            // Compute epsilon
            const double s_hi = S(h, i);
            const double gamma_hi = Gamma(h, i);
            double eps = T(h, i) + Omega(h, i) +
                s_hi * X.row(i).dot(beta) - gamma_hi;
            eps = eps / (s_hi * s_hi * r[i] + 1.0);
            eps = std::min(eps, tau - gamma_hi);
            eps = std::max(eps, -gamma_hi);
            // Update Gamma, Omega, and beta
            Gamma(h, i) += eps;
            beta.noalias() -= eps * s_hi * X.row(i).transpose();
            Omega(h, i) = std::max(0.0, gamma_hi + eps - tau);
        }
    }
}

// Update xi and beta
inline void update_xi_beta(
    const MapMat& A, const MapVec& b, const Vector& p,
    Vector& xi, Vector & beta
)
{
    const int K = A.rows();
    for(int k = 0; k < K; k++)
    {
        // Compute epsilon
        double eps = -(A.row(k).dot(beta) + b[k]) / p[k];
        eps = std::max(eps, -xi[k]);
        // Update xi and beta
        xi[k] += eps;
        beta.noalias() += eps * A.row(k).transpose();
    }
}

// Compute the dual objective function value
inline double dual_objfn(
    const MapMat& X, const MapMat& A, const MapVec& b,
    const MapMat& U, const MapMat& V,
    const MapMat& S, const MapMat& T,
    const Vector& xi, const Matrix& Lambda,
    const Matrix& Gamma, const Matrix& Omega, double tau
)
{
    // Get dimensions
    const int n = X.rows();
    const int d = X.cols();
    const int L = U.rows();
    const int H = S.rows();
    const int K = A.rows();

    // A' * xi, [d x 1], A[K x d] may be empty
    Vector Atxi = Vector::Zero(d);
    if (K > 0)
        Atxi.noalias() = A.transpose() * xi;
    // U3 * vec(Lambda), [n x 1], U[L x n] may be empty
    Vector UL(n), U3L = Vector::Zero(d);
    if (L > 0)
    {
        UL.noalias() = U.cwiseProduct(Lambda).colwise().sum().transpose();
        U3L.noalias() = X.transpose() * UL;
    }
    // S3 * vec(Gamma), [n x 1], S[H x n] may be empty
    Vector SG(n), S3G = Vector::Zero(d);
    if (H > 0)
    {
        SG.noalias() = S.cwiseProduct(Gamma).colwise().sum().transpose();
        S3G.noalias() = X.transpose() * SG;
    }

    // Compute dual objective function value
    double obj = 0.0;
    // If K = 0, all terms that depend on A, xi, or b will be zero
    if (K > 0)
    {
        // 0.5 * ||Atxi||^2 - Atxi' * U3L - Atxi' * S3G + xi' * b
        const double Atxi_U3L = (L > 0) ? (Atxi.dot(U3L)) : 0.0;
        const double Atxi_S3G = (H > 0) ? (Atxi.dot(S3G)) : 0.0;
        obj += 0.5 * Atxi.squaredNorm() - Atxi_U3L - Atxi_S3G + xi.dot(b);
    }
    // If L = 0, all terms that depend on U, V, or Lambda will be zero
    if (L > 0)
    {
        // 0.5 * ||U3L||^2 + U3L' * S3G - tr(Lambda * V')
        const double U3L_S3G = (H > 0) ? (U3L.dot(S3G)) : 0.0;
        obj += 0.5 * U3L.squaredNorm() + U3L_S3G -
            Lambda.cwiseProduct(V).sum();
    }
    // If H = 0, all terms that depend on S, T, Gamma, or Omega will be zero
    // Also note that if tau = Inf, then Omega = 0
    if (H > 0)
    {
        // 0.5 * ||Omega||^2 + 0.5 * ||S3G||^2 + 0.5 * ||Gamma||^2
        // - tr(Gamma * Omega') - tr(Gamma * T') + tau * sum(Omega)
        if (std::isinf(tau))
            obj += 0.5 * S3G.squaredNorm() + 0.5 * Gamma.squaredNorm() -
                Gamma.cwiseProduct(T).sum();
        else
            obj += 0.5 * Omega.squaredNorm() + 0.5 * S3G.squaredNorm() +
                0.5 * Gamma.squaredNorm() - Gamma.cwiseProduct(Omega + T).sum() +
                tau * Omega.sum();
    }

    return obj;
}

struct L3Result
{
    Vector              beta;
    Vector              xi;
    Matrix              Lambda;
    Matrix              Gamma;
    Matrix              Omega;
    int                 niter;
    std::vector<double> dual_objfns;
};

void l3solver_internal(
    L3Result& result,
    const MapMat& X, const MapMat& A, const MapVec& b,
    const MapMat& U, const MapMat& V,
    const MapMat& S, const MapMat& T, double tau,
    int max_iter, double tol, bool verbose = false
    // std::ostream& cout = std::cout
)
{
    // Get dimensions
    const int n = X.rows();
    const int d = X.cols();
    const int L = U.rows();
    const int H = S.rows();
    const int K = A.rows();

    // Pre-compute r and p vectors
    Vector r = precompute_r(X);
    Vector p = precompute_p(A);

    // Create and initialize primal-dual variables
    Vector beta(d), xi(K);
    Matrix Lambda(L, n), Gamma(H, n), Omega(H, n);
    init_params(X, A, U, S, tau, xi, Lambda, Gamma, Omega, beta);

    // Main iterations
    std::vector<double> dual_objfns;
    int i = 0;
    for(; i < max_iter; i++)
    {
        Vector old_xi = xi;
        Vector old_beta = beta;

        update_xi_beta(A, b, p, xi, beta);
        update_Lambda_beta(X, U, V, r, Lambda, beta);
        update_Gamma_Omega_beta(X, S, T, tau, r, Gamma, Omega, beta);

        // Compute difference of alpha and beta
        const double xi_diff = (K > 0) ?
                               (xi - old_xi).norm() :
                               (0.0);
        const double beta_diff = (beta - old_beta).norm();

        // Print progress
        if(verbose && (i % 10 == 0))
        {
            double obj = dual_objfn(
                X, A, b, U, V, S, T, xi, Lambda, Gamma, Omega, tau);
            dual_objfns.push_back(obj);
            std::cout << "Iter " << i << ", dual_objfn = " << obj <<
                ", xi_diff = " << xi_diff <<
                ", beta_diff = " << beta_diff << std::endl;
        }

        // Convergence test
        if(xi_diff < tol && beta_diff < tol)
            break;
    }

    // Save result
    result.beta.swap(beta);
    result.xi.swap(xi);
    result.Lambda.swap(Lambda);
    result.Gamma.swap(Gamma);
    result.Omega.swap(Omega);
    result.niter = i;
    result.dual_objfns.swap(dual_objfns);
}



PYBIND11_MODULE(L3_solver, m) {
    py::class_<L3Result>(m, "L3Result")
        .def(py::init<>())
        .def_readwrite("beta", &L3Result::beta)
        .def_readwrite("xi", &L3Result::xi)
        .def_readwrite("Lambda", &L3Result::Lambda)
        .def_readwrite("Gamma", &L3Result::Gamma)
        .def_readwrite("Omega", &L3Result::Omega)
        .def_readwrite("niter", &L3Result::niter)
        .def_readwrite("dual_objfns", &L3Result::dual_objfns);

    m.doc() = "L3_solver";
    m.def("l3solver_internal", &l3solver_internal);
}
