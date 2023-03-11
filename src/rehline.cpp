#include <RcppEigen.h>
#include <vector>
#include <iostream>

using Rcpp::List;
using Rcpp::NumericVector;
using Rcpp::NumericMatrix;

using Matrix = Eigen::MatrixXd;
using MapMat = Eigen::Map<Matrix>;
using Vector = Eigen::VectorXd;
using MapVec = Eigen::Map<Vector>;

// Dimensions of the matrices involved
// - Input
//   * X        : [n x d]
//   * U, V     : [L x n]
//   * S, T, Tau: [H x n]
//   * A        : [K x d]
//   * b        : [K]
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

class ReHLineSolver
{
private:
    // Dimensions
    const int m_n;
    const int m_d;
    const int m_L;
    const int m_H;
    const int m_K;

    // Input matrices and vectors
    const MapMat& m_X;
    const MapMat& m_U;
    const MapMat& m_V;
    const MapMat& m_S;
    const MapMat& m_T;
    const MapMat& m_Tau;
    const MapMat& m_A;
    const MapVec& m_b;

    // Pre-computed
    Vector m_r;
    Vector m_p;
    Matrix m_Ur;   // u[li] * r[i]
    Matrix m_UrV;  // v[li] / r[i] / u[li]^2
    Matrix m_Sr;   // s[hi]^2 * r[i] + 1

    // Primal variable
    Vector m_beta;

    // Dual variables
    Vector m_xi;
    Matrix m_Lambda;
    Matrix m_Gamma;
    Matrix m_Omega;

public:
    ReHLineSolver(const MapMat& X, const MapMat& U, const MapMat& V,
                  const MapMat& S, const MapMat& T, const MapMat& Tau,
                  const MapMat& A, const MapVec& b) :
        m_n(X.rows()), m_d(X.cols()), m_L(U.rows()), m_H(S.rows()), m_K(A.rows()),
        m_X(X), m_U(U), m_V(V), m_S(S), m_T(T), m_Tau(Tau), m_A(A), m_b(b),
        m_r(m_n), m_p(m_K), m_Ur(m_L, m_n), m_UrV(m_L, m_n), m_Sr(m_H, m_n),
        m_beta(m_d),
        m_xi(m_K), m_Lambda(m_L, m_n), m_Gamma(m_H, m_n), m_Omega(m_H, m_n)
    {
        // Pre-compute the r vector from X
        m_r.noalias() = m_X.rowwise().squaredNorm();

        // Pre-compute the p vector from A
        // A [K x d], K can be zero
        if (m_K > 0)
            m_p.noalias() = m_A.rowwise().squaredNorm();

        if (m_L > 0)
        {
            m_Ur.array() = m_U.array().rowwise() * m_r.transpose().array();
            m_UrV.array() = m_V.array() / m_Ur.array() / m_U.array();
        }

        if (m_H > 0)
        {
            m_Sr.array() = m_S.array().square().rowwise() * m_r.transpose().array() + 1.0;
        }
    }

    // Compute the primal variable beta from dual variables
    // beta = A'xi - U3 * vec(Lambda) - S3 * vec(Gamma)
    // A can be empty, one of U and V may be empty
    inline void set_primal()
    {
        // Initialize beta to zero
        m_beta.setZero();

        // First term
        if (m_K > 0)
            m_beta.noalias() = m_A.transpose() * m_xi;

        // [n x 1]
        Vector LHterm = Vector::Zero(m_n);
        if (m_L > 0)
            LHterm.noalias() = m_U.cwiseProduct(m_Lambda).colwise().sum().transpose();
        // [n x 1]
        if (m_H > 0)
            LHterm.noalias() += m_S.cwiseProduct(m_Gamma).colwise().sum().transpose();

        m_beta.noalias() -= m_X.transpose() * LHterm;
    }

    // Initialize primal and dual variables
    inline void init_params()
    {
        // xi >= 0, initialized to be 1
        if (m_K > 0)
            m_xi.fill(1.0);

        // Each element of Lambda satisfies 0 <= lambda_li <= 1,
        // and we use 0.5 to initialize Lambda
        if (m_L > 0)
            m_Lambda.fill(0.5);

        // Each element of Gamma satisfies 0 <= gamma_hi <= tau_hi,
        // and we use min(0.5 * tau_hi, 1) to initialize (tau_hi can be Inf)
        // Each element of Omega satisfies omega_hi >= 0, initialized to be 0
        if (m_H > 0)
        {
            m_Gamma.noalias() = (0.5 * m_Tau).cwiseMin(1.0);
            // Gamma.fill(std::min(1.0, 0.5 * Tau));
            m_Omega.fill(0.0);
        }

        // Set primal variable based on duals
        set_primal();
    }

    // Update Lambda and beta
    inline void update_Lambda_beta()
    {
        if (m_L < 1)
            return;

        for(int i = 0; i < m_n; i++)
        {
            for(int l = 0; l < m_L; l++)
            {
                const double urv_li = m_UrV(l, i);
                const double ur_li = m_Ur(l, i);
                const double u_li = m_U(l, i);
                const double lambda_li = m_Lambda(l, i);

                // Compute g_li
                const double g_li = urv_li + m_X.row(i).dot(m_beta) / ur_li;
                // Compute new lambda_li
                const double candid = lambda_li + g_li;
                const double newl = std::max(0.0, std::min(1.0, candid));
                // Update Lambda and beta
                m_Lambda(l, i) = newl;
                m_beta.noalias() -= (newl - lambda_li) * u_li * m_X.row(i).transpose();
            }
        }
    }

    // Update Gamma, Omega, and beta
    inline void update_Gamma_Omega_beta()
    {
        if (m_H < 1)
            return;

        for(int i = 0; i < m_n; i++)
        {
            for(int h = 0; h < m_H; h++)
            {
                // tau_hi can be Inf
                const double tau_hi = m_Tau(h, i);
                const double gamma_hi = m_Gamma(h, i);
                const double omega_hi = m_Omega(h, i);
                const double sr_hi = m_Sr(h, i);
                const double s_hi = m_S(h, i);
                const double t_hi = m_T(h, i);

                // Compute g_hi
                const double g_hi = t_hi + s_hi * m_X.row(i).dot(m_beta);
                // Compute epsilon
                double eps = (g_hi + omega_hi - gamma_hi) / sr_hi;
                // Safe to compute std::min(eps, Inf)
                eps = std::min(eps, tau_hi - gamma_hi);
                eps = std::max(eps, -gamma_hi);
                // Update Gamma, Omega, and beta
                m_Gamma(h, i) += eps;
                m_beta.noalias() -= eps * s_hi * m_X.row(i).transpose();
                // Safe to compute std::max(0, -Inf)
                m_Omega(h, i) = std::max(0.0, gamma_hi + eps - tau_hi);
            }
        }
    }

    // Update xi and beta
    inline void update_xi_beta()
    {
        for(int k = 0; k < m_K; k++)
        {
            // Compute g_k
            const double g_k = m_A.row(k).dot(m_beta) + m_b[k];
            // Compute new xi_k
            const double xi_k = m_xi[k];
            const double candid = xi_k - g_k / m_p[k];
            const double newxi = std::max(0.0, candid);
            // Update xi and beta
            m_xi[k] = newxi;
            m_beta.noalias() += (newxi - xi_k) * m_A.row(k).transpose();
        }
    }

    // Compute the dual objective function value
    inline double dual_objfn() const
    {
        // A' * xi, [d x 1], A[K x d] may be empty
        Vector Atxi = Vector::Zero(m_d);
        if (m_K > 0)
            Atxi.noalias() = m_A.transpose() * m_xi;
        // U3 * vec(Lambda), [n x 1], U[L x n] may be empty
        Vector UL(m_n), U3L = Vector::Zero(m_d);
        if (m_L > 0)
        {
            UL.noalias() = m_U.cwiseProduct(m_Lambda).colwise().sum().transpose();
            U3L.noalias() = m_X.transpose() * UL;
        }
        // S3 * vec(Gamma), [n x 1], S[H x n] may be empty
        Vector SG(m_n), S3G = Vector::Zero(m_d);
        if (m_H > 0)
        {
            SG.noalias() = m_S.cwiseProduct(m_Gamma).colwise().sum().transpose();
            S3G.noalias() = m_X.transpose() * SG;
        }

        // Compute dual objective function value
        double obj = 0.0;
        // If K = 0, all terms that depend on A, xi, or b will be zero
        if (m_K > 0)
        {
            // 0.5 * ||Atxi||^2 - Atxi' * U3L - Atxi' * S3G + xi' * b
            const double Atxi_U3L = (m_L > 0) ? (Atxi.dot(U3L)) : 0.0;
            const double Atxi_S3G = (m_H > 0) ? (Atxi.dot(S3G)) : 0.0;
            obj += 0.5 * Atxi.squaredNorm() - Atxi_U3L - Atxi_S3G + m_xi.dot(m_b);
        }
        // If L = 0, all terms that depend on U, V, or Lambda will be zero
        if (m_L > 0)
        {
            // 0.5 * ||U3L||^2 + U3L' * S3G - tr(Lambda * V')
            const double U3L_S3G = (m_H > 0) ? (U3L.dot(S3G)) : 0.0;
            obj += 0.5 * U3L.squaredNorm() + U3L_S3G -
                m_Lambda.cwiseProduct(m_V).sum();
        }
        // If H = 0, all terms that depend on S, T, Gamma, or Omega will be zero
        // Also note that if tau_hi = Inf, then omega_hi = 0
        if (m_H > 0)
        {
            // To avoid computing 0*Inf, clip tau_hi to the largest finite value,
            // and then multiply it with omega_hi
            const double max_finite = std::numeric_limits<double>::max();

            // 0.5 * ||Omega||^2 + 0.5 * ||S3G||^2 + 0.5 * ||Gamma||^2
            // - tr(Gamma * Omega') - tr(Gamma * T') + tr(Tau * Omega')
            obj += 0.5 * m_Omega.squaredNorm() + 0.5 * S3G.squaredNorm() +
                0.5 * m_Gamma.squaredNorm() - m_Gamma.cwiseProduct(m_Omega + m_T).sum() +
                m_Omega.cwiseProduct(m_Tau.cwiseMin(max_finite)).sum();
        }

        return obj;
    }

    inline int solve(std::vector<double>& dual_objfns, int max_iter, double tol, bool verbose = false)
    {
        // Main iterations
        int i = 0;
        for(; i < max_iter; i++)
        {
            Vector old_xi = m_xi;
            Vector old_beta = m_beta;

            update_xi_beta();
            update_Lambda_beta();
            update_Gamma_Omega_beta();

            // Compute difference of alpha and beta
            const double xi_diff = (m_K > 0) ?
                (m_xi - old_xi).norm() :
                (0.0);
            const double beta_diff = (m_beta - old_beta).norm();

            // Print progress
            if(verbose && (i % 10 == 0))
            {
                double obj = dual_objfn();
                dual_objfns.push_back(obj);
                std::cout << "Iter " << i << ", dual_objfn = " << obj <<
                    ", xi_diff = " << xi_diff <<
                        ", beta_diff = " << beta_diff << std::endl;
            }

            // Convergence test
            if(xi_diff < tol && beta_diff < tol)
                break;
        }

        return i;
    }

    Vector& get_beta_ref() { return m_beta; }
    Vector& get_xi_ref() { return m_xi; }
    Matrix& get_Lambda_ref() { return m_Lambda; }
    Matrix& get_Gamma_ref() { return m_Gamma; }
    Matrix& get_Omega_ref() { return m_Omega; }
};



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
    const MapMat& U, const MapMat& S, const MapMat& Tau,
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

    // Each element of Gamma satisfies 0 <= gamma_hi <= tau_hi,
    // and we use min(0.5 * tau_hi, 1) to initialize (tau_hi can be Inf)
    // Each element of Omega satisfies omega_hi >= 0, initialized to be 1
    if (H > 0)
    {
        Gamma.noalias() = (0.5 * Tau).cwiseMin(1.0);
        // Gamma.fill(std::min(1.0, 0.5 * Tau));
        Omega.fill(0.0);
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
    const MapMat& Tau, const Vector& r,
    Matrix& Gamma, Matrix& Omega, Vector& beta
)
{
    const int n = X.rows();
    const int H = S.rows();
    for(int h = 0; h < H; h++)
    {
        for(int i = 0; i < n; i++)
        {
            // tau_hi can be Inf
            const double tau_hi = Tau(h, i);
            // Compute epsilon
            const double s_hi = S(h, i);
            const double gamma_hi = Gamma(h, i);
            double eps = T(h, i) + Omega(h, i) +
                s_hi * X.row(i).dot(beta) - gamma_hi;
            eps = eps / (s_hi * s_hi * r[i] + 1.0);
            // Safe to compute std::min(eps, Inf)
            eps = std::min(eps, tau_hi - gamma_hi);
            eps = std::max(eps, -gamma_hi);
            // Update Gamma, Omega, and beta
            Gamma(h, i) += eps;
            beta.noalias() -= eps * s_hi * X.row(i).transpose();
            // Safe to compute std::max(0, -Inf)
            Omega(h, i) = std::max(0.0, gamma_hi + eps - tau_hi);
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
    const Matrix& Gamma, const Matrix& Omega, const MapMat& Tau
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
    // Also note that if tau_hi = Inf, then omega_hi = 0
    if (H > 0)
    {
        // To avoid computing 0*Inf, clip tau_hi to the largest finite value,
        // and then multiply it with omega_hi
        const double max_finite = std::numeric_limits<double>::max();

        // 0.5 * ||Omega||^2 + 0.5 * ||S3G||^2 + 0.5 * ||Gamma||^2
        // - tr(Gamma * Omega') - tr(Gamma * T') + tr(Tau * Omega')
        obj += 0.5 * Omega.squaredNorm() + 0.5 * S3G.squaredNorm() +
               0.5 * Gamma.squaredNorm() - Gamma.cwiseProduct(Omega + T).sum() +
               Omega.cwiseProduct(Tau.cwiseMin(max_finite)).sum();

        // if (std::isinf(tau))
        //     obj += 0.5 * S3G.squaredNorm() + 0.5 * Gamma.squaredNorm() -
        //         Gamma.cwiseProduct(T).sum();
        // else
        //     obj += 0.5 * Omega.squaredNorm() + 0.5 * S3G.squaredNorm() +
        //         0.5 * Gamma.squaredNorm() - Gamma.cwiseProduct(Omega + T).sum() +
        //         tau * Omega.sum();
    }

    return obj;
}



struct OptResult
{
    Vector              beta;
    Vector              xi;
    Matrix              Lambda;
    Matrix              Gamma;
    Matrix              Omega;
    int                 niter;
    std::vector<double> dual_objfns;
};

void rehline_internal(
    OptResult& result,
    const MapMat& X, const MapMat& A, const MapVec& b,
    const MapMat& U, const MapMat& V,
    const MapMat& S, const MapMat& T, const MapMat& Tau,
    int max_iter, double tol, bool verbose = false,
    std::ostream& cout = std::cout
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
    init_params(X, A, U, S, Tau, xi, Lambda, Gamma, Omega, beta);

    // Main iterations
    std::vector<double> dual_objfns;
    int i = 0;
    for(; i < max_iter; i++)
    {
        Vector old_xi = xi;
        Vector old_beta = beta;

        update_xi_beta(A, b, p, xi, beta);
        update_Lambda_beta(X, U, V, r, Lambda, beta);
        update_Gamma_Omega_beta(X, S, T, Tau, r, Gamma, Omega, beta);

        // Compute difference of alpha and beta
        const double xi_diff = (K > 0) ?
                               (xi - old_xi).norm() :
                               (0.0);
        const double beta_diff = (beta - old_beta).norm();

        // Print progress
        if(verbose && (i % 10 == 0))
        {
            double obj = dual_objfn(
                X, A, b, U, V, S, T, xi, Lambda, Gamma, Omega, Tau);
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

// [[Rcpp::export(rehline_)]]
List rehline(
    NumericMatrix Xmat, NumericMatrix Amat, NumericVector bvec,
    NumericMatrix Umat, NumericMatrix Vmat,
    NumericMatrix Smat, NumericMatrix Tmat, NumericMatrix TauMat,
    int max_iter, double tol, bool verbose = false
)
{
    MapMat X = Rcpp::as<MapMat>(Xmat);
    MapMat A = Rcpp::as<MapMat>(Amat);
    MapVec b = Rcpp::as<MapVec>(bvec);
    MapMat U = Rcpp::as<MapMat>(Umat);
    MapMat V = Rcpp::as<MapMat>(Vmat);
    MapMat S = Rcpp::as<MapMat>(Smat);
    MapMat T = Rcpp::as<MapMat>(Tmat);
    MapMat Tau = Rcpp::as<MapMat>(TauMat);
    OptResult result;

    rehline_internal(
        result,
        X, A, b, U, V, S, T,
        Tau, max_iter, tol, verbose, Rcpp::Rcout
    );

    return List::create(
        Rcpp::Named("beta")        = result.beta,
        Rcpp::Named("xi")          = result.xi,
        Rcpp::Named("Lambda")      = result.Lambda,
        Rcpp::Named("Gamma")       = result.Gamma,
        Rcpp::Named("Omega")       = result.Omega,
        Rcpp::Named("niter")       = result.niter,
        Rcpp::Named("dual_objfns") = result.dual_objfns
    );
}



void rehline_internal2(
    OptResult& result,
    const MapMat& X, const MapMat& A, const MapVec& b,
    const MapMat& U, const MapMat& V,
    const MapMat& S, const MapMat& T, const MapMat& Tau,
    int max_iter, double tol, bool verbose = false,
    std::ostream& cout = std::cout
)
{
    // Create solver
    ReHLineSolver solver(X, U, V, S, T, Tau, A, b);

    // Initialize parameters
    solver.init_params();

    // Main iterations
    std::vector<double> dual_objfns;
    int niter = solver.solve(dual_objfns, max_iter, tol, verbose);

    // Save result
    result.beta.swap(solver.get_beta_ref());
    result.xi.swap(solver.get_xi_ref());
    result.Lambda.swap(solver.get_Lambda_ref());
    result.Gamma.swap(solver.get_Gamma_ref());
    result.Omega.swap(solver.get_Omega_ref());
    result.niter = niter;
    result.dual_objfns.swap(dual_objfns);
}

// [[Rcpp::export(rehline2_)]]
List rehline2(
    NumericMatrix Xmat, NumericMatrix Amat, NumericVector bvec,
    NumericMatrix Umat, NumericMatrix Vmat,
    NumericMatrix Smat, NumericMatrix Tmat, NumericMatrix TauMat,
    int max_iter, double tol, bool verbose = false
)
{
    MapMat X = Rcpp::as<MapMat>(Xmat);
    MapMat A = Rcpp::as<MapMat>(Amat);
    MapVec b = Rcpp::as<MapVec>(bvec);
    MapMat U = Rcpp::as<MapMat>(Umat);
    MapMat V = Rcpp::as<MapMat>(Vmat);
    MapMat S = Rcpp::as<MapMat>(Smat);
    MapMat T = Rcpp::as<MapMat>(Tmat);
    MapMat Tau = Rcpp::as<MapMat>(TauMat);
    OptResult result;

    rehline_internal2(
        result,
        X, A, b, U, V, S, T,
        Tau, max_iter, tol, verbose, Rcpp::Rcout
    );

    return List::create(
        Rcpp::Named("beta")        = result.beta,
        Rcpp::Named("xi")          = result.xi,
        Rcpp::Named("Lambda")      = result.Lambda,
        Rcpp::Named("Gamma")       = result.Gamma,
        Rcpp::Named("Omega")       = result.Omega,
        Rcpp::Named("niter")       = result.niter,
        Rcpp::Named("dual_objfns") = result.dual_objfns
    );
}
