#include <RcppEigen.h>
#include <vector>
#include <iostream>
#include <type_traits>

using Rcpp::List;
using Rcpp::NumericVector;
using Rcpp::NumericMatrix;

using Matrix = Eigen::MatrixXd;
using MapMat = Eigen::Map<Matrix>;
using Vector = Eigen::VectorXd;
using MapVec = Eigen::Map<Vector>;

// We really want some matrices to be row-majored, since they can be more
// efficient in certain matrix operations, for example X.row(i).dot(v)
//
// If Matrix is already row-majored, we save a const reference; otherwise
// we make a copy
using RMatrix = std::conditional<
    Matrix::IsRowMajor,
    Eigen::Ref<const Matrix>,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
>::type;

// Used in random_shuffle(), generating a random integer from {0, 1, ..., i-1}
// This function is designed for R, as CRAN requires using R's own RNG
// For C++ or Python, the following simplified version can be used:
/*

inline int rand_less_than(int i)
{
    return int(std::rand() % i);
}

*/
inline int rand_less_than(int i)
{
    // Typically on Linux and MacOS, RAND_MAX == 2147483647
    // Windows has different definition, RAND_MAX == 32767
    // We manually set the limit to make sure that different OS are compatible
    std::int32_t rand_max = std::numeric_limits<std::int32_t>::max();
    std::int32_t r = std::int32_t(R::unif_rand() * rand_max);
    return int(r % i);
}

// Randomly shuffle a vector
//
// On Mac, std::random_shuffle() uses a "backward" implementation,
// which leads to different results from Windows and Linux
// Therefore, we use a consistent implementation based on GCC code
template <typename RandomAccessIterator, typename RandomNumberGenerator>
void random_shuffle(RandomAccessIterator first, RandomAccessIterator last, RandomNumberGenerator& gen)
{
    if(first == last)
        return;
    for(RandomAccessIterator i = first + 1; i != last; ++i)
    {
        RandomAccessIterator j = first + gen((i - first) + 1);
        if(i != j)
            std::iter_swap(i, j);
    }
}

// Reset the free variable set to [0, 1, ..., n-1]
inline void reset_fv_set(std::vector<int>& fvset, std::size_t n)
{
    fvset.resize(n);
    // Fill the vector with 0, 1, ..., n-1
    std::iota(fvset.begin(), fvset.end(), 0);
}

// Reset the free variable set to [(0, 0), (0, 1), ..., (n-1, m-2), (n-1, m-1)]
inline void reset_fv_set(std::vector<std::pair<int, int>>& fvset, std::size_t n, std::size_t m)
{
    fvset.resize(n * m);
    for(std::size_t i = 0; i < n * m; i++)
        fvset[i] = std::make_pair(i % n, i / n);
}

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
    RMatrix       m_X;
    const MapMat& m_U;
    const MapMat& m_V;
    const MapMat& m_S;
    const MapMat& m_T;
    const MapMat& m_Tau;
    RMatrix       m_A;
    const MapVec& m_b;

    // Pre-computed
    Vector m_gk_denom;   // ||a[k]||^2
    Matrix m_gli_denom;  // (u[li] * ||x[i]||)^2
    Matrix m_ghi_denom;  // (s[hi] * ||x[i]||)^2 + 1

    // Primal variable
    Vector m_beta;

    // Dual variables
    Vector m_xi;
    Matrix m_Lambda;
    Matrix m_Gamma;

    // Free variable sets
    std::vector<int> m_fv_feas;
    std::vector<std::pair<int, int>> m_fv_relu;
    std::vector<std::pair<int, int>> m_fv_rehu;

    // =================== Initialization functions =================== //

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

    // =================== Evaluating objection function =================== //

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
        // If H = 0, all terms that depend on S, T, or Gamma will be zero
        if (m_H > 0)
        {
            // 0.5 * ||S3G||^2 + 0.5 * ||Gamma||^2 - tr(Gamma * T')
            obj += 0.5 * S3G.squaredNorm() + 0.5 * m_Gamma.squaredNorm() - m_Gamma.cwiseProduct(m_T).sum();
        }

        return obj;
    }

    // =================== Updating functions (sequential) =================== //

    // Update xi and beta
    inline void update_xi_beta()
    {
        if (m_K < 1)
            return;

        for(int k = 0; k < m_K; k++)
        {
            const double xi_k = m_xi[k];

            // Compute g_k
            const double g_k = m_A.row(k).dot(m_beta) + m_b[k];
            // Compute new xi_k
            const double candid = xi_k - g_k / m_gk_denom[k];
            const double newxi = std::max(0.0, candid);
            // Update xi and beta
            m_xi[k] = newxi;
            m_beta.noalias() += (newxi - xi_k) * m_A.row(k).transpose();
        }
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
                const double u_li = m_U(l, i);
                const double v_li = m_V(l, i);
                const double lambda_li = m_Lambda(l, i);

                // Compute g_li
                const double g_li = -(u_li * m_X.row(i).dot(m_beta) + v_li);
                // Compute new lambda_li
                const double candid = lambda_li - g_li / m_gli_denom(l, i);
                const double newl = std::max(0.0, std::min(1.0, candid));
                // Update Lambda and beta
                m_Lambda(l, i) = newl;
                m_beta.noalias() -= (newl - lambda_li) * u_li * m_X.row(i).transpose();
            }
        }
    }

    // Update Gamma, and beta
    inline void update_Gamma_beta()
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
                const double s_hi = m_S(h, i);
                const double t_hi = m_T(h, i);

                // Compute g_hi
                const double g_hi = gamma_hi - (s_hi * m_X.row(i).dot(m_beta) + t_hi);
                // Compute new gamma_hi
                const double candid = gamma_hi - g_hi / m_ghi_denom(h, i);
                const double newg = std::max(0.0, std::min(tau_hi, candid));
                // Update Gamma and beta
                m_Gamma(h, i) = newg;
                m_beta.noalias() -= (newg - gamma_hi) * s_hi * m_X.row(i).transpose();
            }
        }
    }

    // =================== Updating functions (free variable set) ================ //

    // Determine whether to shrink xi, and compute the projected gradient (PG)
    // Shrink if xi=0 and grad>ub
    // PG is zero if xi=0 and grad>=0
    inline bool pg_xi(double xi, double grad, double ub, double& pg) const
    {
        pg = (xi == 0.0 && grad >= 0.0) ? 0.0 : grad;
        const bool shrink = (xi == 0.0) && (grad > ub);
        return shrink;
    }
    // Update xi and beta
    // Overloaded version based on free variable set
    inline void update_xi_beta(std::vector<int>& fv_set, double& min_pg, double& max_pg)
    {
        if (m_K < 1)
            return;

        // Permutation
        random_shuffle(fv_set.begin(), fv_set.end(), rand_less_than);
        // New free variable set
        std::vector<int> new_set;
        new_set.reserve(fv_set.size());

        // Compute shrinking threshold
        constexpr double Inf = std::numeric_limits<double>::infinity();
        const double ub = (max_pg > 0.0) ? max_pg : Inf;
        // Compute minimum and maximum projected gradient (PG) for this round
        min_pg = Inf;
        max_pg = -Inf;
        for(auto k: fv_set)
        {
            const double xi_k = m_xi[k];

            // Compute g_k
            const double g_k = m_A.row(k).dot(m_beta) + m_b[k];
            // PG and shrink
            double pg;
            const bool shrink = pg_xi(xi_k, g_k, ub, pg);
            if (shrink)
               continue;

            // Update PG bounds
            max_pg = std::max(max_pg, pg);
            min_pg = std::min(min_pg, pg);
            // Compute new xi_k
            const double candid = xi_k - g_k / m_gk_denom[k];
            const double newxi = std::max(0.0, candid);
            // Update xi and beta
            m_xi[k] = newxi;
            m_beta.noalias() += (newxi - xi_k) * m_A.row(k).transpose();

            // Add to new free variable set
            new_set.push_back(k);
        }

        // Update free variable set
        fv_set.swap(new_set);
    }

    // Determine whether to shrink lambda, and compute the projected gradient (PG)
    // Shrink if (lambda=0 and grad>ub) or (lambda=1 and grad<lb)
    // PG is zero if (lambda=0 and grad>=0) or (lambda=1 and grad<=0)
    inline bool pg_lambda(double lambda, double grad, double lb, double ub, double& pg) const
    {
        pg = ((lambda == 0.0 && grad >= 0.0) || (lambda == 1.0 && grad <= 0.0)) ?
             0.0 :
             grad;
        const bool shrink = (lambda == 0.0 && grad > ub) || (lambda == 1.0 && grad < lb);
        return shrink;
    }
    // Update Lambda and beta
    // Overloaded version based on free variable set
    inline void update_Lambda_beta(std::vector<std::pair<int, int>>& fv_set, double& min_pg, double& max_pg)
    {
        if (m_L < 1)
            return;

        // Permutation
        random_shuffle(fv_set.begin(), fv_set.end(), rand_less_than);
        // New free variable set
        std::vector<std::pair<int, int>> new_set;
        new_set.reserve(fv_set.size());

        // Compute shrinking thresholds
        constexpr double Inf = std::numeric_limits<double>::infinity();
        const double lb = (min_pg < 0.0) ? min_pg : -Inf;
        const double ub = (max_pg > 0.0) ? max_pg : Inf;
        // Compute minimum and maximum projected gradient (PG) for this round
        min_pg = Inf;
        max_pg = -Inf;
        for(auto rc: fv_set)
        {
            const int l = rc.first;
            const int i = rc.second;

            const double u_li = m_U(l, i);
            const double v_li = m_V(l, i);
            const double lambda_li = m_Lambda(l, i);

            // Compute g_li
            const double g_li = -(u_li * m_X.row(i).dot(m_beta) + v_li);
            // PG and shrink
            double pg;
            const bool shrink = pg_lambda(lambda_li, g_li, lb, ub, pg);
            if (shrink)
                continue;

            // Update PG bounds
            max_pg = std::max(max_pg, pg);
            min_pg = std::min(min_pg, pg);
            // Compute new lambda_li
            const double candid = lambda_li - g_li / m_gli_denom(l, i);;
            const double newl = std::max(0.0, std::min(1.0, candid));
            // Update Lambda and beta
            m_Lambda(l, i) = newl;
            m_beta.noalias() -= (newl - lambda_li) * u_li * m_X.row(i).transpose();

            // Add to new free variable set
            new_set.emplace_back(l, i);
        }

        // Update free variable set
        fv_set.swap(new_set);
    }

    // Determine whether to shrink gamma, and compute the projected gradient (PG)
    // Shrink if (gamma=0 and grad>ub) or (lambda=tau and grad<lb)
    // PG is zero if (lambda=0 and grad>=0) or (lambda=1 and grad<=0)
    inline bool pg_gamma(double gamma, double grad, double tau, double lb, double ub, double& pg) const
    {
        pg = ((gamma == 0.0 && grad >= 0.0) || (gamma == tau && grad <= 0.0)) ?
             0.0 :
             grad;
        const bool shrink = (gamma == 0.0 && grad > ub) || (gamma == tau && grad < lb);
        return shrink;
    }
    // Update Gamma and beta
    // Overloaded version based on free variable set
    inline void update_Gamma_beta(std::vector<std::pair<int, int>>& fv_set, double& min_pg, double& max_pg)
    {
        if (m_H < 1)
            return;

        // Permutation
        random_shuffle(fv_set.begin(), fv_set.end(), rand_less_than);
        // New free variable set
        std::vector<std::pair<int, int>> new_set;
        new_set.reserve(fv_set.size());

        // Compute shrinking thresholds
        constexpr double Inf = std::numeric_limits<double>::infinity();
        const double lb = (min_pg < 0.0) ? min_pg : -Inf;
        const double ub = (max_pg > 0.0) ? max_pg : Inf;
        // Compute minimum and maximum projected gradient (PG) for this round
        min_pg = Inf;
        max_pg = -Inf;
        for(auto rc: fv_set)
        {
            const int h = rc.first;
            const int i = rc.second;

            // tau_hi can be Inf
            const double tau_hi = m_Tau(h, i);
            const double gamma_hi = m_Gamma(h, i);
            const double s_hi = m_S(h, i);
            const double t_hi = m_T(h, i);

            // Compute g_hi
            const double g_hi = gamma_hi - (s_hi * m_X.row(i).dot(m_beta) + t_hi);
            // PG and shrink
            double pg;
            const bool shrink = pg_gamma(gamma_hi, g_hi, tau_hi, lb, ub, pg);
            if (shrink)
                continue;

            // Update PG bounds
            max_pg = std::max(max_pg, pg);
            min_pg = std::min(min_pg, pg);
            // Compute new gamma_hi
            const double candid = gamma_hi - g_hi / m_ghi_denom(h, i);
            const double newg = std::max(0.0, std::min(tau_hi, candid));
            // Update Gamma and beta
            m_Gamma(h, i) = newg;
            m_beta.noalias() -= (newg - gamma_hi) * s_hi * m_X.row(i).transpose();

            // Add to new free variable set
            new_set.emplace_back(h, i);
        }

        // Update free variable set
        fv_set.swap(new_set);
    }

public:
    ReHLineSolver(const MapMat& X, const MapMat& U, const MapMat& V,
                  const MapMat& S, const MapMat& T, const MapMat& Tau,
                  const MapMat& A, const MapVec& b) :
        m_n(X.rows()), m_d(X.cols()), m_L(U.rows()), m_H(S.rows()), m_K(A.rows()),
        m_X(X), m_U(U), m_V(V), m_S(S), m_T(T), m_Tau(Tau), m_A(A), m_b(b),
        m_gk_denom(m_K), m_gli_denom(m_L, m_n), m_ghi_denom(m_H, m_n),
        m_beta(m_d),
        m_xi(m_K), m_Lambda(m_L, m_n), m_Gamma(m_H, m_n)
    {
        // A [K x d], K can be zero
        if (m_K > 0)
            m_gk_denom.noalias() = m_A.rowwise().squaredNorm();

        Vector xi2 = m_X.rowwise().squaredNorm();
        if (m_L > 0)
        {
            m_gli_denom.array() = m_U.array().square().rowwise() * xi2.transpose().array();
        }

        if (m_H > 0)
        {
            m_ghi_denom.array() = m_S.array().square().rowwise() * xi2.transpose().array() + 1.0;
        }
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
        if (m_H > 0)
        {
            m_Gamma.noalias() = (0.5 * m_Tau).cwiseMin(1.0);
            // Gamma.fill(std::min(1.0, 0.5 * tau));
        }

        // Set primal variable based on duals
        set_primal();
    }

    inline int solve_vanilla(std::vector<double>& dual_objfns, int max_iter, double tol,
                             int verbose = 0, std::ostream& cout = std::cout)
    {
        // Main iterations
        int i = 0;
        Vector old_xi(m_K), old_beta(m_d);
        for(; i < max_iter; i++)
        {
            old_xi.noalias() = m_xi;
            old_beta.noalias() = m_beta;

            update_xi_beta();
            update_Lambda_beta();
            update_Gamma_beta();

            // Compute difference of xi and beta
            const double xi_diff = (m_K > 0) ? (m_xi - old_xi).norm() : 0.0;
            const double beta_diff = (m_beta - old_beta).norm();

            // Print progress
            if (verbose && (i % 50 == 0))
            {
                double obj = dual_objfn();
                dual_objfns.push_back(obj);
                cout << "Iter " << i << ", dual_objfn = " << obj <<
                    ", xi_diff = " << xi_diff <<
                    ", beta_diff = " << beta_diff << std::endl;
            }

            // Convergence test based on change of variable values
            const bool vars_conv = (xi_diff < tol) && (beta_diff < tol);
            if (vars_conv)
                break;
        }

        return i;
    }

    inline int solve(std::vector<double>& dual_objfns, int max_iter, double tol,
                     int verbose = 0, std::ostream& cout = std::cout)
    {
        // Free variable sets
        reset_fv_set(m_fv_feas, m_K);
        reset_fv_set(m_fv_relu, m_L, m_n);
        reset_fv_set(m_fv_rehu, m_H, m_n);

        // Shrinking thresholds
        constexpr double Inf = std::numeric_limits<double>::infinity();
        double xi_min_pg = Inf, lambda_min_pg = Inf, gamma_min_pg = Inf;
        double xi_max_pg = -Inf, lambda_max_pg = -Inf, gamma_max_pg = -Inf;

        // Main iterations
        int i = 0;
        Vector old_xi(m_K), old_beta(m_d);
        for(; i < max_iter; i++)
        {
            old_xi.noalias() = m_xi;
            old_beta.noalias() = m_beta;

            update_xi_beta(m_fv_feas, xi_min_pg, xi_max_pg);
            update_Lambda_beta(m_fv_relu, lambda_min_pg, lambda_max_pg);
            update_Gamma_beta(m_fv_rehu, gamma_min_pg, gamma_max_pg);

            // Compute difference of xi and beta
            const double xi_diff = (m_K > 0) ? (m_xi - old_xi).norm() : 0.0;
            const double beta_diff = (m_beta - old_beta).norm();

            // Convergence test based on change of variable values
            const bool vars_conv = (xi_diff < tol) && (beta_diff < tol);
            // Convergence test based on PG
            const bool pg_conv = (xi_max_pg - xi_min_pg < tol) &&
                                 (std::abs(xi_max_pg) < tol) &&
                                 (std::abs(xi_min_pg) < tol) &&
                                 (lambda_max_pg - lambda_min_pg < tol) &&
                                 (std::abs(lambda_max_pg) < tol) &&
                                 (std::abs(lambda_min_pg) < tol) &&
                                 (gamma_max_pg - gamma_min_pg < tol) &&
                                 (std::abs(gamma_max_pg) < tol) &&
                                 (std::abs(gamma_min_pg) < tol);
            // Whether we are using all variables
            const bool all_vars = (m_fv_feas.size() == static_cast<std::size_t>(m_K)) &&
                                  (m_fv_relu.size() == static_cast<std::size_t>(m_L * m_n)) &&
                                  (m_fv_rehu.size() == static_cast<std::size_t>(m_H * m_n));

            // Print progress
            if (verbose && (i % 50 == 0))
            {
                double obj = dual_objfn();
                dual_objfns.push_back(obj);
                cout << "Iter " << i << ", dual_objfn = " << obj <<
                    ", xi_diff = " << xi_diff <<
                    ", beta_diff = " << beta_diff << std::endl;
                if (verbose >= 2)
                {
                    cout << "    xi (" << m_fv_feas.size() << "/" << m_K <<
                        "), lambda (" << m_fv_relu.size() << "/" << m_L * m_n <<
                        "), gamma (" << m_fv_rehu.size() << "/" << m_H * m_n << ")" << std::endl;
                    cout << "    xi_pg = (" << xi_min_pg << ", " << xi_max_pg <<
                        "), lambda_pg = (" << lambda_min_pg << ", " << lambda_max_pg <<
                        "), gamma_pg = (" << gamma_min_pg << ", " << gamma_max_pg << ")" << std::endl;
                }
            }

            // If variable value or PG converges but not on all variables,
            // use all variables in the next iteration
            if ((vars_conv || pg_conv) && (!all_vars))
            {
                if (verbose)
                {
                    cout << "*** Iter " << i <<
                        ", free variables converge; next test on all variables" << std::endl;
                }
                reset_fv_set(m_fv_feas, m_K);
                reset_fv_set(m_fv_relu, m_L, m_n);
                reset_fv_set(m_fv_rehu, m_H, m_n);
                xi_min_pg = lambda_min_pg = gamma_min_pg = Inf;
                xi_max_pg = lambda_max_pg = gamma_max_pg = -Inf;
                // Also recompute beta to improve precision
                // set_primal();
                continue;
            }
            if (all_vars && (vars_conv || pg_conv))
                break;
        }

        return i;
    }

    Vector& get_beta_ref() { return m_beta; }
    Vector& get_xi_ref() { return m_xi; }
    Matrix& get_Lambda_ref() { return m_Lambda; }
    Matrix& get_Gamma_ref() { return m_Gamma; }
};

struct OptResult
{
    Vector              beta;
    Vector              xi;
    Matrix              Lambda;
    Matrix              Gamma;
    int                 niter;
    std::vector<double> dual_objfns;
};

void rehline_internal(
    OptResult& result,
    const MapMat& X, const MapMat& A, const MapVec& b,
    const MapMat& U, const MapMat& V,
    const MapMat& S, const MapMat& T, const MapMat& Tau,
    int max_iter, double tol, bool shrink = true, int verbose = 0,
    std::ostream& cout = std::cout
)
{
    // Create solver
    ReHLineSolver solver(X, U, V, S, T, Tau, A, b);

    // Initialize parameters
    solver.init_params();

    // Main iterations
    std::vector<double> dual_objfns;
    int niter;
    if (shrink)
        niter = solver.solve(dual_objfns, max_iter, tol, verbose, cout);
    else
        niter = solver.solve_vanilla(dual_objfns, max_iter, tol, verbose, cout);

    // Save result
    result.beta.swap(solver.get_beta_ref());
    result.xi.swap(solver.get_xi_ref());
    result.Lambda.swap(solver.get_Lambda_ref());
    result.Gamma.swap(solver.get_Gamma_ref());
    result.niter = niter;
    result.dual_objfns.swap(dual_objfns);
}

// [[Rcpp::export(rehline_)]]
List rehline(
    NumericMatrix Xmat, NumericMatrix Amat, NumericVector bvec,
    NumericMatrix Umat, NumericMatrix Vmat,
    NumericMatrix Smat, NumericMatrix Tmat, NumericMatrix TauMat,
    int max_iter, double tol, bool shrink = true, int verbose = 0
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
        Tau, max_iter, tol, shrink, verbose, Rcpp::Rcout
    );

    return List::create(
        Rcpp::Named("beta")        = result.beta,
        Rcpp::Named("xi")          = result.xi,
        Rcpp::Named("Lambda")      = result.Lambda,
        Rcpp::Named("Gamma")       = result.Gamma,
        Rcpp::Named("niter")       = result.niter,
        Rcpp::Named("dual_objfns") = result.dual_objfns
    );
}
