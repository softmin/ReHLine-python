#ifndef REHLINE_H
#define REHLINE_H

#include <vector>
#include <numeric>
#include <random>
#include <type_traits>
#include <iostream>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/QR>
#include "design.h"

namespace rehline {

// ========================= Internal utility functions ========================= //
namespace internal {

// A simple wrapper of existing RNG
template <typename Index = int>
class SimpleRNG
{
private:
    std::mt19937 m_rng;

public:
    // Set seed
    void seed(Index seed) { m_rng.seed(seed); }

    // Used in random_shuffle(), generating a random integer from {0, 1, ..., i-1}
    Index operator()(Index i)
    {
        return Index(m_rng() % i);
    }
};

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

// Reset the free variable set to [0, 1, ..., n-1] (if the variables form a vector)
template <typename Index = int>
void reset_fv_set(std::vector<Index>& fvset, std::size_t n)
{
    fvset.resize(n);
    // Fill the vector with 0, 1, ..., n-1
    std::iota(fvset.begin(), fvset.end(), Index(0));
}

// Reset the free variable set to [(0, 0), (0, 1), ..., (n-1, m-2), (n-1, m-1)] (if the variables form a matrix)
template <typename Index = int>
void reset_fv_set(std::vector<std::pair<Index, Index>>& fvset, std::size_t n, std::size_t m)
{
    fvset.resize(n * m);
    for(std::size_t i = 0; i < n * m; i++)
        fvset[i] = std::make_pair(i % n, i / n);
}


}  // namespace internal
// ========================= Internal utility functions ========================= //



// Dimensions of the matrices involved
// - Input
//   * X        : [n x d]
//   * U, V     : [L x n]
//   * S, T, Tau: [H x n]
//   * A        : [K x d]
//   * b        : [K]
//   * rho      : [d]
// - Pre-computed
//   * r: [n]
//   * p: [K]
// - Primal
//   * beta: [d]
// - Dual
//   * xi    : [K]
//   * Lambda: [L x n]
//   * Gamma : [H x n]
//   * mu    : [d]

// Results of the optimization algorithm
template <typename Matrix = Eigen::MatrixXd, typename Index = int>
struct ReHLineResult
{
    using Scalar = typename Matrix::Scalar;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    Vector              beta;           // Primal variable
    Vector              xi;             // Dual variables
    Matrix              Lambda;         // Dual variables
    Matrix              Gamma;          // Dual variables
    Vector              mu;             // Dual variables
    Scalar              objective = 0;
    Scalar              dual_objective = 0;
    Scalar              dual_gap = 0;
    Scalar              constraint_violation = 0;
    Scalar              scaled_constraint_violation = 0;
    Scalar              kkt_residual = 0;
    bool                converged = false;
    Index               niter;          // Number of iterations
    std::vector<Scalar> dual_objfns;    // Recorded dual objective function values
    std::vector<Scalar> primal_objfns;  // Recorded primal objective function values
};

// The main ReHLine solver
// "Matrix" is the type of input data matrix, can be row-majored or column-majored
template <typename Matrix = Eigen::MatrixXd, typename Index = int, bool Composite = false>
class ReHLineSolver
{
private:
    using Scalar = typename Matrix::Scalar;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using ConstRefMat = Eigen::Ref<const Matrix>;
    using ConstRefVec = Eigen::Ref<const Vector>;

    // We really want some matrices to be row-majored, since they can be more
    // efficient in certain matrix operations, for example X.row(i).dot(v)
    //
    // If the data Matrix is already row-majored, we save a const reference;
    // otherwise we make a copy
    using RMatrix = typename std::conditional<
        Matrix::IsRowMajor,
        Eigen::Ref<const Matrix>,
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    >::type;

    // RNG
    internal::SimpleRNG<Index> m_rng;

    // Dimensions
    const Index m_n;
    const Index m_d;
    const Index m_L;
    const Index m_H;
    const Index m_K;
    const Index m_W;

    // Input matrices and vectors
    internal::Design<Matrix, Index, Composite> m_X;
    ConstRefMat m_U;
    ConstRefMat m_V;
    ConstRefMat m_S;
    ConstRefMat m_T;
    ConstRefMat m_Tau;
    RMatrix     m_A;
    ConstRefVec m_b;
    ConstRefVec m_rho;

    // Pre-computed
    Vector m_gk_denom;   // ||a[k]||^2
    Matrix m_gli_denom;  // (u[li] * ||x[i]||)^2
    Matrix m_ghi_denom;  // (s[hi] * ||x[i]||)^2 + 1

    // Primal variable
    Vector m_beta;
    Vector m_dual_beta;
    bool m_primal_polished = false;

    // Dual variables
    Vector m_xi;
    Matrix m_Lambda;
    Matrix m_Gamma;
    Vector m_mu;

    // Free variable sets
    std::vector<Index> m_fv_feas;
    std::vector<Index> m_fv_l1mu;
    std::vector<std::pair<Index, Index>> m_fv_relu;
    std::vector<std::pair<Index, Index>> m_fv_rehu;

    // =================== Initialization functions =================== //

    // Compute the primal variable beta from dual variables
    // beta = A'xi - U3 * vec(Lambda) - S3 * vec(Gamma) + 2 * Mu - rho
    // A can be empty, one of U and V may be empty
    inline void set_primal()
    {
        // Initialize beta to zero
        m_primal_polished = false;
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
        m_X.subtract_transpose(m_beta, LHterm);
        // L1 related
        if (m_W > 0)
            m_beta.noalias() += Scalar(2.0) * m_mu - m_rho;
    }

    // =================== Evaluating objection function =================== //

    // Compute the primal objective function value
    inline Scalar primal_objfn() const
    {
        Scalar result = Scalar(0);
        const Vector Xbeta = m_X.multiply(m_beta);
        // ReLU part
        if (m_L > 0)
        {
            result += (m_U.cwiseProduct(Xbeta.transpose().replicate(m_L, 1)) +
                m_V).cwiseMax(Scalar(0)).sum();
        }
        // ReHU part
        if (m_H > 0)
        {
            const Matrix z = (m_S.cwiseProduct(Xbeta.transpose().replicate(m_H, 1)) +
                m_T).cwiseMax(Scalar(0));
            result += (z.array() <= m_Tau.array()).select(
                z.array().square() * Scalar(0.5),
                m_Tau.array() * (z.array() - m_Tau.array() * Scalar(0.5))
            ).sum();
        }
        // Quadratic term
        result += Scalar(0.5) * m_beta.squaredNorm();
        // L1 penalty term
        if (m_W > 0)
            result += m_beta.cwiseAbs().cwiseProduct(m_rho).sum();
        return result;
    }

    // Compute the dual objective function value
    inline Scalar dual_objfn() const
    {
        // The historical trace stores the minimized negative dual.
        const Vector& dual_beta = m_primal_polished ? m_dual_beta : m_beta;
        Scalar result = Scalar(0.5) * dual_beta.squaredNorm();
        if (m_K > 0) result += m_xi.dot(m_b);
        if (m_L > 0) result -= m_Lambda.cwiseProduct(m_V).sum();
        if (m_H > 0)
            result += Scalar(0.5) * m_Gamma.squaredNorm() - m_Gamma.cwiseProduct(m_T).sum();
        return result;
    }

    // =================== Updating functions (sequential) =================== //

    // Update xi and beta
    inline void update_xi_beta()
    {
        if (m_K < 1)
            return;

        for (Index k = 0; k < m_K; k++)
        {
            const Scalar xi_k = m_xi[k];

            if (m_gk_denom[k] == Scalar(0)) { m_xi[k] = Scalar(0); continue; }
            // Compute g_k
            const Scalar g_k = m_A.row(k).dot(m_beta) + m_b[k];
            // Compute new xi_k
            const Scalar candid = xi_k - g_k / m_gk_denom[k];
            const Scalar newxi = std::max(Scalar(0), candid);
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

        for (Index i = 0; i < m_n; i++)
        {
            for (Index l = 0; l < m_L; l++)
            {
                const Scalar u_li = m_U(l, i);
                const Scalar v_li = m_V(l, i);
                const Scalar lambda_li = m_Lambda(l, i);

                if (m_gli_denom(l, i) == Scalar(0)) {
                    m_Lambda(l, i) = v_li > Scalar(0) ? Scalar(1) : Scalar(0);
                    continue;
                }
                // Compute g_li
                const Scalar g_li = -(u_li * m_X.dot(i, m_beta) + v_li);
                // Compute new lambda_li
                const Scalar candid = lambda_li - g_li / m_gli_denom(l, i);
                const Scalar newl = std::max(Scalar(0), std::min(Scalar(1), candid));
                // Update Lambda and beta
                m_Lambda(l, i) = newl;
                m_X.subtract_row(m_beta, i, (newl - lambda_li) * u_li);
            }
        }
    }

    // Update Gamma, and beta
    inline void update_Gamma_beta()
    {
        if (m_H < 1)
            return;

        for (Index i = 0; i < m_n; i++)
        {
            for (Index h = 0; h < m_H; h++)
            {
                // tau_hi can be Inf
                const Scalar tau_hi = m_Tau(h, i);
                const Scalar gamma_hi = m_Gamma(h, i);
                const Scalar s_hi = m_S(h, i);
                const Scalar t_hi = m_T(h, i);

                // Compute g_hi
                const Scalar g_hi = gamma_hi - (s_hi * m_X.dot(i, m_beta) + t_hi);
                // Compute new gamma_hi
                const Scalar candid = gamma_hi - g_hi / m_ghi_denom(h, i);
                const Scalar newg = std::max(Scalar(0), std::min(tau_hi, candid));
                // Update Gamma and beta
                m_Gamma(h, i) = newg;
                m_X.subtract_row(m_beta, i, (newg - gamma_hi) * s_hi);
            }
        }
    }

    // Update mu and beta
    inline void update_mu_beta()
    {
        if (m_W <= 0)
            return;

        // Save original Mu
        const Vector preMu = m_mu;
        // Compute new Mu
        const Vector candid = preMu - m_beta * Scalar(0.5);
        const Vector newMu = candid.cwiseMax(Scalar(0.0)).cwiseMin(m_rho);
        // Update Mu and beta
        m_mu = newMu;
        m_beta.noalias() += Scalar(2.0) * (newMu - preMu);
    }

    // =================== Updating functions (free variable set) ================ //

    // Determine whether to shrink xi, and compute the projected gradient (PG)
    // Shrink if xi=0 and grad>ub
    // PG is zero if xi=0 and grad>=0
    inline bool pg_xi(Scalar xi, Scalar grad, Scalar ub, Scalar& pg) const
    {
        pg = (xi == Scalar(0) && grad >= Scalar(0)) ? Scalar(0) : grad;
        const bool shrink = (xi == Scalar(0)) && (grad > ub);
        return shrink;
    }
    // Update xi and beta
    // Overloaded version based on free variable set
    inline void update_xi_beta(std::vector<Index>& fv_set, Scalar& min_pg, Scalar& max_pg)
    {
        if (m_K < 1)
            return;

        // Permutation
        internal::random_shuffle(fv_set.begin(), fv_set.end(), m_rng);
        // New free variable set
        std::vector<Index> new_set;
        new_set.reserve(fv_set.size());

        // Compute shrinking threshold ub
        // ub is kept unchanged in each outer iteration,
        // and is determined by the maximum PG in the previous outer iteration (i.e., max_pg)
        // If the input max_pg is zero or negative, let ub be Inf (do not shrink)
        // This happens when:
        //     (1) max_pg is initialized to be zero in the first iteration
        //     (2) max_pg is negative, thus not meaningful
        constexpr Scalar Inf = std::numeric_limits<Scalar>::infinity();
        const Scalar ub = (max_pg > Scalar(0)) ? max_pg : Inf;
        // Compute minimum and maximum projected gradient (PG) for this round (outer iteration)
        min_pg = Inf;
        max_pg = -Inf;
        for (auto k: fv_set)
        {
            const Scalar xi_k = m_xi[k];

            // Compute g_k
            const Scalar g_k = m_A.row(k).dot(m_beta) + m_b[k];
            if (m_gk_denom[k] == Scalar(0)) {
                m_xi[k] = Scalar(0);
                new_set.push_back(k);
                continue;
            }
            // PG and shrink
            Scalar pg;
            const bool shrink = pg_xi(xi_k, g_k, ub, pg);
            if (shrink)
               continue;

            // Update PG bounds
            max_pg = std::max(max_pg, pg);
            min_pg = std::min(min_pg, pg);
            // Compute new xi_k
            const Scalar candid = xi_k - g_k / m_gk_denom[k];
            const Scalar newxi = std::max(Scalar(0), candid);
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
    inline bool pg_lambda(Scalar lambda, Scalar grad, Scalar lb, Scalar ub, Scalar& pg) const
    {
        pg = ((lambda == Scalar(0) && grad >= Scalar(0)) || (lambda == Scalar(1) && grad <= Scalar(0))) ?
             Scalar(0) :
             grad;
        const bool shrink = (lambda == Scalar(0) && grad > ub) || (lambda == Scalar(1) && grad < lb);
        return shrink;
    }
    // Update Lambda and beta
    // Overloaded version based on free variable set
    inline void update_Lambda_beta(std::vector<std::pair<Index, Index>>& fv_set, Scalar& min_pg, Scalar& max_pg)
    {
        if (m_L < 1)
            return;

        // Permutation
        internal::random_shuffle(fv_set.begin(), fv_set.end(), m_rng);
        // New free variable set
        std::vector<std::pair<Index, Index>> new_set;
        new_set.reserve(fv_set.size());

        // Compute shrinking thresholds lb and ub
        // More details explained in update_xi_beta()
        constexpr Scalar Inf = std::numeric_limits<Scalar>::infinity();
        const Scalar lb = (min_pg < Scalar(0)) ? min_pg : -Inf;
        const Scalar ub = (max_pg > Scalar(0)) ? max_pg : Inf;
        // Compute minimum and maximum projected gradient (PG) for this round
        min_pg = Inf;
        max_pg = -Inf;
        for (auto rc: fv_set)
        {
            const Index l = rc.first;
            const Index i = rc.second;

            const Scalar u_li = m_U(l, i);
            const Scalar v_li = m_V(l, i);
            const Scalar lambda_li = m_Lambda(l, i);

            if (m_gli_denom(l, i) == Scalar(0)) {
                m_Lambda(l, i) = v_li > Scalar(0) ? Scalar(1) : Scalar(0);
                new_set.emplace_back(l, i);
                continue;
            }
            // Compute g_li
            const Scalar g_li = -(u_li * m_X.dot(i, m_beta) + v_li);
            // PG and shrink
            Scalar pg;
            const bool shrink = pg_lambda(lambda_li, g_li, lb, ub, pg);
            if (shrink)
                continue;

            // Update PG bounds
            max_pg = std::max(max_pg, pg);
            min_pg = std::min(min_pg, pg);
            // Compute new lambda_li
            const Scalar candid = lambda_li - g_li / m_gli_denom(l, i);
            const Scalar newl = std::max(Scalar(0), std::min(Scalar(1), candid));
            // Update Lambda and beta
            m_Lambda(l, i) = newl;
            m_X.subtract_row(m_beta, i, (newl - lambda_li) * u_li);

            // Add to new free variable set
            new_set.emplace_back(l, i);
        }

        // Update free variable set
        fv_set.swap(new_set);
    }

    // Determine whether to shrink gamma, and compute the projected gradient (PG)
    // Shrink if (gamma=0 and grad>ub) or (lambda=tau and grad<lb)
    // PG is zero if (lambda=0 and grad>=0) or (lambda=1 and grad<=0)
    inline bool pg_gamma(Scalar gamma, Scalar grad, Scalar tau, Scalar lb, Scalar ub, Scalar& pg) const
    {
        pg = ((gamma == Scalar(0) && grad >= Scalar(0)) || (gamma == tau && grad <= Scalar(0))) ?
             Scalar(0) :
             grad;
        const bool shrink = (gamma == Scalar(0) && grad > ub) || (gamma == tau && grad < lb);
        return shrink;
    }
    // Update Gamma and beta
    // Overloaded version based on free variable set
    inline void update_Gamma_beta(std::vector<std::pair<Index, Index>>& fv_set, Scalar& min_pg, Scalar& max_pg)
    {
        if (m_H < 1)
            return;

        // Permutation
        internal::random_shuffle(fv_set.begin(), fv_set.end(), m_rng);
        // New free variable set
        std::vector<std::pair<Index, Index>> new_set;
        new_set.reserve(fv_set.size());

        // Compute shrinking thresholds lb and ub
        // More details explained in update_xi_beta()
        constexpr Scalar Inf = std::numeric_limits<Scalar>::infinity();
        const Scalar lb = (min_pg < Scalar(0)) ? min_pg : -Inf;
        const Scalar ub = (max_pg > Scalar(0)) ? max_pg : Inf;
        // Compute minimum and maximum projected gradient (PG) for this round
        min_pg = Inf;
        max_pg = -Inf;
        for (auto rc: fv_set)
        {
            const Index h = rc.first;
            const Index i = rc.second;

            // tau_hi can be Inf
            const Scalar tau_hi = m_Tau(h, i);
            const Scalar gamma_hi = m_Gamma(h, i);
            const Scalar s_hi = m_S(h, i);
            const Scalar t_hi = m_T(h, i);

            // Compute g_hi
            const Scalar g_hi = gamma_hi - (s_hi * m_X.dot(i, m_beta) + t_hi);
            // PG and shrink
            Scalar pg;
            const bool shrink = pg_gamma(gamma_hi, g_hi, tau_hi, lb, ub, pg);
            if (shrink)
                continue;

            // Update PG bounds
            max_pg = std::max(max_pg, pg);
            min_pg = std::min(min_pg, pg);
            // Compute new gamma_hi
            const Scalar candid = gamma_hi - g_hi / m_ghi_denom(h, i);
            const Scalar newg = std::max(Scalar(0), std::min(tau_hi, candid));
            // Update Gamma and beta
            m_Gamma(h, i) = newg;
            m_X.subtract_row(m_beta, i, (newg - gamma_hi) * s_hi);

            // Add to new free variable set
            new_set.emplace_back(h, i);
        }

        // Update free variable set
        fv_set.swap(new_set);
    }

    // Determine whether to shrink mu, and compute the projected gradient (PG)
    // Shrink if (mu=0 and grad>ub) or (mu=rho and grad<lb)
    // PG is zero if (mu=0 and grad>=0) or (mu=rho and grad<=0)
    inline bool pg_mu(Scalar mu, Scalar grad, Scalar rho, Scalar lb, Scalar ub, Scalar& pg) const
    {
        pg = ((mu == Scalar(0) && grad >= Scalar(0)) || (mu == rho && grad <= Scalar(0))) ?
             Scalar(0) :
             grad;
        const bool shrink = (mu == Scalar(0) && grad > ub) || (mu == rho && grad < lb);
        return shrink;
    }
    // Update mu and beta
    // Overloaded version based on free variable set
    inline void update_mu_beta(std::vector<Index>& fv_set, Scalar& min_pg, Scalar& max_pg)
    {
        if (m_W <= 0)
            return;

        // Permutation
        internal::random_shuffle(fv_set.begin(), fv_set.end(), m_rng);
        // New free variable set
        std::vector<Index> new_set;
        new_set.reserve(fv_set.size());

        // Compute shrinking thresholds lb and ub
        // More details explained in update_xi_beta()
        constexpr Scalar Inf = std::numeric_limits<Scalar>::infinity();
        const Scalar lb = (min_pg < Scalar(0)) ? min_pg : -Inf;
        const Scalar ub = (max_pg > Scalar(0)) ? max_pg : Inf;
        // Compute minimum and maximum projected gradient (PG) for this round
        min_pg = Inf;
        max_pg = -Inf;
        for (auto j: fv_set)
        {
            const Scalar mu_j = m_mu[j];
            const Scalar rho_j = m_rho[j];

            // Compute g_j
            const Scalar g_j = m_beta[j];
            // PG and shrink
            Scalar pg;
            const bool shrink = pg_mu(mu_j, g_j, rho_j, lb, ub, pg);
            if (shrink)
               continue;

            // Update PG bounds
            max_pg = std::max(max_pg, pg);
            min_pg = std::min(min_pg, pg);
            // Compute new mu_j
            const Scalar candid = mu_j - g_j * Scalar(0.5);
            const Scalar newmu = std::max(Scalar(0), std::min(rho_j, candid));
            // Update mu and beta
            m_mu[j] = newmu;
            m_beta[j] += Scalar(2.0) * (newmu - mu_j);

            // Add to new free variable set
            new_set.push_back(j);
        }
        // Update free variable set
        fv_set.swap(new_set);
    }
public:
    ReHLineSolver(ConstRefMat X, ConstRefMat U, ConstRefMat V,
                  ConstRefMat S, ConstRefMat T, ConstRefMat Tau,
                  ConstRefMat A, ConstRefVec b,
                  ConstRefVec rho, Index quantile_count = 0) :
        m_n(X.rows() * (Composite ? quantile_count : 1)), m_d(X.cols() + (Composite ? quantile_count : 0)), m_L(U.rows()), m_H(S.rows()), m_K(A.rows()),
        m_W(rho.rows()), // check if l1 penalty is implemented
        m_X(X, quantile_count), m_U(U), m_V(V), m_S(S), m_T(T), m_Tau(Tau), m_A(A), m_b(b),
        m_rho(rho),
        m_gk_denom(m_K), m_gli_denom(m_L, m_n), m_ghi_denom(m_H, m_n),
        m_beta(m_d),
        m_xi(m_K), m_Lambda(m_L, m_n), m_Gamma(m_H, m_n), m_mu(m_W)
    {
        // A [K x d], K can be zero
        if (m_K > 0)
            m_gk_denom.noalias() = m_A.rowwise().squaredNorm();

        Vector xi2 = m_X.squared_norms();
        if (m_L > 0)
        {
            m_gli_denom.array() = m_U.array().square().rowwise() * xi2.transpose().array();
        }

        if (m_H > 0)
        {
            m_ghi_denom.array() = m_S.array().square().rowwise() * xi2.transpose().array() + Scalar(1);
        }
    }

    // Initialize primal and dual variables
    inline void init_params()
    {
        // Start at zero duals to avoid large initial beta with imbalanced weights.
        if (m_K > 0)
            m_xi.setZero();

        // Each element of Lambda satisfies 0 <= lambda_li <= 1,
        // and we initialize Lambda at zero.
        if (m_L > 0)
            m_Lambda.setZero();

        // Each element of Gamma satisfies 0 <= gamma_hi <= tau_hi,
        // and zero is feasible even when tau_hi is infinite.
        if (m_H > 0)
        {
            m_Gamma.setZero();
        }

        // Each element of Mu satisfies 0 <= mu_j <= rho_j,
        // and rho / 2 gives a zero initial contribution to beta.
        if (m_W > 0)
            m_mu.noalias() = m_rho * Scalar(0.5);

        // Set primal variable based on duals
        set_primal();
    }

    // Warm start: set dual variables to be the given ones
    inline void warmstart_params(ConstRefVec xi_ws, ConstRefMat Lambda_ws, ConstRefMat Gamma_ws, ConstRefVec mu_ws)
    {
        // Warmstart parameters
        if (m_K > 0)
        {
            // Check shape of warmstart parameters
            if (xi_ws.size() != m_K) {
                throw std::invalid_argument("xi_ws must have size K");
            }
            // Check values of warmstart parameters
            if ((xi_ws.array() < 0).any()) {
                throw std::invalid_argument("xi_ws must be non-negative");
            }
            m_xi = xi_ws;
        }


        if (m_L > 0)
        {
            // Check shape of warmstart parameters
            if (Lambda_ws.rows() != m_L || Lambda_ws.cols() != m_n) {
                throw std::invalid_argument("Lambda_ws must have shape (L, n)");
            }
            // Check values of warmstart parameters
            if ((Lambda_ws.array() < 0).any() || (Lambda_ws.array() > 1).any()) {
                throw std::invalid_argument("Lambda_ws must be in [0, 1]");
            }
            m_Lambda = Lambda_ws;
        }


        if (m_H > 0)
        {
            // Check shape of warmstart parameters
            if (Gamma_ws.rows() != m_H || Gamma_ws.cols() != m_n) {
                throw std::invalid_argument("Gamma_ws must have shape (H, n)");
            }
            // Check values of warmstart parameters
            if ((Gamma_ws.array() < 0).any() || (Gamma_ws.array() > m_Tau.array()).any()) {
                throw std::invalid_argument("Gamma_ws must be in [0, tau_hi]");
            }
            m_Gamma = Gamma_ws;
        }


        if (m_W > 0)
        {
            // Check shape of warmstart parameters
            if (mu_ws.size() != m_d){
                throw std::invalid_argument("mu_ws must have size d");
            }
            // Check values of warmstart parameters
            if ((mu_ws.array() < Scalar(0)).any() || (mu_ws.array() > m_rho.array()).any()){
                throw std::invalid_argument("mu_ws_j must be in [0, rho_j]");
            }
            m_mu = mu_ws;
        }

        // Set primal variable based on duals
        set_primal();
    }

    inline Scalar constraint_violation() const
    {
        if (m_K == 0) return Scalar(0);
        const Vector slack = m_A * m_beta + m_b;
        if (!slack.allFinite()) numerical_failure();
        return std::max(Scalar(0), -slack.minCoeff());
    }

    [[noreturn]] static void numerical_failure()
    {
        throw std::overflow_error(
            "ReHLine numerical overflow or nonfinite computation; "
            "the problem exceeds the solver's floating-point range");
    }

    inline Scalar kkt_residual() const
    {
        if (!m_beta.allFinite() || !m_xi.allFinite() || !m_Lambda.allFinite() ||
            !m_Gamma.allFinite() || !m_mu.allFinite())
            numerical_failure();
        const Vector scores = m_X.multiply(m_beta);
        if (!scores.allFinite()) numerical_failure();
        Scalar residual = m_primal_polished ?
            (m_beta - m_dual_beta).cwiseAbs().maxCoeff() : Scalar(0);
        if (!std::isfinite(residual)) numerical_failure();
        for (Index k = 0; k < m_K; ++k) {
            Scalar g = m_A.row(k).dot(m_beta) + m_b[k];
            if (!std::isfinite(g)) numerical_failure();
            if (m_xi[k] == Scalar(0)) g = std::min(Scalar(0), g);
            residual = std::max(residual, std::abs(g));
        }
        for (Index i = 0; i < m_n; ++i) {
            for (Index l = 0; l < m_L; ++l) {
                Scalar g = -(m_U(l, i) * scores[i] + m_V(l, i));
                if (!std::isfinite(g)) numerical_failure();
                if (m_Lambda(l, i) == Scalar(0)) g = std::min(Scalar(0), g);
                if (m_Lambda(l, i) == Scalar(1)) g = std::max(Scalar(0), g);
                residual = std::max(residual, std::abs(g));
            }
            for (Index h = 0; h < m_H; ++h) {
                if (m_Tau(h, i) == Scalar(0)) continue;
                Scalar g = m_Gamma(h, i) - (m_S(h, i) * scores[i] + m_T(h, i));
                if (!std::isfinite(g)) numerical_failure();
                if (m_Gamma(h, i) == Scalar(0)) g = std::min(Scalar(0), g);
                if (m_Gamma(h, i) == m_Tau(h, i)) g = std::max(Scalar(0), g);
                residual = std::max(residual, std::abs(g));
            }
        }
        for (Index j = 0; j < m_W; ++j) {
            if (m_rho[j] == Scalar(0)) continue;
            Scalar g = m_beta[j];
            if (m_mu[j] == Scalar(0)) g = std::min(Scalar(0), g);
            if (m_mu[j] == m_rho[j]) g = std::max(Scalar(0), g);
            residual = std::max(residual, std::abs(g));
        }
        return residual;
    }

    // A small projected gradient alone is not an objective certificate: its
    // units depend on the dual coordinates and multipliers can be large.
    inline bool certificate(Scalar tol) const
    {
        if (kkt_residual() > tol || constraint_violation() > tol) return false;
        const Scalar primal = primal_objfn(), dual = -dual_objfn();
        const Scalar gap = primal - dual;
        if (!std::isfinite(primal) || !std::isfinite(dual) || !std::isfinite(gap))
            numerical_failure();
        const Scalar scale = std::max(Scalar(1), std::max(std::abs(primal), std::abs(dual)));
        return std::abs(gap) / scale <= tol;
    }

    // Compensated products/sums retain small residuals under cancellation.
    inline Vector recover_dual_accurately() const
    {
        Vector beta(m_d);
        for (Index j = 0; j < m_d; ++j) {
            Scalar total = Scalar(0), compensation = Scalar(0);
            const auto add = [&](Scalar value) {
                const Scalar next = total + value;
                compensation += std::abs(total) >= std::abs(value) ?
                    (total - next) + value : (value - next) + total;
                total = next;
            };
            const auto product = [&](Scalar a, Scalar b) {
                const Scalar value = a * b;
                add(value);
                add(std::fma(a, b, -value));
            };
            const auto triple = [&](Scalar a, Scalar b, Scalar c) {
                const Scalar bc = b * c;
                product(a, bc);
                product(a, std::fma(b, c, -bc));
            };
            for (Index k = 0; k < m_K; ++k) product(m_A(k, j), m_xi[k]);
            for (Index i = 0; i < m_n; ++i) {
                for (Index l = 0; l < m_L; ++l) triple(-m_X.coeff(i, j), m_U(l, i), m_Lambda(l, i));
                for (Index h = 0; h < m_H; ++h) triple(-m_X.coeff(i, j), m_S(h, i), m_Gamma(h, i));
            }
            if (m_W > 0) { product(Scalar(2), m_mu[j]); add(-m_rho[j]); }
            beta[j] = total + compensation;
        }
        return beta;
    }

    // Primal refinement near the precision floor; the caller has recovered the
    // dual coefficient vector. Only accept a correction passing the original
    // tolerance, including the formerly implicit stationarity equation.
    inline bool polish_primal(Scalar tol)
    {
        if (m_K == 0) return false;
        std::vector<Index> active;
        for (Index k = 0; k < m_K; ++k)
            if (m_xi[k] > Scalar(0) &&
                std::abs(m_A.row(k).dot(m_beta) + m_b[k]) <= tol)
                active.push_back(k);
        if (active.empty()) return false;
        Matrix equalities(active.size(), m_d);
        Vector rhs(active.size());
        for (Index i = 0; i < static_cast<Index>(active.size()); ++i) {
            equalities.row(i) = m_A.row(active[i]);
            rhs[i] = -(m_A.row(active[i]).dot(m_beta) + m_b[active[i]]);
        }
        const Vector correction = equalities.completeOrthogonalDecomposition().solve(rhs);
        if (!correction.allFinite()) return false;
        const Vector recovered = m_beta;
        m_dual_beta = recover_dual_accurately();
        if (!m_dual_beta.allFinite()) return false;
        m_beta += correction;
        m_primal_polished = true;
        if (certificate(tol))
            return true;
        m_beta = recovered;
        m_primal_polished = false;
        return false;
    }

    inline void diagnostics(ReHLineResult<Matrix, Index>& result, Scalar tol)
    {
        result.converged = false;
        result.dual_gap = std::numeric_limits<Scalar>::infinity();
        if (!m_primal_polished) set_primal();
        result.objective = primal_objfn();
        result.dual_objective = -dual_objfn();
        result.constraint_violation = constraint_violation();
        result.scaled_constraint_violation = result.constraint_violation;
        result.kkt_residual = kkt_residual();
        if (!std::isfinite(result.objective) || !std::isfinite(result.dual_objective) ||
            !std::isfinite(result.constraint_violation) || !std::isfinite(result.kkt_residual))
            numerical_failure();
        const Scalar gap = result.objective - result.dual_objective;
        if (!std::isfinite(gap)) numerical_failure();
        const Scalar scale = std::max(Scalar(1), std::max(std::abs(result.objective), std::abs(result.dual_objective)));
        result.converged = result.kkt_residual <= tol && result.constraint_violation <= tol &&
            std::abs(gap) / scale <= tol;
        // For infeasible iterates the unconstrained loss is not a primal bound.
        result.dual_gap = result.constraint_violation > tol ?
            std::numeric_limits<Scalar>::infinity() :
            std::max(Scalar(0), gap);
    }

    // Correlated coordinates can make individual updates arbitrarily slow.
    // Periodically minimize a small block of the same dual quadratic. Bounds
    // and an explicit descent check keep this acceleration feasible and monotone.
    inline bool update_block()
    {
        struct Coordinate {
            Index kind, row, col;
            Scalar value, upper, gradient;
        };
        std::vector<Coordinate> candidates;
        const Vector scores = m_X.multiply(m_beta);
        const auto add = [&](Index kind, Index row, Index col, Scalar value, Scalar upper, Scalar gradient) {
            if (upper == Scalar(0) || (value == Scalar(0) && gradient > Scalar(0)) ||
                (value == upper && gradient < Scalar(0))) return;
            candidates.push_back({kind, row, col, value, upper, gradient});
        };
        for (Index k = 0; k < m_K; ++k)
            add(0, k, 0, m_xi[k], std::numeric_limits<Scalar>::infinity(), m_A.row(k).dot(m_beta) + m_b[k]);
        for (Index i = 0; i < m_n; ++i) {
            for (Index l = 0; l < m_L; ++l)
                add(1, l, i, m_Lambda(l, i), Scalar(1), -(m_U(l, i) * scores[i] + m_V(l, i)));
            for (Index h = 0; h < m_H; ++h)
                add(2, h, i, m_Gamma(h, i), m_Tau(h, i), m_Gamma(h, i) - m_S(h, i) * scores[i] - m_T(h, i));
        }
        for (Index j = 0; j < m_W; ++j)
            add(3, j, 0, m_mu[j], m_rho[j], Scalar(2) * m_beta[j]);
        // ReHU blocks may need many simultaneous quadratic coordinates. For
        // piecewise-linear losses, use a smaller block tied to primal dimension.
        const Index limit = m_H > 0 ? Index(256) :
            std::min(Index(128), std::max(Index(32), 2 * m_d + m_K + m_W));
        if (candidates.size() > static_cast<std::size_t>(limit)) {
            std::partial_sort(candidates.begin(), candidates.begin() + limit, candidates.end(),
                [](const Coordinate& a, const Coordinate& b) {
                    const bool a_penalty = a.kind == 0 || a.kind == 3;
                    const bool b_penalty = b.kind == 0 || b.kind == 3;
                    if (a_penalty != b_penalty) return a_penalty;
                    return std::abs(a.gradient) > std::abs(b.gradient);
                });
            candidates.resize(limit);
        }
        Index count = static_cast<Index>(candidates.size());
        if (count == 0) return false;
        Matrix B(m_d, count);
        Vector gradient(count), diagonal = Vector::Zero(count);
        for (Index k = 0; k < count; ++k) {
            const auto& c = candidates[k];
            gradient[k] = c.gradient;
            if (c.kind == 0) B.col(k) = m_A.row(c.row).transpose();
            else if (c.kind == 1) m_X.scaled_column(B, k, c.col, -m_U(c.row, c.col));
            else if (c.kind == 2) {
                m_X.scaled_column(B, k, c.col, -m_S(c.row, c.col));
                diagonal[k] = Scalar(1);
            } else {
                B.col(k).setZero();
                B(c.row, k) = Scalar(2);
            }
        }
        Matrix hessian = B.transpose() * B;
        hessian.diagonal() += diagonal;
        // A tiny positive diagonal also gives a descent direction for singular
        // blocks. The line search below uses the original, unmodified Hessian.
        const Scalar damping = std::max(Scalar(1), hessian.diagonal().maxCoeff()) * Scalar(1e-12);
        std::vector<Index> free(count);
        std::iota(free.begin(), free.end(), Index(0));
        Vector direction = Vector::Zero(count);
        while (!free.empty()) {
            const Index size = static_cast<Index>(free.size());
            Matrix reduced(size, size);
            Vector rhs(size);
            for (Index i = 0; i < size; ++i) {
                rhs[i] = -gradient[free[i]];
                for (Index j = 0; j < size; ++j) reduced(i, j) = hessian(free[i], free[j]);
            }
            reduced.diagonal().array() += damping;
            Eigen::LDLT<Matrix> factor(reduced);
            if (factor.info() != Eigen::Success) return false;
            const Vector step = factor.solve(rhs);
            if (!step.allFinite()) return false;
            direction.setZero();
            std::vector<Index> next;
            for (Index i = 0; i < size; ++i) {
                const Index k = free[i];
                const auto& c = candidates[k];
                if ((c.value == Scalar(0) && step[i] < Scalar(0)) ||
                    (c.value == c.upper && step[i] > Scalar(0))) continue;
                direction[k] = step[i];
                next.push_back(k);
            }
            if (next.size() == free.size()) break;
            free.swap(next);
        }
        const Scalar slope = gradient.dot(direction);
        const Scalar curvature = direction.dot(hessian * direction);
        if (!(slope < Scalar(0)) || !std::isfinite(curvature)) return false;
        Scalar alpha = curvature > Scalar(0) ? -slope / curvature : Scalar(1);
        for (Index k = 0; k < count; ++k) {
            const auto& c = candidates[k];
            if (direction[k] > Scalar(0)) alpha = std::min(alpha, (c.upper - c.value) / direction[k]);
            else if (direction[k] < Scalar(0)) alpha = std::min(alpha, -c.value / direction[k]);
        }
        if (!(alpha > Scalar(0)) || !std::isfinite(alpha)) return false;
        Vector delta(count), values(count);
        for (Index k = 0; k < count; ++k) {
            const auto& c = candidates[k];
            values[k] = std::max(Scalar(0), std::min(c.upper, c.value + alpha * direction[k]));
            // Snap the limiting coordinate exactly to its bound, preventing a
            // tiny rounding remainder from blocking subsequent Newton steps.
            if (direction[k] < Scalar(0) && alpha >= -c.value / direction[k]) values[k] = Scalar(0);
            if (direction[k] > Scalar(0) && alpha >= (c.upper - c.value) / direction[k]) values[k] = c.upper;
            delta[k] = values[k] - c.value;
        }
        const Scalar change = gradient.dot(delta) + Scalar(0.5) * delta.dot(hessian * delta);
        if (!std::isfinite(change) || change >= Scalar(0)) return false;
        for (Index k = 0; k < count; ++k) {
            const auto& c = candidates[k];
            if (c.kind == 0) m_xi[c.row] = values[k];
            else if (c.kind == 1) m_Lambda(c.row, c.col) = values[k];
            else if (c.kind == 2) m_Gamma(c.row, c.col) = values[k];
            else m_mu[c.row] = values[k];
        }
        set_primal();
        return true;
    }

    inline void set_seed(Index seed) { m_rng.seed(seed); }

    inline Index solve_vanilla(
        std::vector<Scalar>& dual_objfns, std::vector<Scalar>& primal_objfns,
        Index max_iter, Scalar tol,
        Index verbose = 0, Index trace_freq = 100,
        std::ostream& cout = std::cout)
    {
        // Main iterations
        Index i = 0;
        Vector old_xi(m_K), old_beta(m_d);
        // With many samples and constraints, shrinking makes individual sweeps
        // cheap. Space out full block scans to preserve that advantage.
        const Index block_period = m_K > 0 ? std::max(Index(50), m_n) : Index(50);
        for(; i < max_iter; i++)
        {
            old_xi.noalias() = m_xi;
            old_beta.noalias() = m_beta;

            update_xi_beta();
            update_Lambda_beta();
            update_Gamma_beta();
            update_mu_beta();
            if ((i + 1) % block_period == 0)
                for (Index block = 0; block < 64; ++block)
                    if (!update_block() || kkt_residual() <= tol) break;

            // Compute difference of xi and beta
            const Scalar xi_diff = (m_K > 0) ? (m_xi - old_xi).norm() : Scalar(0);
            const Scalar beta_diff = (m_beta - old_beta).norm();

            // Print progress
            if (verbose && (i % trace_freq == 0))
            {
                Scalar dual = dual_objfn();
                dual_objfns.push_back(dual);
                Scalar primal = primal_objfn();
                primal_objfns.push_back(primal);
                cout << "Iter " << i << ", dual_objfn = " << dual <<
                    ", primal_objfn = " << primal <<
                    ", xi_diff = " << xi_diff <<
                    ", beta_diff = " << beta_diff << std::endl;
            }

            // Convergence test based on change of variable values
            const bool vars_conv = (xi_diff < tol) && (beta_diff < tol);
            if (vars_conv) {
                set_primal();
                if (certificate(tol) || polish_primal(tol)) return i + 1;
            }
        }

        return i;
    }

    inline Index solve(
        std::vector<Scalar>& dual_objfns, std::vector<Scalar>& primal_objfns,
        Index max_iter, Scalar tol,
        Index verbose = 0, Index trace_freq = 100,
        std::ostream& cout = std::cout)
    {
        // Free variable sets
        internal::reset_fv_set(m_fv_feas, m_K);
        internal::reset_fv_set(m_fv_relu, m_L, m_n);
        internal::reset_fv_set(m_fv_rehu, m_H, m_n);
        internal::reset_fv_set(m_fv_l1mu, m_W);

        // Minimum and maximum projected gradients of dual variables in each outer iteration
        // These variables will be updated in update_*_beta() functions below
        // If some dual variables are not used, the corresponding pg variables
        // will always be zero, so that the related tests in pg_conv below return true values
        Scalar xi_min_pg = Scalar(0), lambda_min_pg = Scalar(0), gamma_min_pg = Scalar(0), mu_min_pg = Scalar(0);
        Scalar xi_max_pg = Scalar(0), lambda_max_pg = Scalar(0), gamma_max_pg = Scalar(0), mu_max_pg = Scalar(0);

        // Main iterations
        Index i = 0;
        Vector old_xi(m_K), old_beta(m_d);
        const Index block_period = m_K > 0 ? std::max(Index(50), m_n) : Index(50);
        for(; i < max_iter; i++)
        {
            old_xi.noalias() = m_xi;
            old_beta.noalias() = m_beta;

            update_xi_beta(m_fv_feas, xi_min_pg, xi_max_pg);
            update_Lambda_beta(m_fv_relu, lambda_min_pg, lambda_max_pg);
            update_Gamma_beta(m_fv_rehu, gamma_min_pg, gamma_max_pg);
            update_mu_beta(m_fv_l1mu, mu_min_pg, mu_max_pg);
            if ((i + 1) % block_period == 0) {
                for (Index block = 0; block < 64; ++block)
                    if (!update_block() || kkt_residual() <= tol) break;
                internal::reset_fv_set(m_fv_feas, m_K);
                internal::reset_fv_set(m_fv_relu, m_L, m_n);
                internal::reset_fv_set(m_fv_rehu, m_H, m_n);
                internal::reset_fv_set(m_fv_l1mu, m_W);
                xi_min_pg = lambda_min_pg = gamma_min_pg = mu_min_pg = Scalar(0);
                xi_max_pg = lambda_max_pg = gamma_max_pg = mu_max_pg = Scalar(0);
            }

            // Compute difference of xi and beta
            const Scalar xi_diff = (m_K > 0) ? (m_xi - old_xi).norm() : Scalar(0);
            const Scalar beta_diff = (m_beta - old_beta).norm();

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
                                 (std::abs(gamma_min_pg) < tol) &&
                                 (mu_max_pg - mu_min_pg < tol) &&
                                 (std::abs(mu_max_pg) < tol) &&
                                 (std::abs(mu_min_pg) < tol);
            // Whether we are using all variables
            const bool all_vars = (m_fv_feas.size() == static_cast<std::size_t>(m_K)) &&
                                  (m_fv_relu.size() == static_cast<std::size_t>(m_L * m_n)) &&
                                  (m_fv_rehu.size() == static_cast<std::size_t>(m_H * m_n)) &&
                                  (m_fv_l1mu.size() == static_cast<std::size_t>(m_W));

            // Print progress
            if (verbose && (i % trace_freq == 0))
            {
                Scalar dual = dual_objfn();
                dual_objfns.push_back(dual);
                Scalar primal = primal_objfn();
                primal_objfns.push_back(primal);
                cout << "Iter " << i << ", dual_objfn = " << dual <<
                    ", primal_objfn = " << primal <<
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
                internal::reset_fv_set(m_fv_feas, m_K);
                internal::reset_fv_set(m_fv_relu, m_L, m_n);
                internal::reset_fv_set(m_fv_rehu, m_H, m_n);
                internal::reset_fv_set(m_fv_l1mu, m_W);
                xi_min_pg = lambda_min_pg = gamma_min_pg = mu_min_pg = Scalar(0);
                xi_max_pg = lambda_max_pg = gamma_max_pg = mu_max_pg = Scalar(0);
                // Also recompute beta to improve precision
                // set_primal();
                continue;
            }
            if (all_vars && (vars_conv || pg_conv)) {
                set_primal();
                if (certificate(tol) || polish_primal(tol)) return i + 1;
            }
        }

        return i;
    }

    Vector& get_beta_ref() { return m_beta; }
    Vector& get_xi_ref() { return m_xi; }
    Vector& get_mu_ref() { return m_mu; }
    Matrix& get_Lambda_ref() { return m_Lambda; }
    Matrix& get_Gamma_ref() { return m_Gamma; }
};

// Main solver interface
// template <typename Matrix = Eigen::MatrixXd, typename Index = int>
template <typename DerivedMat, typename DerivedVec, typename Index = int, bool Composite = false>
void rehline_solver(
    ReHLineResult<typename DerivedMat::PlainObject, Index>& result,
    const Eigen::MatrixBase<DerivedMat>& X, const Eigen::MatrixBase<DerivedMat>& A,
    const Eigen::MatrixBase<DerivedVec>& b, const Eigen::MatrixBase<DerivedVec>& rho,
    const Eigen::MatrixBase<DerivedMat>& U, const Eigen::MatrixBase<DerivedMat>& V,
    const Eigen::MatrixBase<DerivedMat>& S, const Eigen::MatrixBase<DerivedMat>& T, const Eigen::MatrixBase<DerivedMat>& Tau,
    Index max_iter, typename DerivedMat::Scalar tol, 
    Index shrink = 1, Index verbose = 0, Index trace_freq = 100,
    std::ostream& cout = std::cout, Index quantile_count = 0
)
{
    if (Composite && (quantile_count <= 0 || X.rows() > std::numeric_limits<Index>::max() / quantile_count ||
        X.cols() > std::numeric_limits<Index>::max() - quantile_count))
        throw std::invalid_argument("Invalid composite quantile dimensions");
    const auto n = X.rows() * (Composite ? quantile_count : 1);
    const auto d = X.cols() + (Composite ? quantile_count : 0);
    if (n <= 0 || d <= 0 || max_iter <= 0 || !std::isfinite(tol) || tol <= 0 ||
        shrink < 0 || verbose < 0 || trace_freq <= 0)
        throw std::invalid_argument("Invalid dimensions or solver options");
    if (!X.allFinite() || !A.allFinite() || !b.allFinite() || !rho.allFinite() ||
        !U.allFinite() || !V.allFinite() || !S.allFinite() || !T.allFinite())
        throw std::invalid_argument("Solver inputs must be finite");
    if ((U.rows() > 0 && U.cols() != n) || U.rows() != V.rows() || U.cols() != V.cols() ||
        (S.rows() > 0 && S.cols() != n) || S.rows() != T.rows() || S.cols() != T.cols() ||
        S.rows() != Tau.rows() || S.cols() != Tau.cols() ||
        (A.rows() > 0 && A.cols() != d) || b.size() != A.rows() ||
        (rho.size() != 0 && rho.size() != d))
        throw std::invalid_argument("Inconsistent solver input shapes");
    if (!(Tau.array() >= 0).all() || !(rho.array() >= 0).all())
        throw std::invalid_argument("Tau and rho must be non-negative");
    // Positive row scaling preserves the feasible set. Divide by the largest
    // coefficient, without squaring it, to avoid both underflow and overflow.
    // Public xi and constraint_violation retain the caller's original units.
    using Scalar = typename DerivedMat::Scalar;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    typename DerivedMat::PlainObject scaled_A = A;
    Vector scaled_b = b, row_scale(A.rows());
    for (Index k = 0; k < A.rows(); ++k) {
        const Scalar scale = A.row(k).cwiseAbs().maxCoeff();
        if (scale == Scalar(0) && b[k] < 0)
            throw std::invalid_argument("Infeasible zero constraint row");
        row_scale[k] = scale == Scalar(0) ? Scalar(1) : scale;
        scaled_A.row(k) /= row_scale[k];
        scaled_b[k] /= row_scale[k];
    }
    if (!scaled_b.allFinite())
        ReHLineSolver<typename DerivedMat::PlainObject, Index, Composite>::numerical_failure();

    // Create solver
    ReHLineSolver<typename DerivedMat::PlainObject, Index, Composite> solver(
        X, U, V, S, T, Tau, scaled_A, scaled_b, rho, quantile_count);

    // Initialize parameters
    try {
        // Warm start parameters: if result contains warm start parameters then warm start
        if (result.xi.size() > 0 || result.Lambda.size() > 0 || result.Gamma.size() > 0 || result.mu.size() > 0) {
            Vector scaled_xi = result.xi;
            if (scaled_xi.size() == row_scale.size())
                scaled_xi.array() *= row_scale.array();
            if (!scaled_xi.allFinite()) solver.numerical_failure();
            solver.warmstart_params(scaled_xi, result.Lambda, result.Gamma, result.mu);
        } else {
            solver.init_params();
        }
    } catch (const std::overflow_error&) {
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Warning: warmstart_params failed, using default initialization. Error: " << e.what() << std::endl;
        solver.init_params();
    }

    // Main iterations
    std::vector<typename DerivedMat::Scalar> dual_objfns;
    std::vector<typename DerivedMat::Scalar> primal_objfns;
    Index niter;
    if (shrink > 0)
    {
        solver.set_seed(shrink);
        niter = solver.solve(dual_objfns, primal_objfns, max_iter, tol, verbose, trace_freq, cout);
    } else {
        niter = solver.solve_vanilla(dual_objfns, primal_objfns, max_iter, tol, verbose, trace_freq, cout);
    }

    solver.diagnostics(result, tol);

    Vector raw_slack = b;
    if (A.rows() > 0) raw_slack.noalias() = A * solver.get_beta_ref() + b;
    const Vector raw_xi = solver.get_xi_ref().cwiseQuotient(row_scale);
    if (!raw_slack.allFinite() || !raw_xi.allFinite()) solver.numerical_failure();
    result.constraint_violation = A.rows() == 0 ? Scalar(0) : std::max(Scalar(0), -raw_slack.minCoeff());

    // Save result
    result.beta.swap(solver.get_beta_ref());
    result.xi = raw_xi;
    result.mu.swap(solver.get_mu_ref());
    result.Lambda.swap(solver.get_Lambda_ref());
    result.Gamma.swap(solver.get_Gamma_ref());
    result.niter = niter;
    result.dual_objfns.swap(dual_objfns);
    result.primal_objfns.swap(primal_objfns);
}


}  // namespace rehline


#endif  // REHLINE_H
