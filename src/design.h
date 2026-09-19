#ifndef REHLINE_DESIGN_H
#define REHLINE_DESIGN_H

#include <Eigen/Core>
#include <type_traits>

namespace rehline {
namespace internal {

// Matrix operations needed by the common solver. The dense specialization
// preserves the original Eigen expressions, without per-coordinate branching.
template <typename Matrix, typename Index, bool Composite>
class Design;

template <typename Matrix, typename Index>
class Design<Matrix, Index, false>
{
    using Scalar = typename Matrix::Scalar;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Storage = typename std::conditional<Matrix::IsRowMajor,
        Eigen::Ref<const Matrix>,
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>::type;
    Storage m_X;
public:
    Design(Eigen::Ref<const Matrix> X, Index) : m_X(X) {}
    Vector multiply(const Vector& beta) const { return m_X * beta; }
    void subtract_transpose(Vector& beta, const Vector& weight) const
    { beta.noalias() -= m_X.transpose() * weight; }
    Scalar dot(Index row, const Vector& beta) const { return m_X.row(row).dot(beta); }
    void subtract_row(Vector& beta, Index row, Scalar scale) const
    { beta.noalias() -= scale * m_X.row(row).transpose(); }
    Vector squared_norms() const { return m_X.rowwise().squaredNorm(); }
    Scalar coeff(Index row, Index col) const { return m_X(row, col); }
    void scaled_column(Matrix& out, Index col, Index row, Scalar scale) const
    { out.col(col) = scale * m_X.row(row).transpose(); }
};

// CQR's virtual row (q*n + i) is [X[i], e_q]. Only X is stored. Loss and dual
// arrays retain their quantile-major ordering, and all solver updates, KKT
// checks, block acceleration and warm starts use the same joint formulation.
template <typename Matrix, typename Index>
class Design<Matrix, Index, true>
{
    using Scalar = typename Matrix::Scalar;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    Eigen::Ref<const Matrix> m_X;
    const Index m_n, m_d, m_q;
public:
    Design(Eigen::Ref<const Matrix> X, Index q) : m_X(X), m_n(X.rows()), m_d(X.cols()), m_q(q) {}
    Vector multiply(const Vector& beta) const
    {
        const Vector shared = m_X * beta.head(m_d);
        Vector scores(m_n * m_q);
        for (Index q = 0; q < m_q; ++q)
            scores.segment(q * m_n, m_n) = shared.array() + beta[m_d + q];
        return scores;
    }
    void subtract_transpose(Vector& beta, const Vector& weight) const
    {
        Vector shared = Vector::Zero(m_n);
        for (Index q = 0; q < m_q; ++q) {
            shared += weight.segment(q * m_n, m_n);
            beta[m_d + q] -= weight.segment(q * m_n, m_n).sum();
        }
        beta.head(m_d).noalias() -= m_X.transpose() * shared;
    }
    Scalar dot(Index row, const Vector& beta) const
    { return m_X.row(row % m_n).dot(beta.head(m_d)) + beta[m_d + row / m_n]; }
    void subtract_row(Vector& beta, Index row, Scalar scale) const
    {
        beta.head(m_d).noalias() -= scale * m_X.row(row % m_n).transpose();
        beta[m_d + row / m_n] -= scale;
    }
    Vector squared_norms() const
    {
        const Vector norms = m_X.rowwise().squaredNorm().array() + Scalar(1);
        return norms.replicate(m_q, 1);
    }
    Scalar coeff(Index row, Index col) const
    { return col < m_d ? m_X(row % m_n, col) : Scalar(col - m_d == row / m_n); }
    void scaled_column(Matrix& out, Index col, Index row, Scalar scale) const
    {
        out.col(col).setZero();
        out.col(col).head(m_d) = scale * m_X.row(row % m_n).transpose();
        out(m_d + row / m_n, col) = scale;
    }
};

} // namespace internal
} // namespace rehline
#endif
