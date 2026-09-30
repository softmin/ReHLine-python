#ifndef REHLINE_DESIGN_H
#define REHLINE_DESIGN_H

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <type_traits>

namespace rehline {
namespace internal {

// Matrix operations needed by the common solver. The dense specialization
// preserves the original Eigen expressions, without per-coordinate branching.
template <typename Matrix, typename Index, bool Composite,
          bool Sparse = std::is_same<typename Matrix::StorageKind, Eigen::Sparse>::value>
class Design;

template <typename Matrix, typename Index>
class Design<Matrix, Index, false, false>
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
    Vector transpose_multiply(const Vector& weight) const { return m_X.transpose() * weight; }
    void add_row(Vector& beta, Index row, Scalar scale) const
    { beta.noalias() += scale * m_X.row(row).transpose(); }
    template <typename Function>
    void for_each_in_row(Index row, const Function& visit) const
    {
        for (Index j = 0; j < m_X.cols(); ++j) visit(j, m_X(row, j));
    }
    void subtract_transpose(Vector& beta, const Vector& weight) const
    { beta.noalias() -= m_X.transpose() * weight; }
    Scalar dot(Index row, const Vector& beta) const { return m_X.row(row).dot(beta); }
    void subtract_row(Vector& beta, Index row, Scalar scale) const
    { beta.noalias() -= scale * m_X.row(row).transpose(); }
    Vector squared_norms() const { return m_X.rowwise().squaredNorm(); }
    Scalar coeff(Index row, Index col) const { return m_X(row, col); }
};

// CQR's virtual row (q*n + i) is [X[i], e_q]. Only X is stored. Loss and dual
// arrays retain their quantile-major ordering, and all solver updates, KKT
// checks and warm starts use the same joint formulation.
template <typename Matrix, typename Index>
class Design<Matrix, Index, true, false>
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
    template <typename Function>
    void for_each_in_row(Index row, const Function& visit) const
    {
        for (Index j = 0; j < m_d; ++j) visit(j, m_X(row % m_n, j));
        visit(m_d + row / m_n, Scalar(1));
    }
    Scalar coeff(Index row, Index col) const
    { return col < m_d ? m_X(row % m_n, col) : Scalar(col - m_d == row / m_n); }
};

// CSR rows support both ordinary and implicit CQR designs. Only stored entries
// participate in coordinate updates; no dense row or expanded CQR X is formed.
template <typename Matrix, typename Index, bool Composite>
class Design<Matrix, Index, Composite, true>
{
    using Scalar = typename Matrix::Scalar;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using RowMatrix = Eigen::SparseMatrix<Scalar, Eigen::RowMajor, typename Matrix::StorageIndex>;
    using Storage = typename std::conditional<Matrix::IsRowMajor,
        Eigen::Ref<const Matrix>, RowMatrix>::type;
    Storage m_X;
    const Index m_n, m_d, m_q;
public:
    Design(Eigen::Ref<const Matrix> X, Index q) :
        m_X(X), m_n(X.rows()), m_d(X.cols()), m_q(Composite ? q : 1) {}
    Vector multiply(const Vector& beta) const
    {
        const Vector shared = m_X * beta.head(m_d);
        if (!Composite) return shared;
        Vector scores(m_n * m_q);
        for (Index q = 0; q < m_q; ++q)
            scores.segment(q * m_n, m_n) = shared.array() + beta[m_d + q];
        return scores;
    }
    void subtract_transpose(Vector& beta, const Vector& weight) const
    {
        if (!Composite) {
            beta.noalias() -= m_X.transpose() * weight;
            return;
        }
        Vector shared = Vector::Zero(m_n);
        for (Index q = 0; q < m_q; ++q) {
            shared += weight.segment(q * m_n, m_n);
            beta[m_d + q] -= weight.segment(q * m_n, m_n).sum();
        }
        beta.head(m_d).noalias() -= m_X.transpose() * shared;
    }
    template <typename Function>
    void for_each_in_row(Index row, const Function& visit) const
    {
        const Index actual = Composite ? row % m_n : row;
        for (typename Storage::InnerIterator it(m_X, actual); it; ++it)
            visit(it.col(), it.value());
        if (Composite) visit(m_d + row / m_n, Scalar(1));
    }
    Scalar dot(Index row, const Vector& beta) const
    {
        Scalar value = Scalar(0);
        for_each_in_row(row, [&](Index j, Scalar x) { value += x * beta[j]; });
        return value;
    }
    Vector transpose_multiply(const Vector& weight) const { return m_X.transpose() * weight; }
    void add_row(Vector& beta, Index row, Scalar scale) const
    { for_each_in_row(row, [&](Index j, Scalar x) { beta[j] += scale * x; }); }
    void subtract_row(Vector& beta, Index row, Scalar scale) const
    { for_each_in_row(row, [&](Index j, Scalar x) { beta[j] -= scale * x; }); }
    Vector squared_norms() const
    {
        Vector norms(m_n * m_q);
        for (Index i = 0; i < m_n; ++i) {
            Scalar value = Scalar(0);
            for_each_in_row(i, [&](Index, Scalar x) { value += x * x; });
            for (Index q = 0; q < m_q; ++q) norms[q * m_n + i] = value;
        }
        return norms;
    }
};

// Constraints are normalized in row-major storage, without changing sparsity.
template <typename Matrix,
          bool Sparse = std::is_same<typename Matrix::StorageKind, Eigen::Sparse>::value>
struct ConstraintStorage {
    using Type = Eigen::Matrix<typename Matrix::Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
};
template <typename Matrix>
struct ConstraintStorage<Matrix, true> {
    using Type = Eigen::SparseMatrix<typename Matrix::Scalar, Eigen::RowMajor, typename Matrix::StorageIndex>;
};

template <typename Matrix, typename Vector>
Vector normalize_constraint_rows(Eigen::MatrixBase<Matrix>& A, Vector& b)
{
    using Scalar = typename Matrix::Scalar;
    Vector scales(A.rows());
    for (Eigen::Index k = 0; k < A.rows(); ++k) {
        const Scalar scale = A.row(k).cwiseAbs().maxCoeff();
        if (scale == Scalar(0) && b[k] < 0)
            throw std::invalid_argument("Infeasible zero constraint row");
        scales[k] = scale == Scalar(0) ? Scalar(1) : scale;
        A.row(k) /= scales[k];
        b[k] /= scales[k];
    }
    return scales;
}

template <typename Matrix, typename Vector>
Vector normalize_constraint_rows(Eigen::SparseMatrixBase<Matrix>& A, Vector& b)
{
    using Scalar = typename Matrix::Scalar;
    Vector scales(A.rows());
    for (Eigen::Index k = 0; k < A.rows(); ++k) {
        Scalar scale = Scalar(0);
        for (typename Matrix::InnerIterator it(A.derived(), k); it; ++it)
            scale = std::max(scale, std::abs(it.value()));
        if (scale == Scalar(0) && b[k] < 0)
            throw std::invalid_argument("Infeasible zero constraint row");
        scales[k] = scale == Scalar(0) ? Scalar(1) : scale;
        // Divide directly: computing 1/scale first can overflow for tiny rows.
        for (typename Matrix::InnerIterator it(A.derived(), k); it; ++it)
            it.valueRef() /= scales[k];
        b[k] /= scales[k];
    }
    return scales;
}

template <typename Derived>
bool all_finite(const Eigen::MatrixBase<Derived>& X) { return X.allFinite(); }

template <typename Derived>
bool all_finite(const Eigen::SparseMatrixBase<Derived>& X)
{
    for (Eigen::Index row = 0; row < X.outerSize(); ++row)
        for (typename Derived::InnerIterator it(X.derived(), row); it; ++it)
            if (!std::isfinite(it.value())) return false;
    return true;
}

} // namespace internal
} // namespace rehline
#endif
