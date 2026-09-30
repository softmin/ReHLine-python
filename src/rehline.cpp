#include <vector>
#include <type_traits>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include "rehline.h"

namespace py = pybind11;

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MapMat = Eigen::Ref<const Matrix>;
using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
using Vector = Eigen::VectorXd;
using MapVec = Eigen::Ref<const Vector>;

using ReHLineResult = rehline::ReHLineResult<Matrix>;

template <typename XMatrix, typename AMatrix = MapMat>
void rehline_internal(
    ReHLineResult& result,
    const XMatrix& X, const AMatrix& A, const MapVec& b, const MapVec& rho,
    const MapMat& U, const MapMat& V,
    const MapMat& S, const MapMat& T, const MapMat& Tau,
    int max_iter, double tol, int shrink = 1,
    int verbose = 0, int trace_freq = 100,
    int coordinate_order = 0, int coordinate_seed = -1
)
{
    rehline::rehline_solver(result, X, A, b, rho, U, V, S, T, Tau,
                            max_iter, tol, shrink, verbose, trace_freq, std::cout, 0,
                            coordinate_order, coordinate_seed);
}

template <typename XMatrix, typename AMatrix = MapMat>
void rehline_cqr_internal(
    ReHLineResult& result,
    const XMatrix& X, const AMatrix& A, const MapVec& b, const MapVec& rho,
    const MapMat& U, const MapMat& V,
    const MapMat& S, const MapMat& T, const MapMat& Tau,
    int quantile_count, int max_iter, double tol, int shrink = 1,
    int verbose = 0, int trace_freq = 100,
    int coordinate_order = 0, int coordinate_seed = -1
)
{
    rehline::rehline_solver<MapMat, MapVec, int, true>(
        result, X, A, b, rho, U, V, S, T, Tau,
        max_iter, tol, shrink, verbose, trace_freq, std::cout, quantile_count,
        coordinate_order, coordinate_seed);
}

PYBIND11_MODULE(_internal, m) {
    py::class_<ReHLineResult>(m, "rehline_result")
        .def(py::init<>())
        .def_readwrite("beta",          &ReHLineResult::beta)
        .def_readwrite("xi",            &ReHLineResult::xi)
        .def_readwrite("Lambda",        &ReHLineResult::Lambda)
        .def_readwrite("Gamma",         &ReHLineResult::Gamma)
        .def_readwrite("mu",            &ReHLineResult::mu)
        .def_readwrite("objective", &ReHLineResult::objective)
        .def_readwrite("dual_objective", &ReHLineResult::dual_objective)
        .def_readwrite("dual_gap", &ReHLineResult::dual_gap)
        .def_readwrite("constraint_violation", &ReHLineResult::constraint_violation)
        .def_readwrite("scaled_constraint_violation", &ReHLineResult::scaled_constraint_violation)
        .def_readwrite("kkt_residual", &ReHLineResult::kkt_residual)
        .def_readwrite("converged", &ReHLineResult::converged)
        .def_readwrite("niter",         &ReHLineResult::niter)
        .def_readwrite("dual_objfns",   &ReHLineResult::dual_objfns)
        .def_readwrite("primal_objfns", &ReHLineResult::primal_objfns);

    // https://hopstorawpointers.blogspot.com/2018/06/pybind11-and-python-sub-modules.html
    m.attr("__name__") = "rehline._internal";
    m.doc() = "rehline";
    m.def("rehline_internal", &rehline_internal<MapMat>, py::call_guard<py::gil_scoped_release>(),
          py::arg("result"), py::arg("X"), py::arg("A"), py::arg("b"), py::arg("rho"),
          py::arg("U"), py::arg("V"), py::arg("S"), py::arg("T"), py::arg("Tau"),
          py::arg("max_iter"), py::arg("tol"), py::arg("shrink") = 1,
          py::arg("verbose") = 0, py::arg("trace_freq") = 100,
          py::arg("coordinate_order") = 0, py::arg("coordinate_seed") = -1);
    m.def("rehline_cqr_internal", &rehline_cqr_internal<MapMat>, py::call_guard<py::gil_scoped_release>(),
          py::arg("result"), py::arg("X"), py::arg("A"), py::arg("b"), py::arg("rho"),
          py::arg("U"), py::arg("V"), py::arg("S"), py::arg("T"), py::arg("Tau"),
          py::arg("quantile_count"), py::arg("max_iter"), py::arg("tol"), py::arg("shrink") = 1,
          py::arg("verbose") = 0, py::arg("trace_freq") = 100,
          py::arg("coordinate_order") = 0, py::arg("coordinate_seed") = -1);
    m.def("rehline_sparse_internal", &rehline_internal<SparseMatrix>, py::call_guard<py::gil_scoped_release>(),
          py::arg("result"), py::arg("X"), py::arg("A"), py::arg("b"), py::arg("rho"),
          py::arg("U"), py::arg("V"), py::arg("S"), py::arg("T"), py::arg("Tau"),
          py::arg("max_iter"), py::arg("tol"), py::arg("shrink") = 1,
          py::arg("verbose") = 0, py::arg("trace_freq") = 100,
          py::arg("coordinate_order") = 0, py::arg("coordinate_seed") = -1);
    m.def("rehline_cqr_sparse_internal", &rehline_cqr_internal<SparseMatrix>, py::call_guard<py::gil_scoped_release>(),
          py::arg("result"), py::arg("X"), py::arg("A"), py::arg("b"), py::arg("rho"),
          py::arg("U"), py::arg("V"), py::arg("S"), py::arg("T"), py::arg("Tau"),
          py::arg("quantile_count"), py::arg("max_iter"), py::arg("tol"), py::arg("shrink") = 1,
          py::arg("verbose") = 0, py::arg("trace_freq") = 100,
          py::arg("coordinate_order") = 0, py::arg("coordinate_seed") = -1);
    m.def("rehline_sparse_constraints_internal", &rehline_internal<MapMat, SparseMatrix>, py::call_guard<py::gil_scoped_release>(),
          py::arg("result"), py::arg("X"), py::arg("A"), py::arg("b"), py::arg("rho"),
          py::arg("U"), py::arg("V"), py::arg("S"), py::arg("T"), py::arg("Tau"),
          py::arg("max_iter"), py::arg("tol"), py::arg("shrink") = 1,
          py::arg("verbose") = 0, py::arg("trace_freq") = 100,
          py::arg("coordinate_order") = 0, py::arg("coordinate_seed") = -1);
    m.def("rehline_sparse_both_internal", &rehline_internal<SparseMatrix, SparseMatrix>, py::call_guard<py::gil_scoped_release>(),
          py::arg("result"), py::arg("X"), py::arg("A"), py::arg("b"), py::arg("rho"),
          py::arg("U"), py::arg("V"), py::arg("S"), py::arg("T"), py::arg("Tau"),
          py::arg("max_iter"), py::arg("tol"), py::arg("shrink") = 1,
          py::arg("verbose") = 0, py::arg("trace_freq") = 100,
          py::arg("coordinate_order") = 0, py::arg("coordinate_seed") = -1);
    m.def("rehline_cqr_sparse_constraints_internal", &rehline_cqr_internal<MapMat, SparseMatrix>, py::call_guard<py::gil_scoped_release>(),
          py::arg("result"), py::arg("X"), py::arg("A"), py::arg("b"), py::arg("rho"),
          py::arg("U"), py::arg("V"), py::arg("S"), py::arg("T"), py::arg("Tau"),
          py::arg("quantile_count"), py::arg("max_iter"), py::arg("tol"), py::arg("shrink") = 1,
          py::arg("verbose") = 0, py::arg("trace_freq") = 100,
          py::arg("coordinate_order") = 0, py::arg("coordinate_seed") = -1);
    m.def("rehline_cqr_sparse_both_internal", &rehline_cqr_internal<SparseMatrix, SparseMatrix>, py::call_guard<py::gil_scoped_release>(),
          py::arg("result"), py::arg("X"), py::arg("A"), py::arg("b"), py::arg("rho"),
          py::arg("U"), py::arg("V"), py::arg("S"), py::arg("T"), py::arg("Tau"),
          py::arg("quantile_count"), py::arg("max_iter"), py::arg("tol"), py::arg("shrink") = 1,
          py::arg("verbose") = 0, py::arg("trace_freq") = 100,
          py::arg("coordinate_order") = 0, py::arg("coordinate_seed") = -1);
}
