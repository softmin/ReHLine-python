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
using Vector = Eigen::VectorXd;
using MapVec = Eigen::Ref<Vector>;

using ReHLineResult = rehline::ReHLineResult<Matrix>;

void rehline_internal(
    ReHLineResult& result,
    const MapMat& X, const MapMat& A, const MapVec& b,
    const MapMat& U, const MapMat& V,
    const MapMat& S, const MapMat& T, const MapMat& Tau,
    int max_iter, double tol, int shrink = 1,
    int verbose = 0, int trace_freq = 100
)
{
    rehline::rehline_solver(result, X, A, b, U, V, S, T, Tau,
                            max_iter, tol, shrink, verbose, trace_freq);
}

PYBIND11_MODULE(_internal, m) {
    py::class_<ReHLineResult>(m, "rehline_result")
        .def(py::init<>())
        .def_readwrite("beta",          &ReHLineResult::beta)
        .def_readwrite("xi",            &ReHLineResult::xi)
        .def_readwrite("Lambda",        &ReHLineResult::Lambda)
        .def_readwrite("Gamma",         &ReHLineResult::Gamma)
        .def_readwrite("niter",         &ReHLineResult::niter)
        .def_readwrite("dual_objfns",   &ReHLineResult::dual_objfns)
        .def_readwrite("primal_objfns", &ReHLineResult::primal_objfns);

    // https://hopstorawpointers.blogspot.com/2018/06/pybind11-and-python-sub-modules.html
    m.attr("__name__") = "rehline._internal";
    m.doc() = "rehline";
    m.def("rehline_internal", &rehline_internal);
}
