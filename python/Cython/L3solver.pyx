# distutils: language = c++
# distutils: sources = ../src/l3solver.cpp

from eigency.core cimport *
from libcpp cimport bool
from libcpp.vector cimport vector
# cimport eigency.conversions
# from eigency_tests.eigency cimport *


# import eigency
# include "../eigency.pyx"

cdef extern from "../src/CythonExports.h":

    cdef void _l3solver_external "l3solver_external"(Map[MatrixXd] &X, 
                                                    Map[MatrixXd] &A, 
                                                    Map[VectorXd] &b, 
                                                    Map[MatrixXd] &U, 
                                                    Map[MatrixXd] &V,
                                                    Map[MatrixXd] &S, 
                                                    Map[MatrixXd] &T, 
                                                    double tau,
                                                    int max_iter, 
                                                    double tol,
                                                    Map[VectorXd] &sol_beta,
                                                    Map[VectorXd] &sol_xi,
                                                    Map[MatrixXd] &sol_Lambda,
                                                    Map[MatrixXd] &sol_Gamma,
                                                    Map[MatrixXd] &sol_Omega,
                                                    int niter, 
                                                    double sol_dual_obj,
                                                    bool verbose)

    ctypedef struct L3Result:
            VectorXd beta;
            VectorXd xi;
            MatrixXd Lambda;
            MatrixXd Gamma;
            MatrixXd Omega;
            int niter;
            vector[double] dual_objfns;


def l3solver_py(np.ndarray[np.float64_t, ndim=2, mode='c'] X,
                np.ndarray[np.float64_t, ndim=2, mode='c'] A,
                np.ndarray[np.float64_t, ndim=1, mode='c'] b,
                np.ndarray[np.float64_t, ndim=2, mode='c'] U,
                np.ndarray[np.float64_t, ndim=2, mode='c'] V,
                np.ndarray[np.float64_t, ndim=2, mode='c'] S,
                np.ndarray[np.float64_t, ndim=2, mode='c'] T,
                double tau,
                int max_iter,
                double tol,
                np.ndarray[np.float64_t, ndim=1, mode='fortran'] sol_beta,
                np.ndarray[np.float64_t, ndim=1, mode='fortran'] sol_xi,
                np.ndarray[np.float64_t, ndim=2, mode='fortran'] sol_Lambda,
                np.ndarray[np.float64_t, ndim=2, mode='fortran'] sol_Gamma,
                np.ndarray[np.float64_t, ndim=2, mode='fortran'] sol_Omega,
                int niter,
                double sol_dual_obj,
                bool verbose):
    
    _l3solver_external(Map[MatrixXd](X.copy(order='F')),
                       Map[MatrixXd](A.copy(order='F')),
                       Map[VectorXd](b),
                       Map[MatrixXd](U.copy(order='F')),
                       Map[MatrixXd](V.copy(order='F')),
                       Map[MatrixXd](S.copy(order='F')),
                       Map[MatrixXd](T.copy(order='F')),
                       tau,
                       max_iter,
                       tol,
                       Map[VectorXd](sol_beta),
                       Map[VectorXd](sol_xi),
                       Map[MatrixXd](sol_Lambda),
                       Map[MatrixXd](sol_Gamma),
                       Map[MatrixXd](sol_Omega),
                       niter,
                       sol_dual_obj,
                       verbose)
