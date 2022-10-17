#include "eigency.h"

using Matrix = Eigen::MatrixXd;
using MapMat = Eigen::Map<Matrix>;
using Vector = Eigen::VectorXd;
using MapVec = Eigen::Map<Vector>;

struct L3Result;

void l3solver_external(
    const MapMat& X, const MapMat& A, const MapVec& b,
    const MapMat& U, const MapMat& V,
    const MapMat& S, const MapMat& T, 
    double tau,
    int max_iter, 
    double tol,
    MapVec& sol_beta, 
    MapVec& sol_xi, 
    MapMat& sol_Lambda, 
    MapMat& sol_Gamma, 
    MapMat& sol_Omega,
    int niter, 
    double sol_dual_obj,
    bool verbose = false
);