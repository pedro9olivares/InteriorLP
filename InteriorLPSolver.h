//
// Created by Pedro Olivares on 18/03/25.
//
#ifndef INTERIORLPSOLVER_H
#define INTERIORLPSOLVER_H

#include <Eigen/Dense>
#include <tuple>

using namespace Eigen;
using namespace std;

class InteriorLPSolver {
private:
    MatrixXd A;
    VectorXd b;
    VectorXd c;
    long m;
    long n;

public:
    // Constructor
    InteriorLPSolver(MatrixXd& A_, VectorXd& b_, VectorXd& c_);

    void printAttributes();

    tuple<VectorXd, VectorXd, VectorXd> getStartingPoint();
    tuple<VectorXd, VectorXd, VectorXd> getAffineScaling(VectorXd& x, VectorXd& lam, VectorXd& s);
    double getAlphaPrimalAff(VectorXd& x, VectorXd& x_aff);
    double getAlphaDualAff(VectorXd& s, VectorXd& s_aff);
    double getMuAff(VectorXd& x, double alpha_pri_aff, VectorXd& x_aff,
        VectorXd& s, double alpha_dual_aff, VectorXd& s_aff);
    double getSigma(double mu_aff, VectorXd& x, VectorXd& s);
    tuple<VectorXd, VectorXd, VectorXd> getSearchDirection(VectorXd& x, VectorXd& x_aff,
        VectorXd& s, VectorXd& s_aff, double sigma, VectorXd& lam);
    double getAlphaPrimal(VectorXd& x, VectorXd& delta_x);
    double getAlphaDual(VectorXd& s, VectorXd& delta_s);

    tuple<VectorXd, VectorXd, VectorXd> solve();

};

#endif //INTERIORLPSOLVER_H
