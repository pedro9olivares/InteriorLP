#include <iostream>
#include "InteriorLPSolver.h"
#include <Eigen/Dense>
#include <chrono>


using namespace std;


int main() {
    // Example problem. m=4 and n=6 counting slacks (otherwise n=2)

    MatrixXd A(4, 6);
    A   << 1, 1, 1, 0, 0, 0,
           3, 1, 0, 1, 0, 0,
           1, 0, 0, 0, 1, 0,
           0, 1, 0, 0, 0, 1;

    VectorXd b(4);
    b << 9, 18, 7, 6;

    VectorXd c(6);
    c << -3, -2, 0, 0, 0, 0;

    InteriorLPSolver solver(A, b, c);
    solver.printAttributes();

    cout << "Getting the starting point." << endl;
    auto [x, lam, s] = solver.getStartingPoint();
    cout << "The starting point is: " << endl;
    cout << "x: " << endl;
    cout << x.transpose() << endl;
    cout << "lam: " << endl;
    cout << lam.transpose() << endl;
    cout << "s: " << endl;
    cout << s.transpose() << endl;

    auto [x_aff, lam_aff, s_aff] = solver.getAffineScaling(x, lam, s);
    cout << "\nThe affine solution: " << endl;
    cout << "x aff: " << endl;
    cout << x_aff.transpose() << endl;
    cout << "lam aff: " << endl;
    cout << lam_aff.transpose() << endl;
    cout << "s aff: " << endl;
    cout << s_aff.transpose() << endl;
    cout << endl;

    double alpha_pri_aff = solver.getAlphaPrimalAff(x, x_aff);
    double alpha_dual_aff = solver.getAlphaPrimalAff(s, s_aff);

    cout << "Testing alpha_pri_aff: " << alpha_pri_aff << endl;
    cout << "Came from: " << x.transpose() << endl;
    cout << "And: " << x_aff.transpose() << endl;

    cout << endl;

    cout << "Testing alpha_dual_aff: " << alpha_dual_aff << endl;
    cout << "Came from: " << s.transpose() << endl;
    cout << "And: " << s_aff.transpose() << endl;

    cout << endl;

    //cout << x.rows() << endl;
    //cout << x_aff.rows() << endl;
    double mu_aff = solver.getMuAff(x, alpha_pri_aff, x_aff, s, alpha_dual_aff, s_aff);
    cout << "Testing mu_aff: \n" << mu_aff << endl;

    double sigma = solver.getSigma(mu_aff, x, s);
    cout << "\nTesting sigma: " << sigma << endl;

    auto [delta_x, delta_lam, delta_s] = solver.getSearchDirection(x, x_aff, s, s_aff, sigma, lam);
    cout << "\nThe search direction solution: " << endl;
    cout << "delta x: " << endl;
    cout << delta_x.transpose() << endl;
    cout << "delta lam: " << endl;
    cout << delta_lam.transpose() << endl;
    cout << "delta s: " << endl;
    cout << delta_s.transpose() << endl;
    cout << endl;

    cout << "\nx + delta x: \n" << (x + delta_x).transpose() << endl;

    double alpha_primal = solver.getAlphaPrimal(x, delta_x);
    double alpha_dual= solver.getAlphaPrimalAff(s, delta_s);

    cout << "\nTesting alpha primal: " << alpha_primal << endl;
    cout << "Came from: " << x.transpose() << endl;
    cout << "And: " << delta_x.transpose() << endl;

    cout << endl;

    cout << "Testing alpha dual: " << alpha_dual << endl;
    cout << "Came from: " << s.transpose() << endl;
    cout << "And: " << delta_s.transpose() << endl;

    cout << endl;

    cout << "\nFirst x step: \n" << (x + alpha_primal*delta_x).transpose() << endl;

    // Ahora, si lo bueno
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    auto [x_sol, lam_sol, s_sol] = solver.solve();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<
            "[µs]" << std::endl;
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() <<
            "[ns]" << std::endl;

    std::cout << "Time difference (sec) = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).
        count()) / 1000000.0 << std::endl;

    return 0;
}