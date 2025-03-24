//
// Created by Pedro Olivares on 18/03/25.
//
#include "InteriorLPSolver.h"
#include <iostream>
#include <limits>

// Constructor definition
InteriorLPSolver::InteriorLPSolver(MatrixXd& A_, VectorXd& b_, VectorXd& c_) {
    m = A_.rows();
    n = A_.cols();
    A = A_;
    b = b_;
    c = c_;
}

void InteriorLPSolver::printAttributes() {
    // Print matrix dimensions
    std::cout << "Matrix A (" << m << " x " << n << "):\n";
    std::cout << A << "\n\n";

    // Print vector b
    std::cout << "Vector b (" << m << " x 1):\n";
    std::cout << b.transpose() << "\n\n";  // b is a column vector, so we transpose for better display

    // Print vector c
    std::cout << "Vector c (" << n << " x 1):\n";
    std::cout << c.transpose() << "\n\n";  // c is a column vector, so we transpose for better display
}

// Gets a viable starting point. Refer to NW[06], 2ed. Page 410.
tuple<VectorXd, VectorXd, VectorXd> InteriorLPSolver::getStartingPoint() {
    VectorXd x(n);
    VectorXd lam(m);
    VectorXd s(n);

    const VectorXd e = VectorXd::Ones(n);

    // First, find lambda, which needs no correction.
    lam =  (A * A.transpose()).inverse() * A * c;

    // x and s will need corrections.
    x = A.transpose() * (A * A.transpose()).inverse() * b;
    s = c - A.transpose() * lam;

    // First correction step for positivity in x, s
    double minxi = x.minCoeff();
    double delta_x = std::max( -(3/2) * minxi, 0.0);

    double minsi = s.minCoeff();
    double delta_s = std::max( -(3/2) * minsi, 0.0);

    x += delta_x*e;
    s += delta_s*e;

    // Second correction step. Ensure that x and s are not so close to zero.
    delta_x = (1/2) * (x.dot(s) / e.dot(s));
    delta_s = (1/2) * (x.dot(s) / e.dot(x));

    x += delta_x*e;
    s += delta_s*e;

    return make_tuple(x, lam, s);
}

// Solves system 14.30 from page 407. Refer to NW[06], 2ed. Page 411.
tuple<VectorXd, VectorXd, VectorXd> InteriorLPSolver::getAffineScaling(VectorXd& x, VectorXd& lam, VectorXd& s) {
    // Vectors to return
    VectorXd x_aff(n);
    VectorXd lam_aff(m);
    VectorXd s_aff(n);
    // Vectors for the system
    VectorXd r_b(m);
    VectorXd r_c(m);
    const VectorXd e = VectorXd::Ones(n);
    r_b = A * x - b;
    r_c = A.transpose() * lam + s - c;
    MatrixXd X = x.asDiagonal();
    MatrixXd S = s.asDiagonal();
    MatrixXd I = MatrixXd::Identity(n, n);

    // Building and solving system (14.30)
    long rows = n + m + n;
    long cols = n + m + n;

    MatrixXd block_system(rows, cols);
    block_system.setZero();

    // Fill in the blocks
    block_system.block(0, n, n, m) = A.transpose();  // A^T is of size (nxm) starting at (0,n)
    block_system.block(0, n+m, n, n) = I; // I is of size (nxn) starting at (0, n+m)

    block_system.block(n, 0, m, n) = A;  // A is of size (mxn) starting at (n,0)

    block_system.block(n + m, 0, n, n) = S;  // S is of size (nxn) starting at (n+m,0)
    block_system.block(n + m, n + m, n, n) = X;  // X is of size (nxn) starting at (n+m, n+m)

    // Create the RHS of the system
    VectorXd RHS(n + m + n);
    RHS << -r_c, -r_b, - X * S * e;

    // Solve the system
    VectorXd sol(n + m + n);
    sol << x_aff, lam_aff, s_aff;

    sol = block_system.colPivHouseholderQr().solve(RHS);

    // Extract the affs and return
    x_aff = sol.segment(0, n);
    lam_aff = sol.segment(n, m);
    s_aff = sol.segment(n+m, n);

    return make_tuple(x_aff, lam_aff, s_aff);
}

// Gets correction step
tuple<VectorXd, VectorXd, VectorXd> InteriorLPSolver::getCorrection(VectorXd& x, VectorXd& lam, VectorXd& s) {
    // Vectors to return
    VectorXd x_corr(n);
    VectorXd lam_corr(m);
    VectorXd s_corr(n);
    // Vectors for the system
    auto [x_aff, lam_aff, s_aff] = getAffineScaling(x, lam, s);
    const VectorXd e = VectorXd::Ones(n);
    MatrixXd X = x.asDiagonal();
    MatrixXd S = s.asDiagonal();
    MatrixXd I = MatrixXd::Identity(n, n);
    MatrixXd X_aff = x_aff.asDiagonal();
    MatrixXd S_aff = s_aff.asDiagonal();

    // Building and solving system (14.31)
    long rows = n + m + n;
    long cols = n + m + n;

    MatrixXd block_system(rows, cols);
    block_system.setZero();

    // Fill in the blocks
    block_system.block(0, n, n, m) = A.transpose();  // A^T is of size (nxm) starting at (0,n)
    block_system.block(0, n+m, n, n) = I; // I is of size (nxn) starting at (0, n+m)

    block_system.block(n, 0, m, n) = A;  // A is of size (mxn) starting at (n,0)

    block_system.block(n + m, 0, n, n) = S;  // S is of size (nxn) starting at (n+m,0)
    block_system.block(n + m, n + m, n, n) = X;  // X is of size (nxn) starting at (n+m, n+m)

    // Create the RHS of the system
    VectorXd RHS(n + m + n);
    RHS.setZero();
    //HS << 0, 0, - X_aff * S_aff * e;
    RHS.segment(n + m, n) = -X_aff * S_aff * e;

    // Solve the system
    VectorXd sol(n + m + n);
    sol << x_corr, lam_corr, s_corr;

    sol = block_system.colPivHouseholderQr().solve(RHS);

    // Extract the affs and return
    x_corr = sol.segment(0, n);
    lam_corr = sol.segment(n, m);
    s_corr = sol.segment(n+m, n);

    return make_tuple(x_corr, lam_corr, s_corr);
}

double InteriorLPSolver::getAlphaPrimalAff(VectorXd &x, VectorXd &x_aff) {
    double alpha_pri_aff;
    vector<double> neg_x_aff;

    // Store negative entries from x_aff
    for (double val : x_aff) {
        if (val < 0) {
            neg_x_aff.push_back(val);
        }
    }

    // Iterate over both vectors
    double min_val = std::numeric_limits<double>::infinity();
    double frac;
    for (double xi : x) {
        for (double delta_xi : neg_x_aff) {
            // Calculate the frac
            frac = - xi / delta_xi;
            if (frac < min_val) {
                min_val = frac;
            }
        }
    }
    alpha_pri_aff = min(1.0, frac);

    return alpha_pri_aff;
}

double InteriorLPSolver::getAlphaDualAff(VectorXd &s, VectorXd &s_aff) {
    double alpha_dual_aff;
    vector<double> neg_s_aff;

    // Store negative entries from x_aff
    for (double val : s_aff) {
        if (val < 0) {
            neg_s_aff.push_back(val);
        }
    }

    // Iterate over both vectors
    double min_val = std::numeric_limits<double>::infinity();
    double frac;
    for (double si : s) {
        for (double delta_si : neg_s_aff) {
            // Calculate the frac
            frac = - si / delta_si;
            if (frac < min_val) {
                min_val = frac;
            }
        }
    }
    alpha_dual_aff = min(1.0, frac);

    return alpha_dual_aff;
}

double InteriorLPSolver::getMuAff(VectorXd& x, double alpha_pri_aff, VectorXd& x_aff,
        VectorXd& s, double alpha_dual_aff, VectorXd& s_aff) {
    double mu_aff;

    mu_aff = (x + alpha_pri_aff * x_aff).dot(s + alpha_dual_aff * s_aff);
    //mu_aff /= n;

    return mu_aff;
}

double InteriorLPSolver::getSigma(double mu_aff, VectorXd& x, VectorXd& s) {
    double sigma, mu;

    mu = x.dot(s);
    mu /= n;

    sigma = pow((mu_aff / mu), 3);

    return sigma;
}

tuple<VectorXd, VectorXd, VectorXd> InteriorLPSolver::getSearchDirection(VectorXd& x, VectorXd& x_aff,
        VectorXd& s, VectorXd& s_aff, double sigma, VectorXd& lam) {
    // Vectors to return
    VectorXd delta_x(n);
    VectorXd delta_lam(m);
    VectorXd delta_s(n);
    // Vectors for the system
    VectorXd r_b(m);
    VectorXd r_c(m);
    const VectorXd e = VectorXd::Ones(n);
    r_b = A * x - b;
    r_c = A.transpose() * lam + s - c;
    MatrixXd X = x.asDiagonal();
    MatrixXd S = s.asDiagonal();
    MatrixXd I = MatrixXd::Identity(n, n);
    MatrixXd X_aff = x_aff.asDiagonal();
    MatrixXd S_aff = s_aff.asDiagonal();
    double mu = x.dot(s);
    mu /= n;

    // Building and solving system (14.35)
    long rows = n + m + n;
    long cols = n + m + n;

    MatrixXd block_system(rows, cols);
    block_system.setZero();

    // Fill in the blocks
    block_system.block(0, n, n, m) = A.transpose();  // A^T is of size (nxm) starting at (0,n)
    block_system.block(0, n + m, n, n) = I; // I is of size (nxn) starting at (0, n+m)

    block_system.block(n, 0, m, n) = A;  // A is of size (mxn) starting at (n,0)

    block_system.block(n + m, 0, n, n) = S;  // S is of size (nxn) starting at (n+m,0)
    block_system.block(n + m, n + m, n, n) = X;  // X is of size (nxn) starting at (n+m, n+m)

    // Create the RHS of the system
    VectorXd RHS(n + m + n);
    RHS << -r_c, -r_b, -(X * S * e) -(X_aff * S_aff * e) + sigma * mu * e;

    // Solve the system
    VectorXd sol(n + m + n);
    sol << delta_x, delta_lam, delta_s;

    sol = block_system.colPivHouseholderQr().solve(RHS);

    // Extract the affs and return
    delta_x = sol.segment(0, n);
    delta_lam = sol.segment(n, m);
    delta_s = sol.segment(n+m, n);

    return make_tuple(delta_x, delta_lam, delta_s);
}

double InteriorLPSolver::getAlphaPrimal(VectorXd& x, VectorXd& delta_x) {
    double alpha_primal;
    double eta = .9;
    double alpha_primal_max;
    vector<double> neg_delta_x;

    // Store negative entries from delta_x
    for (double val : delta_x) {
        if (val < 0) {
            neg_delta_x.push_back(val);
        }
    }

    // Iterate over both vectors
    double min_val = std::numeric_limits<double>::infinity();
    for (double xi : x) {
        for (double delta_xi : neg_delta_x) {
            // Calculate the frac
            alpha_primal_max = - xi / delta_xi;
            if (alpha_primal_max < min_val) {
                min_val = alpha_primal_max;
            }
        }
    }

    // The following is (14.38) from page 409
    alpha_primal = min(1.0, eta * min_val);

    return alpha_primal;
}

double InteriorLPSolver::getAlphaDual(VectorXd& s, VectorXd& delta_s) {
    double alpha_dual;
    double eta = .9;
    double alpha_dual_max;
    vector<double> neg_delta_s;

    // Store negative entries from delta_s
    for (double val : delta_s) {
        if (val < 0) {
            neg_delta_s.push_back(val);
        }
    }

    // Iterate over both vectors
    double min_val = std::numeric_limits<double>::infinity();
    for (double si : s) {
        for (double delta_si : neg_delta_s) {
            // Calculate the frac
            alpha_dual_max = - si / delta_si;
            if (alpha_dual_max < min_val) {
                min_val = alpha_dual_max;
            }
        }
    }

    // The following is (14.38) from page 409
    alpha_dual = min(1.0, eta * min_val);

    return alpha_dual;
}


// Mehrotra's algorithm for solve
tuple<VectorXd, VectorXd, VectorXd> InteriorLPSolver::solve() {
    cout << "\n============================" << endl;
    cout << "Solving...\n" << endl;


    auto [x_k, lam_k, s_k] = getStartingPoint();
    VectorXd prev_x_k;
    int counter = 0;

    do {
        auto [x_aff, lam_aff, s_aff] = getAffineScaling(x_k, lam_k, s_k);

        // The following four lines add correction
        auto [x_corr, lam_corr, s_corr] = getCorrection(x_k, lam_k, s_k);
        x_aff += x_corr;
        lam_aff += lam_corr;
        s_aff += s_corr;

        double alpha_primal_aff = getAlphaPrimalAff(x_k, x_aff);
        double alpha_dual_aff = getAlphaPrimalAff(s_k, s_aff);
        double mu_aff = getMuAff(x_k, alpha_primal_aff, x_aff, s_k, alpha_dual_aff, s_aff);
        double sigma = getSigma(mu_aff, x_k, s_k);

        auto [delta_x, delta_lam, delta_s] = getSearchDirection(x_k, x_aff, s_k, s_aff, sigma, lam_k);

        double alpha_primal = getAlphaPrimal(x_k, delta_x);
        double alpha_dual= getAlphaPrimalAff(s_k, delta_s);

        // Take the step:
        prev_x_k = x_k;

        x_k = x_k + alpha_primal * delta_x;

        lam_k = lam_k + alpha_dual * delta_lam;

        s_k = s_k + alpha_dual * delta_s;

        counter++;

        cout << "x_" << counter << ": " << x_k.head(2).transpose() << endl;
    } while ((prev_x_k - x_k).norm() > 1e-6);


    cout << "\n============================" << endl;
    cout << "Finished in " << counter << " iterations.\n" << endl;
    cout << "x solution : " << "(" << x_k.head(2).transpose() << ")" << endl;

    return make_tuple(x_k, lam_k, s_k);
}
