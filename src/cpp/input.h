#pragma once

#include <stdint.h>
#include <string>
#include <vector>
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>

using DenseMatrix = Eigen::MatrixXd;

struct CTriplet {
    unsigned long row;
    unsigned long col;
    double value;
};

extern "C" void cholesky_decomposition(const double* data, unsigned long nrows, unsigned long ncols,
                                       const unsigned long* rows, const unsigned long* cols,
                                       const double* values, unsigned long n_triplets) {
    // Function implementation
    Eigen::Map<const DenseMatrix> B(data, nrows, ncols);

    Eigen::SparseMatrix<double> A(nrows, nrows);
    std::vector<Eigen::Triplet<double>> eigen_triplets;

    for (unsigned long i = 0; i < n_triplets; ++i) {
        eigen_triplets.emplace_back(rows[i], cols[i], values[i]);
    }
    A.setFromTriplets(eigen_triplets.begin(), eigen_triplets.end());

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);

    if (solver.info() != Eigen::Success) {
        std::cerr << "Decomposition failed" << std::endl;
        return;
    }

    Eigen::MatrixXd X = solver.solve(B);

    if (solver.info() != Eigen::Success) {
        std::cerr << "Solving failed" << std::endl;
        return;
    }

    std::cout << "Solution: \n" << X.size() << std::endl;
}




extern "C" double eigen_operations(const double* data, std::size_t nrows, std::size_t ncols) {
    Eigen::Map<const Eigen::MatrixXd> mat(data, nrows, ncols);
    return mat.determinant();
}

inline uint32_t add_numbers(uint32_t a, uint32_t b) { return a+b; }
