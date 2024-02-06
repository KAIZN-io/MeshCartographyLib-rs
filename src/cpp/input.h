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
                                       const double* values, unsigned long n_triplets,
                                       double* output) {
    Eigen::Map<const DenseMatrix> bb_mtx(data, nrows, ncols);

    // Convert the input triplets to a sparse matrix
    Eigen::SparseMatrix<double> sparse_mtx(nrows, nrows);
    std::vector<Eigen::Triplet<double>> eigen_triplets;

    for (unsigned long i = 0; i < n_triplets; ++i) {
        eigen_triplets.emplace_back(rows[i], cols[i], values[i]);
    }
    sparse_mtx.setFromTriplets(eigen_triplets.begin(), eigen_triplets.end());

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(sparse_mtx);

    if (solver.info() != Eigen::Success) {
        std::cerr << "Decomposition failed" << std::endl;
        return;
    }

    Eigen::MatrixXd X = solver.solve(bb_mtx);

    if (solver.info() != Eigen::Success) {
        std::cerr << "Solving failed" << std::endl;
        return;
    }

    // Flatten the matrix X and write it into the output array
    unsigned long index = 0;
    for (unsigned long i = 0; i < nrows; ++i) {
        for (unsigned long j = 0; j < ncols; ++j) {
            output[index++] = X(i, j);
        }
    }
}

extern "C" double eigen_operations(const double* data, std::size_t nrows, std::size_t ncols) {
    Eigen::Map<const Eigen::MatrixXd> mat(data, nrows, ncols);
    return mat.determinant();
}
