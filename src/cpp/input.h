#pragma once

#include <stdint.h>
#include <string>
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>

using DenseMatrix = Eigen::MatrixXd;
// using Triplet = Eigen::Triplet<double>;

inline uint32_t add_numbers(uint32_t a, uint32_t b) { return a+b; }

extern "C" double eigen_operations(const double* data, std::size_t nrows, std::size_t ncols) {
    Eigen::Map<const Eigen::MatrixXd> mat(data, nrows, ncols);
    return mat.determinant();
}
