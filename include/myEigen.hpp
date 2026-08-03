#pragma once
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <complex>
#include <vector>
namespace myEigen // 基于Eigen开发的一些常用矩阵操作
{
    using Idx = Eigen::Index;
    using Complex = std::complex<double>;
    template <typename Scalar>
    using SparseMat_t = Eigen::SparseMatrix<Scalar, Eigen::ColMajor>;
    // 矩阵块拼接操作
    template <typename Scalar>
    SparseMat_t<Scalar> blkdiag(const std::vector<SparseMat_t<Scalar>> &matBloks);
    template <typename Scalar>
    SparseMat_t<Scalar> blkMat(const std::vector<std::vector<SparseMat_t<Scalar>>> &matBloks);
};