#pragma once
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <complex>
#include <vector>
#include <set>
namespace myEigen // 基于Eigen开发的一些常用矩阵操作
{
    using Idx = Eigen::Index;
    using Complex = std::complex<double>;
    template <typename Scalar>
    using SparseMat_t = Eigen::SparseMatrix<Scalar, Eigen::ColMajor>;
    template <typename Scalar>
    using Mat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    template <typename Scalar>
    using Vec_t = Eigen::Vector<Scalar, Eigen::Dynamic>;
    // 矩阵块拼接操作
    template <typename Scalar>
    SparseMat_t<Scalar> blkdiag(const std::vector<SparseMat_t<Scalar>> &matBloks);
    template <typename Scalar>
    SparseMat_t<Scalar> blkMat(const std::vector<std::vector<SparseMat_t<Scalar>>> &matBloks);
    template <typename Scalar>
    SparseMat_t<Scalar>
    delSpMatRowOrCol(const SparseMat_t<Scalar> &K, const std::set<Eigen::Index> &idxDel, Eigen::Index flag);
    template <typename Scalar>
    Mat_t<Scalar>
    delColRowDenMat(const Mat_t<Scalar> &K, const std::set<Eigen::Index> &idxDel, Eigen::Index flag); // 删除稠密矩阵的行列
    template <typename Scalar>
    SparseMat_t<Scalar> spMatAddColOrRow(SparseMat_t<Scalar> &K, std::vector<Idx> Idx1, std::vector<Idx> Idx2, Idx flag); // flag 0,1,2依次对应行、列、行列相加
    // template <typename Scalar>
    // void addSpMatColOrRow(SparseMat_t<Scalar> &K, std::vector<Idx> Idx1, std::vector<Idx> Idx2, Idx flag); // flag 0,1,2依次对应行、列、行列相加
    template <typename Scalar>
    Vec_t<Scalar> delValDenVec(const Vec_t<Scalar> &V, const std::set<Eigen::Index> &idxDel); // 删除稠密向量的元素
};