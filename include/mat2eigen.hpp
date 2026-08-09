#pragma once
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <complex>
namespace wrMatFile
{
    using Complex = std::complex<double>;
    using SparseMat = Eigen::SparseMatrix<Complex, Eigen::ColMajor>;
    using DenseMat = Eigen::MatrixXd;
    using DenseMatcd = Eigen::MatrixXcd;
    using Vec = Eigen::VectorXcd;
    using Vec_i = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;
    // 文件读写都在主线程进行
    void readSparseMatFromMatlab(const char *filename, const char *varname, SparseMat &K);
    void readDenseMatFromMatlab(const char *filename, const char *varname, DenseMat &K);
    void readVecFromMatlab(const char *filename, const char *varname, Vec &K);
    void saveEigenDenseMat2Mat(const char *filename, DenseMat &K);
    void saveEigenDenseMat2Mat(const char *filename, DenseMatcd &K);
    void saveEigenDenseMat2Mat(const char *filename, Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic> &K);
    void saveEigenSparseMat2Mat(const char *filename, SparseMat &K);
    void saveEigenSparseMat2Mat(const char *filename, Eigen::SparseMatrix<double, Eigen::ColMajor> &K);
    void saveEigenVec2Mat(const char *filename, Vec &V);
    void saveEigenVec2Mat(const char *filename, Eigen::VectorXd &V);
    void saveEigenVec2Mat(const char *filename, Vec_i &V);
}
