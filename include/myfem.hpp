#pragma once
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <vector>
#include <complex>
#include <functional>
#include <iostream>
namespace myFEM
{
#define EPS 1e-16
    /*变量声明*/
    using Complex = std::complex<double>;
    using Mat_d = Eigen::MatrixXd;
    using Mat_i = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic>;
    using Vec_d = Eigen::VectorXd;
    using Vec_cd = Eigen::VectorXcd;
    using Vec_i = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;
    using Idx = Eigen::Index;
    // using SparseMat_d = Eigen::SparseMatrix<double, Eigen::ColMajor>;
    // using SparseMat_cd = Eigen::SparseMatrix<Complex, Eigen::ColMajor>;
    // using realFun = std::function<Vec_d(const Eigen::MatrixX3d &)>;
    // using complexFun = std::function<Vec_cd(const Eigen::MatrixX3d &)>;
    template <typename Scalar>
    using SparseMat_t = Eigen::SparseMatrix<Scalar, Eigen::ColMajor>;
    template <typename Scalar>
    using Fun_t = std::function<
        Eigen::Vector<Scalar, Eigen::Dynamic>(const Eigen::MatrixX3d &)>;
    template <typename Scalar>
    using Vec_t = Eigen::Vector<Scalar, Eigen::Dynamic>;
    template <typename Scalar>
    using Mat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    enum meshType
    {
        hex, // 六面体
        tet  // 四面体
    };
    enum elemType
    {
        Q1, // 1次矩形lagrange元
        Q2  // 2次矩形lagrange元
    };
    struct mesh
    {
        Idx nElems;
        Idx nPts;
        meshType meshtype;
        elemType elemtype;
        Mat_d Dom;   // 区域
        Mat_d nodes; // 所有节点的坐标，每行一个点
        Mat_i elems; // 所有单元的节点编号，每行一个单元，顺序为：单元顶点、边中点、面中点、体中点，编号与nodes对应
        Vec_i bound_xl, bound_xr, bound_yl, bound_yr, bound_zl, bound_zr;
        Vec_i nXnYnZ = Vec_i(3);
        Vec_d Jacobi;               // 每个单元的Jacobi行列式
        Mat_d Jacobi_inv;           // Jacobi矩阵的逆
        std::vector<Mat_d> GaussPt; // 各个单元上的Gauss积分点，每行一个点
    };
    struct refGaussInfo // 参考单元Gauss点信息
    {
        meshType meshtype;
        elemType elemtype;
        Mat_d pt;               // Gauss点,每行一个点
        Vec_d weight;           // 权重
        std::vector<Mat_d> phi; // 参考单元基函数以及各一阶偏导在Gauss点处的值，
                                // 每行表示一个基函数，每列对应一个Gauss积分节点
                                // 0-3号矩阵分别为基函数，基函数关于x,y,z的导数
    };
    /*函数声明*/
    // 网格剖分
    mesh genMesh3D(Mat_d dom, const Vec_d &xGrid,
                   const Vec_d &yGrid, const Vec_d &zGrid,
                   meshType type); // 长方体区域网格剖分
    mesh genFEMMesh3D(Mat_d dom, const Vec_d &xGrid,
                      const Vec_d &yGrid, const Vec_d &zGrid,
                      elemType type);
    void addEdgeMidPt(mesh &myMesh, const Vec_d &xGrid,
                      const Vec_d &yGrid, const Vec_d &zGrid); // 向网格添加边中点
    void addFaceMidPt(mesh &myMesh, const Vec_d &xGrid,
                      const Vec_d &yGrid, const Vec_d &zGrid); // 向网格添加面中点
    void addBodyMidPt(mesh &myMesh, const Vec_d &xGrid,
                      const Vec_d &yGrid, const Vec_d &zGrid); // 向网格添加体中点
    // 基函数信息以及仿射变换信息生成
    double baseFunRef3D(Eigen::Vector3d pt, Idx idxFun, Vec_i diff, elemType type);          // 参考单元[-1,1]^3上的有限元基函数
    double baseFun3D(Eigen::Vector3d pt, Idx idxFun, Vec_i diff, elemType type, Mat_d elem); // 任意矩形单元的有限元基函数
    void genAffineInfo(mesh &myMesh);                                                        // 生成各单元与
    // 生成Gauss点信息
    refGaussInfo genRefGauss(Idx order, meshType meshtype, elemType elemtype); // 生成参考单元上的Gauss点信息
    void affineGauss2AllElems(const refGaussInfo &GaussInfo, mesh &myMesh);
    // 组装刚度矩阵
    SparseMat_t<double> assembleMat(const mesh &myMesh, const refGaussInfo &Gauss, Eigen::Vector3i DfTest, Eigen::Vector3i DfTrail, bool enable_omp = true); // 常系数用这个
    template <typename Scalar>
    SparseMat_t<Scalar> assembleMat(const Fun_t<Scalar> &coef, const mesh &myMesh, const refGaussInfo &Gauss, Eigen::Vector3i DfTest, Eigen::Vector3i DfTrail, bool enable_omp = true); // 变系数用这个
    // 组装载荷向量
    Vec_t<double> assembleVec(const mesh &myMesh, const refGaussInfo &Gauss); // 常数右端项用这个
    template <typename Scalar>
    Vec_t<Scalar> assembleVec(const Fun_t<Scalar> &RHS, const mesh &myMesh, const refGaussInfo &Gauss);
    // 区域拼接
    bool almostEqual(double x, double y, double eps = EPS);
    Vec_d getPointNd(const Mat_d &X, Idx row);
    struct PointNdHash
    {
        std::size_t operator()(const Vec_d &p) const;
    };
    struct PointNdEqual
    {
        bool operator()(const Vec_d &a, const Vec_d &b) const;
    };
    std::pair<std::vector<Idx>, std::vector<Idx>>
    findCommonDof(const Mat_d &dofIdx1, const Mat_d &dofIdx2,
                  std::pair<int, double> interFace,
                  double scale = 1.0, Idx searchStart1 = 0, Idx searchStart2 = 0,
                  Idx searchEnd1 = -1, Idx searchEnd2 = -1);
    struct mergeFEMMesh_info
    {
        mesh myMesh;                     // 拼接后的网格
        size_t nInterPts;                // 交界面点数
        std::vector<Idx> nodes2_old2new; // 二号网格的网格点的旧序号与新序号的对应关系，行号为旧序号，值为新序号
    };
    mergeFEMMesh_info mergeFEMMesh(const mesh &myMesh1, const mesh &myMesh2,
                                   std::pair<int, double> interFace, double scale); // 拼接两个有限元网格
    template <typename Scalar>
    SparseMat_t<Scalar> mergeSparseMat(const SparseMat_t<Scalar> &K1, const SparseMat_t<Scalar> &K2,
                                       const std::vector<Idx> &idx1, const std::vector<Idx> &idx2);
    template <typename Scalar>
    Vec_t<Scalar> mergeVec(const Vec_t<Scalar> &F1, const Vec_t<Scalar> &F2,
                           const std::vector<Idx> &idx1, const std::vector<Idx> &idx2);
    template <typename Scalar>
    Mat_t<Scalar> mergeDofIdx(const Mat_t<Scalar> &dofIdx1,
                              const Mat_t<Scalar> &dofIdx2,
                              const std::vector<Idx> &idx2);
    template <typename Scalar>
    struct mergeFEMMat_info
    {
        SparseMat_t<Scalar> K;
        Mat_d dofIdx;
        std::vector<Idx> idx1;
        std::vector<Idx> idx2;
    };
    template <typename Scalar>
    mergeFEMMat_info<Scalar>
    mergeFEMMat(const Mat_d &dofIdx1, const Mat_d &dofIdx2,
                const SparseMat_t<Scalar> &K1, const SparseMat_t<Scalar> &K2,
                std::pair<int, double> interFace, double scale);
    template <typename Scalar>
    Vec_t<Scalar> mergeFEMVec(Mat_d dofIdx1, Mat_d dofIdx2,
                              Vec_t<Scalar> F1, Vec_t<Scalar> F2,
                              std::pair<int, double> interFace, double scale);
    template <typename Scalar1, typename Scalar2>
    std::pair<SparseMat_t<Scalar1>, Vec_t<Scalar2>>
    mergeFEMMatVec(Mat_d dofIdx1, Mat_d dofIdx2,
                   SparseMat_t<Scalar1> K1, SparseMat_t<Scalar1> K2,
                   Vec_t<Scalar2> F1, Vec_t<Scalar2> F2,
                   std::pair<int, double> interFace, double scale);
};