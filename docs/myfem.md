# myFEM / myfem.hpp 文档

文件: include/myfem.hpp
命名空间: `myFEM`
语言: C++

## 概述

`myfem.hpp` 是一个轻量级的有限元辅助头文件，基于 Eigen 提供常用的网格、基函数、高斯积分、刚度矩阵/载荷向量组装以及网格拼接与合并工具。该文件仅包含接口声明（实现位于源码其它文件），并以模板与 Eigen 类型为主，便于在不同标量类型（double / complex）下复用。

主要功能：
- 网格数据结构（mesh）与参考单元高斯点信息（refGaussInfo）；
- 规则长方体（或参考单元）到三维有限元网格的生成；
- 基函数、仿射变换与参考单元高斯点生成；
- 刚度矩阵与载荷向量的组装（支持常系数与变系数函数形式）；
- 网格 / 矩阵 / 向量的拼接（合并）工具；
- 若干常用工具函数（数值比较、点取出、点哈希/相等比较）。

## 主要类型与别名

- Complex = std::complex<double>
- Mat_d = Eigen::MatrixXd
- Mat_i = Eigen::Matrix<Eigen::Index, Dynamic, Dynamic>
- Vec_d = Eigen::VectorXd
- Vec_cd = Eigen::VectorXcd
- Vec_i = Eigen::Vector<Eigen::Index, Dynamic>
- Idx = Eigen::Index
- 模板别名：
  - template<typename Scalar> using SparseMat_t = Eigen::SparseMatrix<Scalar, ColMajor>
  - template<typename Scalar> using Fun_t = std::function<Eigen::Vector<Scalar, Dynamic>(const Eigen::MatrixX3d &)>
  - template<typename Scalar> using Vec_t = Eigen::Vector<Scalar, Dynamic>
  - template<typename Scalar> using Mat_t = Eigen::Matrix<Scalar, Dynamic, Dynamic>

- 枚举：
  - meshType: hex（六面体），tet（四面体）
  - elemType: Q1（一次 Lagrange），Q2（二次 Lagrange）

## 重要结构体

### struct mesh

表示一个有限元网格（主要用于长方体划分与 Q1/Q2 元）。字段：
- Idx nElems：单元数量
- Idx nPts：节点数量
- meshType meshtype：网格类型（hex / tet）
- elemType elemtype：单元类型（Q1 / Q2）
- Mat_d Dom：全局区域（可能为 2x3 或 3x2 等，具体实现中定义）
- Mat_d nodes：节点坐标矩阵，每行一个点
- Mat_i elems：单元节点索引矩阵，每行一个单元；顺序为“单元顶点、边中点、面中点、体中点”，编号对应 nodes
- Vec_i bound_xl, bound_xr, bound_yl, bound_yr, bound_zl, bound_zr：边界节点索引（左右/前后/上下）
- Vec_i nXnYnZ = Vec_i(3)：网格在 x,y,z 方向的点数分布
- Vec_d Jacobi：每个单元的雅可比行列式（行向量）
- Mat_d Jacobi_inv：雅可比矩阵的逆（存储策略受实现决定）
- std::vector<Mat_d> GaussPt：每个单元上的 Gauss 积分点坐标集合（每行一个点）

### struct refGaussInfo

参考单元上的高斯点信息：
- meshType meshtype
- elemType elemtype
- Mat_d pt：Gauss 点矩阵，每行一个点（参考单元坐标）
- Vec_d weight：对应权重向量
- std::vector<Mat_d> phi：基函数信息（及其一阶导数）在参考 Gauss 点的值。注：phi[0..3] 分别为基函数与其关于 x,y,z 的导数。

## 函数接口（按功能分组）

### 网格生成与拓扑完善

- mesh genMesh3D(Mat_d dom, const Vec_d &xGrid, const Vec_d &yGrid, const Vec_d &zGrid, meshType type, bool ifOMP = false)
  - 描述：在矩形域 `dom` 上沿给定轴向网格点生成 3D 网格，返回 `mesh`。
  - 参数：`xGrid,yGrid,zGrid` 为一维坐标向量，`type` 指定 hex/tet，`ifOMP` 控制并行化。

- mesh genFEMMesh3D(Mat_d dom, const Vec_d &xGrid, const Vec_d &yGrid, const Vec_d &zGrid, elemType type, bool ifOMP = false)
  - 描述：生成用于 FEM 的网格（Q1 / Q2 元），并可能补齐边/面/体中点信息。

- void addEdgeMidPt(mesh &myMesh, const Vec_d &xGrid, const Vec_d &yGrid, const Vec_d &zGrid, bool ifOMP = false)
  - 描述：向网格添加边中点（用于二次元件 Q2）。

- void addFaceMidPt(mesh &myMesh, const Vec_d &xGrid, const Vec_d &yGrid, const Vec_d &zGrid, bool ifOMP = false)
  - 描述：向网格添加面中点。

- void addBodyMidPt(mesh &myMesh, const Vec_d &xGrid, const Vec_d &yGrid, const Vec_d &zGrid, bool ifOMP = false)
  - 描述：向网格添加体中点（用于高阶元的体中心点）。

### 基函数与仿射变换

- double baseFunRef3D(Eigen::Vector3d pt, Idx idxFun, Vec_i diff, elemType type)
  - 描述：在参考单元 [-1,1]^3 上计算有限元基函数值或其导数。
  - 参数：`pt` 参考坐标；`idxFun` 基函数编号；`diff` 指定求导阶次（例如 [1,0,0] 表示对 x 求一阶偏导）；`type` 元类型。

- double baseFun3D(Eigen::Vector3d pt, Idx idxFun, Vec_i diff, elemType type, Mat_d elem)
  - 描述：在任意矩形单元上计算基函数（通过仿射映射或插值实现）。

- void genAffineInfo(mesh &myMesh, bool ifOMP = false)
  - 描述：为每个单元生成仿射变换信息（Jacobi、逆矩阵等），以便将参考 Gauss 积分映射到物理单元。

### Gauss 积分信息

- refGaussInfo genRefGauss(Idx order, meshType meshtype, elemType elemtype, bool ifOMP = false)
  - 描述：根据给定阶数 `order`（积分阶数）与单元/网格类型产生参考单元上的 Gauss 点与权重，以及基函数在这些点上的值。

- void affineGauss2AllElems(const refGaussInfo &GaussInfo, mesh &myMesh, bool ifOMP = false)
  - 描述：将参考单元的 Gauss 点坐标映射到所有单元，并在 `myMesh.GaussPt` 中存储各单元的 Gauss 点。

### 刚度矩阵 / 载荷向量 组装

- SparseMat_t<double> assembleMat(const mesh &myMesh, const refGaussInfo &Gauss, Eigen::Vector3i DfTest, Eigen::Vector3i DfTrail, bool enable_omp = false)
  - 描述：为常系数问题（系数为标量常数）组装刚度矩阵。`DfTest`/`DfTrail` 指示试函数与试验函数的自由度分量配置。

- template<typename Scalar>
  SparseMat_t<Scalar> assembleMat(const Fun_t<Scalar> &coef, const mesh &myMesh, const refGaussInfo &Gauss, Eigen::Vector3i DfTest, Eigen::Vector3i DfTrail, bool enable_omp = false)
  - 描述：为变系数或标量函数系数问题组装刚度矩阵，`coef` 为一个在 Gauss 点上返回系数向量/矩阵的函数。

- Vec_t<double> assembleVec(const mesh &myMesh, const refGaussInfo &Gauss, bool ifOMP = false)
  - 描述：组装常数右端项的载荷向量。

- template<typename Scalar>
  Vec_t<Scalar> assembleVec(const Fun_t<Scalar> &RHS, const mesh &myMesh, const refGaussInfo &Gauss, bool ifOMP = false)
  - 描述：通过给定右端项函数 `RHS`（在 Gauss 点上返回值）组装载荷向量。

### 网格 / 系统 合并与拼接

这些函数用于将两个网格或两个有限元系统（dofIdx / K / F）按交界面拼接为单一系统，返回合并后的网格或矩阵/向量，以及映射关系。

- bool almostEqual(double x, double y, double eps = EPS)
  - 描述：数值近似比较（默认 eps = 1e-16）。

- Vec_d getPointNd(const Mat_d &X, Idx row)
  - 描述：从坐标矩阵 X 中取出第 row 行点作为向量返回。

- struct PointNdHash / PointNdEqual
  - 描述：为 Vec_d 点提供哈希与相等比较器，便于在 unordered_map / unordered_set 中作为键使用（用于点去重或交界面点查找）。

- findCommonDof(const Mat_d &dofIdx1, const Mat_d &dofIdx2, std::pair<int,double> interFace, double scale = 1.0, Idx searchStart1 = 0, Idx searchStart2 = 0, Idx searchEnd1 = -1, Idx searchEnd2 = -1)
  - 描述：查找两个自由度索引矩阵在给定交界面（通过 int/double pair 指示维度与位置）上的公共自由度，返回两侧对应索引的向量对。
  - 参数说明：`interFace.first` 为维度索引（0/1/2），`interFace.second` 为该维度上的坐标值或阈值；`scale` 为容差/缩放；`searchStart`/`searchEnd` 控制在索引区间内搜索。

- mergeFEMMesh_info mergeFEMMesh(const mesh &myMesh1, const mesh &myMesh2, std::pair<int,double> interFace, double scale)
  - 描述：将两个有限元网格沿交界面拼接，返回 `mergeFEMMesh_info`，其中包含拼接后的网格、交界面点数和第二个网格节点的旧序号->新序号映射。

- template<typename Scalar>
  SparseMat_t<Scalar> mergeSparseMat(const SparseMat_t<Scalar> &K1, const SparseMat_t<Scalar> &K2, const std::vector<Idx> &idx1, const std::vector<Idx> &idx2)
  - 描述：根据给定索引映射，将两个稀疏矩阵合并到一个更大的稀疏矩阵中。

- template<typename Scalar>
  Vec_t<Scalar> mergeVec(const Vec_t<Scalar> &F1, const Vec_t<Scalar> &F2, const std::vector<Idx> &idx1, const std::vector<Idx> &idx2)
  - 描述：合并两个右端向量，按映射拼接。

- template<typename Scalar>
  Mat_t<Scalar> mergeDofIdx(const Mat_t<Scalar> &dofIdx1, const Mat_t<Scalar> &dofIdx2, const std::vector<Idx> &idx2)
  - 描述：合并两个自由度索引矩阵，`idx2` 指示第二个 dofIdx 的行如何映射到合并后网格的行。

- template<typename Scalar>
  struct mergeFEMMat_info { SparseMat_t<Scalar> K; Mat_d dofIdx; std::vector<Idx> idx1; std::vector<Idx> idx2; };

- template<typename Scalar>
  mergeFEMMat_info<Scalar> mergeFEMMat(const Mat_d &dofIdx1, const Mat_d &dofIdx2, const SparseMat_t<Scalar> &K1, const SparseMat_t<Scalar> &K2, std::pair<int,double> interFace, double scale)
  - 描述：基于交界面信息将两个稀疏系统（K1/K2）与它们的 dofIdx 合并，返回合并后的矩阵、合并后的 dofIdx 与索引映射信息。

- template<typename Scalar>
  Vec_t<Scalar> mergeFEMVec(Mat_d dofIdx1, Mat_d dofIdx2, Vec_t<Scalar> F1, Vec_t<Scalar> F2, std::pair<int,double> interFace, double scale)
  - 描述：合并两个右端向量并考虑 dofIdx 的合并映射。

- template<typename Scalar1, typename Scalar2>
  std::pair<SparseMat_t<Scalar1>, Vec_t<Scalar2>> mergeFEMMatVec(Mat_d dofIdx1, Mat_d dofIdx2, SparseMat_t<Scalar1> K1, SparseMat_t<Scalar1> K2, Vec_t<Scalar2> F1, Vec_t<Scalar2> F2, std::pair<int,double> interFace, double scale)
  - 描述：便利接口同时合并 K 与 F，支持 K 与 F 不同标量类型（例如 K 为 Complex，F 为 Complex / double 混合场景）。

## 并发与性能说明

- 许多生成/组装函数带有 `ifOMP` 或 `enable_omp` 参数，表明实现中可能使用 OpenMP 进行并行加速。调用者在开启并行前应确保环境与依赖（例如 Eigen 的线程安全配置）合适。
- 合并 / 拼接函数会涉及稀疏矩阵重构与索引映射，输入规模较大时请关注内存峰值。建议在合并完成后释放不再使用的局部数据（例如局部 K/F）以降低内存占用。

## 使用示例（伪代码）

1) 生成一个 Q2 长方体网格并组装常系数刚度矩阵：

```cpp
using namespace myFEM;
Mat_d dom = ...; // 定义域
Vec_d xGrid = ...; yGrid = ...; zGrid = ...;
mesh myMesh = genFEMMesh3D(dom, xGrid, yGrid, zGrid, elemType::Q2, /*ifOMP*/ true);
refGaussInfo G = genRefGauss(4, meshType::hex, elemType::Q2);
affineGauss2AllElems(G, myMesh);
Eigen::Vector3i DfTest(0,1,2), DfTrail(0,1,2); // 示例
auto K = assembleMat(myMesh, G, DfTest, DfTrail, /*enable_omp*/ true);
```

2) 合并两个网格并得到合并后的稀疏系统（伪代码）：

```cpp
auto info = mergeFEMMesh(mesh1, mesh2, {2, zInterface}, scale);
// info.myMesh 为合并后的网格，info.nodes2_old2new 为第二个网格节点索引映射

auto merged = mergeFEMMat(dofIdx1, dofIdx2, K1, K2, {2, zInterface}, scale);
SparseMat_t<double> K_combined = merged.K;
Mat_d dofIdx_combined = merged.dofIdx;
```

## 注意事项与提示

- `mesh.elems` 的节点顺序/中点插入规则在实现中约定，不同元类型（Q1/Q2）插入中点的具体索引需参考实现以确保与基函数一致。
- `EPS` 定义为 1e-16，`almostEqual` 使用该阈值进行浮点近似比较。对于工程级容差，可在调用处传入更大的 eps。
- `Fun_t<Scalar>` 要求函数在给定的 Gauss 点矩阵（每行是一个三维点）上返回一个对应的系数向量或场值向量；确保返回维度与组装调用期望一致。
- 合并函数（merge*）通常要求两个网格在交界面处拓扑对齐（或至少在一定容差内点坐标匹配），`interFace` 与 `scale` 用于控制匹配容差与搜索维度。

## 我已完成的操作

我已根据 include/myfem.hpp 的接口声明生成这份中文 Markdown 文档。

## 提交计划

如果你确认，我可以将此文档保存到仓库（例如 `docs/myfem.md`）并提交；也可以随后把链接添加到 `readme.md`（或 `README.md`）。

请确认是否将文档写入仓库（默认路径：`docs/myfem.md`），我将继续提交。