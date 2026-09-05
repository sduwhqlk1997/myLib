# SAW2_5D / SAW.hpp 文档

文件: include/SAW.hpp
命名空间: `SAW2_5D`
语言: C++

## 概述

本文件为 2.5D 表面声波（SAW）器件建模提供接口与数据结构，依赖库 `myFEM`。
主要功能包括：
- 材料库与材料参数结构（压电 / 弹性 / 各向异性等）；
- 网格/子问题生成辅助（IDT 网格步长等）；
- 子问题矩阵组装（压电 / 弹性 / 含 PML 的复数矩阵）；
- 常用边界条件处理（周期边界、接地、终端电压、悬浮电势、PML 外边界）；
- 器件级封装：`baseStructure`（子结构）与 `deviceArray`（器件序列、区域分解、全局组装与无量纲化）。

该头文件只是声明（接口）；具体实现依赖于同仓库中对应实现文件及 `myFEM` 的类型定义（例如 `Mat_d`、`SparseMat_t`、`mesh`、`refGaussInfo` 等）。

## 依赖与类型（来自 myFEM）

- Mat_d / Mat_t<double>：矩阵类型
- SparseMat_t<T>：稀疏矩阵模板
- Vec_t<Complex>：复数向量
- Vec_i：整型向量
- Idx：索引别名
- mesh / refGaussInfo：网格与高斯积分信息
- Fun_t<T>：函数模板（如 PML 复伸缩因子）

## 全局函数与结构

### findCommonFaceDof(const Mat_d &dofIdx1, const Mat_d &dofIdx2, int interFaceDim, double interFacePt1, double interFacePt2, double scale)

- 功能：提取两组自由度索引（dofIdx1/dofIdx2）在某一与坐标平面平行的交界面（即某一坐标相同的面）上的公共点/自由度。
- 参数：
  - `dofIdx1`, `dofIdx2`：自由度索引矩阵（格式与 `baseStructure::dofIdx` 一致）；
  - `interFaceDim`：指示哪一维为法向（0/1/2）；
  - `interFacePt1`, `interFacePt2`：界面位置或判断阈值；
  - `scale`：缩放/容差因子。
- 返回：`pair<vector<Idx>, vector<Idx>>`，表示两侧对应的公共自由度索引集合。

### translate3dPoints(Mat_d &Pt, Eigen::Vector3d oriOld, Eigen::Vector3d oriNew)

- 功能：将点矩阵 `Pt` 从原点 `oriOld` 平移到 `oriNew`（就地修改 `Pt`）。

### material / materialLib / VoigtIdx

- `struct material`：包含材料类型 `materialTag type`，弹性张量 `c (6x6)`，压电常数 `e (3x6)`，介电常数 `epcl (3x3)`，拉梅常数 `lambda, mu`，频率 `w`，密度 `rho` 等。
- `material materialLib(typeMaterial type)`：材料库函数，根据 `typeMaterial` 返回对应 `material` 参数。
- `Idx VoigtIdx(Idx i, Idx j)`：将矩阵索引 `(i,j)` 转为 Voigt 索引（应力/应变到矩阵索引的映射）。

## 网格与子问题生成

- `struct meshStep`：包含向量 `xGrid_p, yGrid, zGrid_p, xGrid_e, zGrid_e`，表示 p/e 网格的一维坐标分布。
- `meshStep genMeshStepOfIDT(Mat_d dom1, Mat_d dom2, int Nx_p, int Ny_p, int Nz_p, int Nx_e, int Nz_e)`：为 IDT 相关的两域生成网格步长/坐标。

## 矩阵组装接口

- `assemblePiezMat(material para, mesh myMesh, refGaussInfo Gauss, bool ifOMP = false)`
  - 功能：为压电区域组装局部矩阵（返回 pair<Mat_d, SparseMat_t<double>>）。
  - `ifOMP`：若为 true，内部实现可能启用 OpenMP 并行。

- `assembleElasticMat(material para, mesh myMesh, refGaussInfo Gauss, bool ifOMP = false)`
  - 功能：为弹性区域组装局部矩阵（非压电）。

- `assemblePiezPMLMat(Fun_t<Complex> alpha_x, Fun_t<Complex> alpha_y, Fun_t<Complex> alpha_z, material para, mesh myMesh, refGaussInfo Gauss, bool ifOMP = false)`
  - 功能：为压电区域在经过 PML 伸缩变换后组装复数系统（返回 pair<Mat_d, SparseMat_t<Complex>>）。

- `assembleElaPMLMat(...)`
  - 功能：非压电弹性块的 PML 组装（注：头文件备注为 TODO，可能未完成）。

## 边界条件处理函数

- `set2_5DAssumption(double yBehind, double yFront, Mat_d &dofIdx, SparseMat_t<Complex> &K, Vec_i &dof2Nodes)`
  - 功能：将区域在 y 方向的前后面处理为周期边界（2.5D 假设）。

- `treatFixedGroundCond(double zBottom, Mat_d &dofIdx, SparseMat_t<Complex> &K, Vec_i &dof2Nodes, double scale = 1)`
  - 功能：处理底面固定接地边界条件（k 矩阵相应处理），`scale` 为尺度因子。

- `treatTerminalBoundCond(double zIntFace, double xIntFaceL, double xIntFaceR, double V0, Mat_d &dofIdx, SparseMat_t<Complex> &K, Vec_t<Complex> &F, Vec_i &dof2Nodes, bool ifOMP = false)`
  - 功能：在电极/终端面施加电压边界并更新右端项 `F`。通常应在局部矩阵组装后、合并全局矩阵前调用。

- `treatFloatPotentialCond(double zIntFace, double xIntFaceL, double xIntFaceR, Mat_d &dofIdx, SparseMat_t<Complex> &K, Vec_i &dof2Nodes)`
  - 功能：处理悬浮电势（例：开路反射栅）。

- `treatPMLDirBoundCond(Mat_d dom, Mat_d &dofIdx, SparseMat_t<Complex> &K, Vec_i &dof2Nodes, pmlPosition pos, double scale)`
  - 功能：对 PML 外边界施加合适处理（吸收/阻尼/Dirichlet），`pos` 指示 PML 在哪个边。

## 类：baseStructure

表示器件的一个子结构（基底 / 电极 / PML / 弹性体 / IDT / 反射栅等），包含几何、材料、网格、自由度与局部线性系统。

主要成员：
- `typeBaseStructure type`：子结构类型；
- `Mat_d dom_p`、`material para_p`：基底的几何域与材料；
- `Mat_d dom_e`、`material para_e`：电极域与材料（若有）；
- `double V0`：电极电压（IDT 用）；
- `Fun_t<Complex> alpha_x, alpha_y, alpha_z`：PML 伸缩函数；
- `mesh meshFEM`：网格；
- `Mat_t<double> dofIdx`：自由度索引（列约定：前 3 列为坐标，第 4 列为自由度类型：0-2 位移，3 电势）；
- `Vec_i dof2Nodes`：自由度对应的节点索引；
- `SparseMat_t<Complex> K, Vec_t<Complex> F`：局部系统矩阵和右端项。

构造函数支持：仅基底、基底+电极、PML 特殊构造（带 alpha_*）。

常用成员函数：
- `saveNmeshPt(...)` / `readNmeshPt(...)`：保存/读取网格点数；
- `setTerminalVoltage(double V)`：仅当 `type==IDT` 时有效，否则程序退出；
- `initLinearSystem(const refGaussInfo &Gauss, const mesh &myMesh, bool ifOMP=false)`：组装局部矩阵（立方体类）；
- `initLinearSystem(const refGaussInfo &Gauss, const mesh &myMesh1, const mesh &myMesh2, bool ifOMP=false)`：IDT 类子结构矩阵初始化；
- 边界条件便捷调用：
  - `myTreatPeriodBoundCond()` -> 调用 `set2_5DAssumption`；
  - `myTreatFloatPotentialCond()` -> 仅在 `refGratOpen` 有意义，否则退出；
  - `myTreatTerminalBoundCond(bool ifOMP=false)` -> 仅 `IDT` 有效，会修改 `F`；
  - `myTreatFixedGroundCond()` -> 处理固定接地（使用 `dom_p` 计算 scale）；
  - `myTreatPMLDirBoundCond()` -> 处理 PML 外边界（计算 scale 并调用 `treatPMLDirBoundCond`）。

注意：部分函数在类型不匹配时会直接 `std::exit(EXIT_FAILURE)`，使用者需保证类型正确。

## 类：deviceArray

用于组合多个 `baseStructure`，执行区域分解（带状子结构提取）、生成子问题、无量纲化、组装全局线性系统并保存/恢复解。还包含方便构造常见器件（如 `basicResonator`）的模板函数。

主要成员：
- `std::vector<baseStructure> subStructures`：子结构表；
- `std::vector<SubProb> subProbs`：区域分解后得到的子问题；
- `int numSubStructures`；
- `Mat_i baseStructureArray`、`geoArray`：器件序列描述与几何阵列；
- `DimScales dimScales`：无量纲尺度集合（L0, U0, c0, e0, epcl0, rho0, omega0, Phi0, sigma0, D0, v0, k0）；
- 全局系统：`SparseMat_t<Complex> K, Vec_t<Complex> F, Vec_t<Complex> sol`；
- `Mat_t<double> dofIdx`：全局自由度索引矩阵。

主要方法：
- `setDeviceArray(Mat_i &deviceArray, Eigen::Vector3d ori = {0,0,0})`：设置器件序列与原点；
- 无量纲化/恢复：`genDimScales(double U0)`, `dimensionless()`, `recoverDimSolution()`；
- 区域分解：`genBandedSubProbs()`（内部调用 `extractBandedSubstructure()`），会清空 `subStructures` 并生成 `subProbs`/`geoArray`；
- 全局组装：`formGlobalSystem()` 将所有子问题合并为全局矩阵并清除局部数据；
- 模板构造：`basicResonator(...)`：生成一类常用的谐振器结构（参数详见头文件）。

## 使用建议与注意事项

- 边界条件的调用顺序很重要：修改右端项 `F` 的边界（如 `treatTerminalBoundCond`）应在局部矩阵组装完成并在合并到全局矩阵前调用。
- 并行化：若开启 OpenMP（`ifOMP=true`），确保 `myFEM` 的实现与所用环境线程安全。
- PML：需要为 PML 区域提供 `alpha_x/alpha_y/alpha_z` 函数（`Fun_t<Complex>`），这些函数接收坐标并返回相应复数伸缩系数。
- 错误处理：部分接口在错误输入时会直接退出程序，调用方应做好类型/前置条件检查以避免非预期中断。

## 典型调用流程（伪代码）

1. 单个 `baseStructure` 初始化与边界处理：

```cpp
// 构造并组装 IDT 子结构
baseStructure b(typeBaseStructure::IDT, dom_p, dom_e, typeMaterial::LN128YX, typeMaterial::Al, w);
b.initLinearSystem(Gauss, myMesh1, myMesh2, /*ifOMP*/ true);
b.setTerminalVoltage(V0);
b.myTreatTerminalBoundCond(/*ifOMP*/ true);
b.myTreatFixedGroundCond();
```

2. 使用 `deviceArray` 生成谐振器并组装全局系统：

```cpp
deviceArray device;
// 使用 basicResonator 辅助函数生成子结构序列并设置 device
device.basicResonator(... /* 详见头文件 */);
// 无量纲化（可选）
device.genDimScales(U0);
device.dimensionless();
// 区域分解（根据需要）
device.genBandedSubProbs();
// 组装全局矩阵
device.formGlobalSystem();
// 此后调用线性求解器： solve(device.K, device.F) -> device.sol
// 若做了无量纲化，恢复量纲
// device.recoverDimSolution();
```

