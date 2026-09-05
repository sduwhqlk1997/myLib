#pragma once
#include <myfem.hpp>
#include <map>
#include <set>
namespace SAW2_5D
{
    using namespace myFEM;
    std::pair<std::vector<Idx>, std::vector<Idx>>
    findCommonFaceDof(const Mat_d &dofIdx1, const Mat_d &dofIdx2,
                      int interFaceDim, double interFacePt1, double interFacePt2,
                      double scale); // 提取两组点在某个与坐标平面平行的面上（有三个坐标为度相同）的公共点
    void translate3dPoints(Mat_d &Pt,
                           Eigen::Vector3d oriOld,
                           Eigen::Vector3d oriNew); // 将离散点平移到新的坐标系
    /*材料定义*/
    enum materialTag
    {
        Piez,
        LinearEla,     // 线弹性
        anisotropicEla // 各向异性弹性
    };
    enum typeMaterial
    {
        LN128YX,
        Al,
        LT42YX,
        Cu,
        testPiez,
        testEla
    };
    enum pmlPosition // PML位置
    {
        left,
        right,
        bottom,
        leftBottom,
        rightBottom
    };
    struct material
    {
        materialTag type;
        Mat_d c = Mat_d(6, 6);
        Mat_d e = Mat_d(3, 6);
        Mat_d epcl = Mat_d(3, 3); // 弹性张量、压电张量、介电张量
        double lambda = 0.0;
        double mu = 0.0; // 拉梅常数
        double w = 0.0;  // 频率
        double rho = 0.0;
    };
    material materialLib(typeMaterial type); // 材料库
    Idx VoigtIdx(Idx i, Idx j);              // Voigt表示法
    /*子问题矩阵组装*/
    struct meshStep
    {
        Vec_d xGrid_p, yGrid, zGrid_p, xGrid_e, zGrid_e;
    };
    meshStep genMeshStepOfIDT(Mat_d dom1, Mat_d dom2,
                              int Nx_p, int Ny_p, int Nz_p,
                              int Nx_e, int Nz_e);
    std::pair<Mat_d, SparseMat_t<double>> assemblePiezMat(material para, mesh myMesh,
                                                          refGaussInfo Gauss, bool ifOMP = false);
    std::pair<Mat_d, SparseMat_t<double>> assembleElasticMat(material para, mesh myMesh,
                                                             refGaussInfo Gauss, bool ifOMP = false);

    std::pair<Mat_d, SparseMat_t<Complex>>
    assemblePiezPMLMat(Fun_t<Complex> alpha_x,
                       Fun_t<Complex> alpha_y,
                       Fun_t<Complex> alpha_z,
                       material para, mesh myMesh,
                       refGaussInfo Gauss, bool ifOMP = false); // 压电块的PML组装
    std::pair<Mat_d, SparseMat_t<Complex>>
    assembleElaPMLMat(Fun_t<Complex> alpha_x,
                      Fun_t<Complex> alpha_y,
                      Fun_t<Complex> alpha_z,
                      material para, mesh myMesh,
                      refGaussInfo Gauss); // 非压电块的PML矩阵组装 (TO DO)
    /*边界条件处理*/
    void set2_5DAssumption(double yBehind, double yFront,
                           Mat_d &dofIdx, SparseMat_t<Complex> &K,
                           Vec_i &dof2Nodes); // 将区域前后面处理成周期边界条件
    void treatFixedGroundCond(double zBottom, Mat_d &dofIdx,
                              SparseMat_t<Complex> &K,
                              Vec_i &dof2Nodes, double scale = 1);                   // 处理固定接地边界条件
    void treatTerminalBoundCond(double zIntFace, double xIntFaceL, double xIntFaceR, // 交界面z坐标以及左右x坐标
                                double V0, Mat_d &dofIdx,
                                SparseMat_t<Complex> &K,
                                Vec_t<Complex> &F,
                                Vec_i &dof2Nodes, bool ifOMP = false);
    void treatFloatPotentialCond(double zIntFace, double xIntFaceL, double xIntFaceR, // 交界面z坐标以及左右x坐标
                                 Mat_d &dofIdx, SparseMat_t<Complex> &K, Vec_i &dof2Nodes);
    void treatPMLDirBoundCond(Mat_d dom, Mat_d &dofIdx, SparseMat_t<Complex> &K, Vec_i &dof2Nodes, pmlPosition pos, double scale);
    /*子结构定义*/
    enum typeBaseStructure
    {
        piez = -1,         // 压电基底
        linearElastic = 0, // 线弹性部分
        IDT = 1,           // IDT单根指条
        refGratOpen = -2,  // 开路反射栅
        refGratShort = -3, // 短路反射栅
        piezPML = -4,
        none // 表示没有任何意义，仅占位
    };
    /**************************器件基本结构************************************************* */
    class baseStructure
    {
    private:
        std::vector<Idx> nPt_sub, nPt_Ele;

    public:
        void saveNmeshPt(Idx Nx_p = -1, Idx Ny_p = -1, Idx Nz_p = -1,
                         Idx Nx_e = -1, Idx Ny_e = -1, Idx Nz_e = -1) // 存储网格点数，如果需要用的话
        {
            nPt_sub = std::vector<Idx>{Nx_p, Ny_p, Nz_p};
            nPt_Ele = std::vector<Idx>{Nx_e, Ny_e, Nz_e};
        }
        Idx readNmeshPt(Idx str, Idx dim)
        {
            Idx N;
            if (str == 0)
            {
                N = nPt_sub[dim];
            }
            else if (str == 1)
            {
                N = nPt_Ele[dim];
            }
            if (N == -1)
            {
                std::cout << "baseStructure::readNmeshPt:请注意，此结构的此方向网格点数未存储！\n";
            }
            return N;
        }

    public:
        /*材料参数*/
        // 基底或一切方块几何体上定义的问题用这部分
        typeBaseStructure type = none;
        Mat_d dom_p;
        material para_p;
        // 电极 只有反射栅和IDT需要初始化这部分
        Mat_d dom_e;
        material para_e;
        double V0 = 0; // 电极电压
        // PML
        Fun_t<Complex> alpha_x, alpha_y, alpha_z;
        pmlPosition pmlType; // PML的位置
        /*子结构的有限元离散信息*/
        mesh meshFEM;
        // 自由度索引
        Mat_t<double> dofIdx; // 0--2 列为点坐标，第3列为自由度类型:0--2为位移，3为电势
        Vec_i dof2Nodes;      // 自由度对应的网格点编号，行号与dofIdx对应，值与meshNodes对应 (未检查过！！！若后续使用出问题请检查)
        // 子问题矩阵与右端项
        SparseMat_t<Complex> K;
        Vec_t<Complex> F;
        // 构造函数
        baseStructure() {};
        baseStructure(typeBaseStructure typeBaseStructure, Mat_d dom_p, typeMaterial typematerial_p, double w = 0) // 初始化光板
            : dom_p(dom_p), type(typeBaseStructure)
        {
            para_p = materialLib(typematerial_p);
            para_p.w = w;
        }
        baseStructure(typeBaseStructure typeBaseStructure, Mat_d dom_p, Mat_d dom_e, // 初始化IDT、反射栅
                      typeMaterial typematerial_p, typeMaterial typematerial_e, double w = 0)
            : dom_p(dom_p), dom_e(dom_e), type(typeBaseStructure)
        {
            para_p = materialLib(typematerial_p);
            para_e = materialLib(typematerial_e);
            para_p.w = w;
            para_e.w = w;
        }
        baseStructure(typeBaseStructure typeBaseStructure, Mat_d dom_pml, typeMaterial typematerial_pml, // 初始化PML
                      Fun_t<Complex> alpha_x, Fun_t<Complex> alpha_y, Fun_t<Complex> alpha_z,
                      pmlPosition pmlType, double w = 0) : dom_p(dom_pml),
                                                           type(typeBaseStructure),
                                                           alpha_x(alpha_x), alpha_y(alpha_y),
                                                           alpha_z(alpha_z), pmlType(pmlType)
        {
            para_p = materialLib(typematerial_pml);
            para_p.w = w;
        }
        // 设置电极电压
        void setTerminalVoltage(double V)
        {
            if (type == IDT)
                V0 = V;
            else
            {
                std::cout << "只有IDT结构可以设置电压！";
                std::exit(EXIT_FAILURE);
            }
        }
        // 矩阵组装
        void initLinearSystem(const refGaussInfo &Gauss, const mesh &myMesh, bool ifOMP = false);                       // 初始化立方体类子结构的矩阵，未作边界条件处理
        void initLinearSystem(const refGaussInfo &Gauss, const mesh &myMesh1, const mesh &myMesh2, bool ifOMP = false); // 初始化IDT类子结构的矩阵，未作边界条件处理
        // 边界条件处理函数
        void myTreatPeriodBoundCond()
        {
            set2_5DAssumption(dom_p(1, 0), dom_p(1, 1), dofIdx, K, dof2Nodes);
        }
        void myTreatFloatPotentialCond() // 处理悬浮电势
        {
            if (type != refGratOpen)
            {
                std::cout << "myTreatFloatPotentialCond：该子结构不是refGratOpen！\n";
                std::exit(EXIT_FAILURE);
            }
            treatFloatPotentialCond(dom_e(2, 0), dom_e(0, 0), dom_e(0, 1),
                                    dofIdx, K, dof2Nodes);
        }
        void myTreatTerminalBoundCond(bool ifOMP = false) // 处理电极电压边界条件，因涉及右端项的计算，请最后调用
        {
            if (type != IDT)
            {
                std::cout << "myTreatFixedGroundCond：该子结构不是IDT！\n";
                std::exit(EXIT_FAILURE);
            }
            treatTerminalBoundCond(dom_e(2, 0), dom_e(0, 0), dom_e(0, 1), V0, dofIdx, K, F, dof2Nodes, ifOMP);
        }
        void myTreatFixedGroundCond() // 处理固定接地边界条件
        {
            double scale = dom_p(2, 1) - dom_p(2, 0);
            treatFixedGroundCond(dom_p(2, 0), dofIdx, K, dof2Nodes, scale);
        }
        void myTreatPMLDirBoundCond() // 处理PML外边界
        {
            double scale = (dom_p.col(1) - dom_p.col(0)).minCoeff();
            treatPMLDirBoundCond(dom_p, dofIdx, K, dof2Nodes, pmlType, scale);
        }
    };
    /**************************SAW器件序列************************************************* */
    class deviceArray
    {
        /*数据*/
    private:
        struct geoDom // 子结构的几何区域
        {
            Mat_d dom_e;
            Mat_d dom_p;
        };
        struct SubProb // 区域分解子问题
        {
            Idx tag;
            Mat_d dom_p, dom_e;
            Mat_t<double> dofIdx;
            SparseMat_t<Complex> K;
            Vec_t<Complex> F;
            // 下面这两个变量因为暂时用不到，先不进行考虑
            Vec_i dof2Nodes;
            mesh meshFEM;
        };
        struct DimScales // 无量纲化特征量
        {
            double L0 = 1.0;     // 特征长度
            double U0 = 1.0;     // 位移尺度
            double c0 = 1.0;     // 刚度尺度
            double e0 = 1.0;     // 压电系数尺度
            double epcl0 = 1.0;  // 介电常数尺度
            double rho0 = 1.0;   // 密度尺度
            double omega0 = 1.0; // 频率尺度
            double Phi0 = 1.0;   // 电势尺度
            double sigma0 = 1.0; // 应力尺度
            double D0 = 1.0;     // 电位移尺度
            double v0 = 1.0;     // 波速尺度
            double k0 = 1.0;     // 波数尺度
        };
        /* 子结构数据*/
        Eigen::Vector3d ori = Eigen::Vector3d(0, 0, 0);
        std::vector<baseStructure> subStructures; // 器件子结构，索引号表示子结构类型编号
        std::vector<SubProb> subProbs;            // 子问题（用于区域分解），索引号表示子问题类型编号
        int numSubStructures = 0;                 // 子结构种类数

    public:
        Mat_i baseStructureArray; // 器件子结构序列
        Eigen::Matrix<geoDom, Eigen::Dynamic,
                      Eigen::Dynamic>
            geoArray;        // 子结构的实际几何阵列
        DimScales dimScales; // 无量纲化特征量
        /*全局问题数据*/
        SparseMat_t<Complex> K;
        Vec_t<Complex> F;
        Vec_t<Complex> sol;
        Mat_t<double> dofIdx;
        /*初始化*/
    public:
        deviceArray() {} // 空构造函数，若要使用现成的器件结构使用这个定义变量
        deviceArray(std::vector<baseStructure> &baseStructures)
        {
            numSubStructures = baseStructures.size();
            subStructures.reserve(numSubStructures);
            subStructures = std::move(baseStructures);
        }
        void setDeviceArray(Mat_i &deviceArray, Eigen::Vector3d ori = Eigen::Vector3d(0, 0, 0));
        /*无量纲化*/
    public:
        void genDimScales(double U0);
        void dimensionless();      // 无量纲化
        void recoverDimSolution(); // 还原解的量纲
        /* 获取私有变量取值*/
    public:
        baseStructure getSubStructure(Idx index) const
        {
            return subStructures.at(index);
        }
        int getNumBaseStructure() const { return numSubStructures; }
        /*区域分解*/
    private:
        std::map<std::vector<Idx>, Idx> extractBandedSubstructure(); // 提取所有带状子结构，并将DeviceArray更新为新剖分
    public:
        void genBandedSubProbs(); // 带状区域分解
                                  // 生成带状区域分解的子问题，
                                  // 该函数会清空subStructures，
                                  // 并使用新的子问题编号生成deviceArray、geoArray
        /*全局问题组装*/
    public:
        void formGlobalSystem(); // 形成全局系统，并清除所有的局部数据
        /*一些常用的器件模板*/
    private:
        Mat_i genbaResArray(Idx nIDT, Idx nRef, Idx nBar); // basicResonator的子函数，形成基本谐振器模型的子结构序列
        using Fun = std::function<Eigen::VectorXcd(const Eigen::MatrixX3d &)>;
        void initBaResPMLInfo(Idx nRef, Idx nBar, typeMaterial materialSub,
                              double dx, double dz, Fun alpha_x, Fun alpha_z);                  // basicResonator的子函数，根据器件子结构类型初始化PML信息
        void formBaResLoPro4SubStrs(myFEM::meshType mType, myFEM::elemType eType, Idx quaOrder, // 形成基本Resonator问题的子结构的问题
                                    Idx NyDev, Idx NzDev, Idx NxIDT, Idx NxRef, Idx NxBar,
                                    Idx NxEleIDT, Idx NxEleRef, Idx NzEle,
                                    Idx NxPML, Idx NzPML);
        void formGeoArray(Eigen::Vector3d myori);

    public:
        void basicResonator(                                                  // 基本的谐振器，一对反射栅，一组叉指换能器，多项式PML
            Idx nIDT, Idx nRef, Idx nBar,                                     // 叉指换能器、反射栅和光板的个数，其中反射栅和光板个数为单组的指条数
            typeMaterial materialSub, typeMaterial materialEle,               // 压电基底材料和电极材料
            double xIDT, double xRef, double xBar,                            // 叉指换能器、反射栅和光板的宽度（x方向的长度）
            double yDev, double zDev,                                         // 器件在y方向和z方向的长度
            double metCovRat, double zEle,                                    // 金属覆盖率、电极高度
            double omega, double V0,                                          // 工作频率、终端电压
            double xDamp, double zDamp, Idx nx, Idx nz, double dx, double dz, // PML的阻尼系数、多项式次数和深度
            Idx elemPtsPerWaveLen = 20,
            myFEM::meshType mType = myFEM::hex, myFEM::elemType eType = myFEM::Q2, // 网格类型和有限元类型
            Idx quaOrder = 4,                                                      // Gauss积分阶数
            double U0 = 1e-14,
            bool ifNd = true); // 是否执行无量纲化
    };
}
