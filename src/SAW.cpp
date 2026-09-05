#include <SAW.hpp>
#include <myEigen.hpp>
#include <myOMP.hpp>
#include <mat2eigen.hpp>
namespace SAW2_5D
{
    std::pair<std::vector<Idx>, std::vector<Idx>>
    findCommonFaceDof(const Mat_d &dofIdx1, const Mat_d &dofIdx2,
                      int interFaceDim, double interFacePt1, double interFacePt2,
                      double scale)
    {
        // 返回两组点在各自的一个平行与同一坐标平面上的对应点的索引
        std::unordered_map<Vec_d, Idx, PointNdHash, PointNdEqual> mapB;
        mapB.reserve(static_cast<std::size_t>(dofIdx2.rows()));
        double eps = EPS * scale;
        for (Idx j = 0; j < dofIdx2.rows(); ++j)
        {
            if (almostEqual(dofIdx2(j, interFaceDim), interFacePt2, eps))
            {
                // W << V.head(i), V.tail(V.size() - i - 1);
                Vec_d ptTemp(dofIdx2.cols() - 1);
                ptTemp << dofIdx2.row(j).head(interFaceDim),
                    dofIdx2.row(j).tail(dofIdx2.cols() - interFaceDim - 1);
                mapB.emplace(ptTemp, j);
            }
        }
        std::vector<Idx> Idx1, Idx2;
        for (Idx i = 0; i < dofIdx1.rows(); ++i)
        {
            if (!almostEqual(dofIdx1(i, interFaceDim), interFacePt1, eps))
            {
                continue;
            }
            Vec_d ptTemp(dofIdx1.cols() - 1);
            ptTemp << dofIdx1.row(i).head(interFaceDim),
                dofIdx1.row(i).tail(dofIdx1.cols() - interFaceDim - 1);
            auto it = mapB.find(ptTemp);
            if (it != mapB.end())
            {
                Idx1.push_back(i);
                Idx2.push_back(it->second);
            }
        }
        return {Idx1, Idx2};
    }
    void translate3dPoints(Mat_d &Pt,
                           Eigen::Vector3d oriOld,
                           Eigen::Vector3d oriNew) // 将离散点平移到新的坐标系
    {                                              // 要保证输入的Pt前三列依次为x,y,z坐标
        Eigen::Vector3d d = oriNew - oriOld;
        Pt.leftCols(3).rowwise() += d.transpose();
    }
    Idx VoigtIdx(Idx i, Idx j)
    {
        if (!((0 <= i && i <= 2) || (0 <= j && j <= 2)))
        {
            std::cout << "0<=i,j<=2!\n";
            std::exit(EXIT_FAILURE);
        }
        i += 1;
        j += 1;
        Idx n = i + j;
        if (i == j)
            n = i;
        else if (n == 5)
            n = 4;
        else if (n == 4)
            n = 5;
        else if (n == 3)
            n = 6;
        return n - 1;
    }
    material materialLib(typeMaterial type)
    {
        material para;
        switch (type)
        {
        case testPiez:
        {
            para.type = Piez;
            para.c << 19.8390000000000, 6.63117268564586, 5.35382731435414, 0.695673379259194, 0, 0,
                6.63117268564586, 18.6562584117760, 8.04969279896124, 0.631377898645344, 0, 0,
                5.35382731435414, 8.04969279896124, 20.8993559903016, 0.609658991742351, 0, 0,
                0.695673379259194, 0.631377898645342, 0.609658991742350, 7.50169279896124, 0, 0,
                0, 0, 0, 0, 5.66226605280042, -0.400518217501112,
                0, 0, 0, 0, -0.400518217501112, 7.48623394719958;
            para.e << 0, 0, 0, 0, 4.39766045109690, 0.364804820223411,
                -1.72228758113057, 4.53328790352236, -1.35188262586998, 0.217350207764581, 0, 0,
                1.72630399637011, -2.45400624696221, 2.59528773664003, 0.735209775484002, 0, 0;
            para.epcl << 4.03750964455200, 0, 0,
                0, 3.38978563362752, -0.829048927753329,
                0, -0.829048927753330, 2.97637540679548;
            para.rho = 4628;
            break;
        }
        case LN128YX:
        {
            para.type = Piez;
            para.c << 198390000000.000, 66311726856.4586, 53538273143.5414, 6956733792.59194, 0, 0,
                66311726856.4586, 186562584117.760, 80496927989.6124, 6313778986.45344, 0, 0,
                53538273143.5414, 80496927989.6124, 208993559903.016, 6096589917.42351, 0, 0,
                6956733792.59194, 6313778986.45342, 6096589917.42350, 75016927989.6124, 0, 0,
                0, 0, 0, 0, 56622660528.0042, -4005182175.01112,
                0, 0, 0, 0, -4005182175.01112, 74862339471.9958;
            para.e << 0, 0, 0, 0, 4.39766045109690, 0.364804820223411,
                -1.72228758113057, 4.53328790352236, -1.35188262586998, 0.217350207764581, 0, 0,
                1.72630399637011, -2.45400624696221, 2.59528773664003, 0.735209775484002, 0, 0;
            para.epcl << 4.03750964455200e-10, 0, 0,
                0, 3.38978563362752e-10, -8.29048927753329e-11,
                0, -8.29048927753330e-11, 2.97637540679548e-10;
            para.rho = 4628;
            break;
        }
        case Al:
        {
            para.type = LinearEla;
            para.lambda = 61000000000.0000;
            para.mu = 25000000000.0000;
            para.rho = 2695;
            break;
        }
        case testEla:
        {
            para.type = LinearEla;
            para.lambda = 6.10000000000000;
            para.mu = 2.50000000000000;
            para.rho = 2695;
            break;
        }
        default:
        {
            std::cout << "材料未收录!\n";
            std::exit(EXIT_FAILURE);
            break;
        }
        }
        return para;
    }
    meshStep genMeshStepOfIDT(Mat_d dom1, Mat_d dom2,
                              int Nx_p, int Ny_p, int Nz_p,
                              int Nx_e, int Nz_e)
    {
        double scale_y = dom2(2, 1) - dom2(2, 0);
        double scale_z = dom2(2, 1) - dom2(2, 0);
        double eps_y = EPS * scale_y;
        double eps_z = EPS * scale_z;
        if (dom1(0, 0) > dom2(0, 0) ||
            dom1(0, 1) < dom2(0, 1) ||
            std::abs(dom1(1, 0) - dom2(1, 0)) > eps_y ||
            std::abs(dom1(1, 1) - dom2(1, 1)) > eps_y ||
            std::abs(dom1(2, 1) - dom2(2, 0)) > eps_z)
        {
            std::cout << "电极位置不正确，请检查后重新输入！\n";
            std::exit(EXIT_FAILURE);
        }
        if (Nx_p <= Nx_e)
        {
            std::cout << "基底x方向的网格点数必须大于电极\n";
            std::exit(EXIT_FAILURE);
        }
        Vec_d yGrid = Vec_d::LinSpaced(Ny_p, dom1(1, 0), dom1(1, 1));
        Vec_d zGrid_p = Vec_d::LinSpaced(Nz_p, dom1(2, 0), dom1(2, 1));
        Vec_d zGrid_e = Vec_d::LinSpaced(Nz_e, dom2(2, 0), dom2(2, 1));
        Vec_d xGrid_e = Vec_d::LinSpaced(Nx_e, dom2(0, 0), dom2(0, 1));
        int Nx_p_l, Nx_p_r;
        Nx_p_l = floor((Nx_p - Nx_e) / 2.0 + 1.0);
        Nx_p_r = ceil((Nx_p - Nx_e) / 2.0 + 1.0);
        Vec_d xGrid_p_l = Vec_d::LinSpaced(Nx_p_l, dom1(0, 0), dom2(0, 0));
        Vec_d xGrid_p_r = Vec_d::LinSpaced(Nx_p_r, dom2(0, 1), dom1(0, 1));
        Vec_d xGrid_p(xGrid_p_l.size() + xGrid_p_r.size() + xGrid_e.size() - 2);
        xGrid_p << xGrid_p_l, xGrid_e.segment(1, Nx_e - 2), xGrid_p_r;
        return meshStep{xGrid_p, yGrid, zGrid_p, xGrid_e, zGrid_e};
    }
    std::pair<Mat_d, SparseMat_t<double>>
    assemblePiezMat(material para, mesh myMesh, refGaussInfo Gauss, bool ifOMP)
    {
        Idx (*I)(Idx, Idx) = &VoigtIdx;
        SparseMat_t<double> A00;
        std::vector<SparseMat_t<double>>
            baseMat(6); // {A11,A22,A33,A23,A13,A12}
        if (ifOMP)
        {
#pragma omp parallel for schedule(dynamic) collapse(2)
            for (Idx Dtest = 0; Dtest < 3; ++Dtest) // 行
            {
                for (Idx Dtrail = 0; Dtrail < 3; ++Dtrail) // 列
                {
                    if (Dtrail >= Dtest)
                    {
                        Eigen::Vector3i dTest(0, 0, 0);
                        dTest(Dtest) = 1;
                        Eigen::Vector3i dTrail(0, 0, 0);
                        dTrail(Dtrail) = 1;
                        baseMat[I(Dtest, Dtrail)] = assembleMat(myMesh, Gauss, dTest, dTrail, false);
                    }
                }
            }
        }
        else
        {
#pragma omp simd collapse(2)
            for (Idx Dtest = 0; Dtest < 3; ++Dtest) // 行
            {
                for (Idx Dtrail = 0; Dtrail < 3; ++Dtrail) // 列
                {
                    if (Dtrail >= Dtest)
                    {
                        Eigen::Vector3i dTest(0, 0, 0);
                        dTest(Dtest) = 1;
                        Eigen::Vector3i dTrail(0, 0, 0);
                        dTrail(Dtrail) = 1;
                        baseMat[I(Dtest, Dtrail)] = assembleMat(myMesh, Gauss, dTest, dTrail, false);
                    }
                }
            }
        }
        // 组装Muu
        SparseMat_t<double> Muu;
        if (para.w > 0)
        {
            A00 = assembleMat(myMesh, Gauss, Eigen::Vector3i{0, 0, 0}, Eigen::Vector3i{0, 0, 0}, ifOMP);
            A00 *= para.rho;
            Muu = myEigen::blkdiag<double>({A00, A00, A00}, ifOMP);
        }
        // 组装K的四个块Kuu,Kup,Kpp
        SparseMat_t<double> Kuu, Kup, Kpp;
        std::vector<std::vector<SparseMat_t<double>>> blkKuu(3), blkKup(3);
        for (int i = 0; i < 3; ++i)
        {
            blkKuu[i].resize(3);
            blkKup[i].resize(1);
            blkKup[i][0].resize(myMesh.nPts, myMesh.nPts);
            for (int j = 0; j < 3; ++j)
                blkKuu[i][j].resize(myMesh.nPts, myMesh.nPts);
        }
        if (ifOMP)
        {
#pragma omp parallel
            {
                // 组装Kuu
#pragma omp for collapse(2) schedule(dynamic)
                for (Idx i = 0; i < 3; ++i) // uil=\int cijkl*djvi*dkul
                {
                    for (Idx l = 0; l < 3; ++l)
                    {
                        if (i <= l)
                        {
                            for (Idx j = 0; j < 3; ++j)
                            {
                                for (Idx k = 0; k < 3; ++k)
                                {
                                    if (j <= k)
                                    {
                                        blkKuu[i][l] += para.c(I(i, j), I(k, l)) * baseMat[I(j, k)];
                                    }
                                    else
                                    {
                                        blkKuu[i][l] += para.c(I(i, j), I(k, l)) *
                                                        SparseMat_t<double>(baseMat[I(k, j)].transpose());
                                    }
                                }
                            }
                            if (i != l)
                                blkKuu[l][i] = blkKuu[i][l].transpose();
                        }
                    }
                }
// 组装Kup
#pragma omp for
                for (Idx i = 0; i < 3; ++i)
                {
                    for (Idx j = 0; j < 3; ++j)
                    {
                        for (Idx k = 0; k < 3; ++k)
                        {
                            if (j <= k)
                            {
                                blkKup[i][0] += para.e(k, I(i, j)) * baseMat[I(j, k)];
                            }
                            else
                            {
                                blkKup[i][0] +=
                                    para.e(k, I(i, j)) *
                                    SparseMat_t<double>(baseMat[I(k, j)].transpose());
                            }
                        }
                    }
                }
            }
        }
        else
        {
// 组装Kuu
#pragma omp simd collapse(2)
            for (Idx i = 0; i < 3; ++i) // uil=\int cijkl*djvi*dkul
            {
                for (Idx l = 0; l < 3; ++l)
                {
                    if (i <= l)
                    {
                        for (Idx j = 0; j < 3; ++j)
                        {
                            for (Idx k = 0; k < 3; ++k)
                            {
                                if (j <= k)
                                {
                                    blkKuu[i][l] += para.c(I(i, j), I(k, l)) * baseMat[I(j, k)];
                                }
                                else
                                {
                                    blkKuu[i][l] += para.c(I(i, j), I(k, l)) *
                                                    SparseMat_t<double>(baseMat[I(k, j)].transpose());
                                }
                            }
                        }
                        if (i != l)
                            blkKuu[l][i] = blkKuu[i][l].transpose();
                    }
                }
            }
// 组装Kup
#pragma omp simd
            for (Idx i = 0; i < 3; ++i)
            {
                for (Idx j = 0; j < 3; ++j)
                {
                    for (Idx k = 0; k < 3; ++k)
                    {
                        if (j <= k)
                        {
                            blkKup[i][0] += para.e(k, I(i, j)) * baseMat[I(j, k)];
                        }
                        else
                        {
                            blkKup[i][0] +=
                                para.e(k, I(i, j)) *
                                SparseMat_t<double>(baseMat[I(k, j)].transpose());
                        }
                    }
                }
            }
        }
        // 组装Kpp
        Kpp.resize(myMesh.nPts, myMesh.nPts);
        for (Idx i = 0; i < 3; ++i)
        {
            for (Idx j = 0; j < 3; ++j)
            {
                if (i <= j)
                    Kpp -= para.epcl(i, j) * baseMat[I(i, j)];
                else
                    Kpp -= para.epcl(i, j) * SparseMat_t<double>(baseMat[I(j, i)].transpose());
            }
        }
        // 拼装Kuu和Kup
        // wrMatFile::saveEigenSparseMat2Mat("Kuu11.mat", blkKuu[0][0]);
        // wrMatFile::saveEigenSparseMat2Mat("Kuu22.mat", blkKuu[1][1]);
        // wrMatFile::saveEigenSparseMat2Mat("Kuu33.mat", blkKuu[2][2]);
        Kuu = myEigen::blkMat<double>(blkKuu, ifOMP);
        // wrMatFile::saveEigenSparseMat2Mat("Kuu.mat", Kuu);
        Kup = myEigen::blkMat<double>(blkKup, ifOMP);
        if (para.w != 0)
            Kuu -= para.w * para.w * Muu;
        SparseMat_t<double> Kpu = Kup.transpose();
        // wrMatFile::saveEigenSparseMat2Mat("Kuu.mat", Kuu);
        // wrMatFile::saveEigenSparseMat2Mat("Kup.mat", Kup);
        // wrMatFile::saveEigenSparseMat2Mat("Kpp.mat", Kpp);
        SparseMat_t<double> K = myEigen::blkMat<double>({{Kuu, Kup}, {Kpu, Kpp}}, ifOMP);
        // 形成dofIdx
        Mat_d dofIdx(4 * myMesh.nPts, 4);
        dofIdx << myMesh.nodes, Vec_d::Constant(myMesh.nPts, 0),
            myMesh.nodes, Vec_d::Constant(myMesh.nPts, 1),
            myMesh.nodes, Vec_d::Constant(myMesh.nPts, 2),
            myMesh.nodes, Vec_d::Constant(myMesh.nPts, 3);
        return {dofIdx, K};
    }
    std::pair<Mat_d, SparseMat_t<double>> assembleElasticMat(material para, mesh myMesh,
                                                             refGaussInfo Gauss, bool ifOMP)
    {
        Idx (*I)(Idx, Idx) = &VoigtIdx;
        Mat_d dofIdx(myMesh.nPts * 3, 4);
        SparseMat_t<double> K;
        std::vector<SparseMat_t<double>> baseMat(6); // {A11,A22,A33,A23,A13,A12}
        if (ifOMP)
        {
#pragma omp parallel for schedule(dynamic) collapse(2)
            for (Idx Dtest = 0; Dtest < 3; ++Dtest) // 行
            {
                for (Idx Dtrail = 0; Dtrail < 3; ++Dtrail) // 列
                {
                    if (Dtrail >= Dtest)
                    {
                        Eigen::Vector3i dTest(0, 0, 0);
                        dTest(Dtest) = 1;
                        Eigen::Vector3i dTrail(0, 0, 0);
                        dTrail(Dtrail) = 1;
                        baseMat[I(Dtest, Dtrail)] = assembleMat(myMesh, Gauss, dTest, dTrail, false);
                    }
                }
            }
        }
        else
        {
#pragma omp simd collapse(2)
            for (Idx Dtest = 0; Dtest < 3; ++Dtest) // 行
            {
                for (Idx Dtrail = 0; Dtrail < 3; ++Dtrail) // 列
                {
                    if (Dtrail >= Dtest)
                    {
                        Eigen::Vector3i dTest(0, 0, 0);
                        dTest(Dtest) = 1;
                        Eigen::Vector3i dTrail(0, 0, 0);
                        dTrail(Dtrail) = 1;
                        baseMat[I(Dtest, Dtrail)] = assembleMat(myMesh, Gauss, dTest, dTrail, false);
                    }
                }
            }
        }
        // 质量阵
        SparseMat_t<double> A00;
        SparseMat_t<double> Muu;
        if (para.w > 0)
        {
            A00 = assembleMat(myMesh, Gauss, Eigen::Vector3i{0, 0, 0}, Eigen::Vector3i{0, 0, 0}, ifOMP);
            A00 *= para.rho;
            Muu = myEigen::blkdiag<double>({A00, A00, A00}, ifOMP);
        }
        // 刚度矩阵
        std::vector<std::vector<SparseMat_t<double>>> blkK(3);
        for (int i = 0; i < 3; ++i)
        {
            blkK[i].resize(3);
            for (int j = 0; j < 3; ++j)
                blkK[i][j].resize(myMesh.nPts, myMesh.nPts);
        }
        blkK[0][0] = (para.lambda + 2 * para.mu) * baseMat[I(0, 0)] + para.mu * (baseMat[I(1, 1)] + baseMat[I(2, 2)]);
        blkK[0][1] = para.lambda * baseMat[I(0, 1)] + para.mu * SparseMat_t<double>(baseMat[I(0, 1)].transpose());
        blkK[0][2] = para.lambda * baseMat[I(0, 2)] + para.mu * SparseMat_t<double>(baseMat[I(0, 2)].transpose());
        blkK[1][1] = (para.lambda + 2 * para.mu) * baseMat[I(1, 1)] + para.mu * (baseMat[I(0, 0)] + baseMat[I(2, 2)]);
        blkK[1][2] = para.lambda * baseMat[I(1, 2)] + para.mu * SparseMat_t<double>(baseMat[I(1, 2)].transpose());
        blkK[2][2] = (para.lambda + 2 * para.mu) * baseMat[I(2, 2)] + para.mu * (baseMat[I(0, 0)] + baseMat[I(1, 1)]);
        blkK[1][0] = blkK[0][1].transpose();
        blkK[2][0] = blkK[0][2].transpose();
        blkK[2][1] = blkK[1][2].transpose();
        K = myEigen::blkMat(blkK);
        if (para.w != 0)
            K -= para.w * para.w * Muu;
        dofIdx << myMesh.nodes, Vec_d::Constant(myMesh.nPts, 0),
            myMesh.nodes, Vec_d::Constant(myMesh.nPts, 1),
            myMesh.nodes, Vec_d::Constant(myMesh.nPts, 2);
        return {dofIdx, K};
    }
    std::pair<Mat_d, SparseMat_t<Complex>>
    assemblePiezPMLMat(Fun_t<Complex> alpha_x,
                       Fun_t<Complex> alpha_y,
                       Fun_t<Complex> alpha_z,
                       material para, mesh myMesh,
                       refGaussInfo Gauss, bool ifOMP)
    {
        // alpha_x, alpha_y, alpha_z：PML在各方向坐标变换的导数
        Idx (*I)(Idx, Idx) = &VoigtIdx;
        SparseMat_t<Complex> K;
        Mat_d dofIdx(myMesh.nPts * 4, 4);
        std::vector<Fun_t<Complex>> coef(6);
        // 定义变系数
        Fun_t<Complex> coef00 = [alpha_x, alpha_y, alpha_z](const Eigen::MatrixX3d &x) -> Vec_t<Complex>
        {
            Vec_t<Complex> temp1, temp2, temp3, result;
            temp1 = alpha_x(x);
            temp2 = alpha_y(x);
            temp3 = alpha_z(x);
            result = temp1.array() * temp2.array() * temp3.array();
            return result;
        };
        coef[I(0, 0)] = [alpha_x, alpha_y, alpha_z](const Eigen::MatrixX3d &x) -> Vec_t<Complex>
        {
            Vec_t<Complex> temp1, temp2, temp3, result;
            temp1 = alpha_x(x);
            temp2 = alpha_y(x);
            temp3 = alpha_z(x);
            result = temp2.array() * temp3.array() / temp1.array();
            return result;
        };
        coef[I(0, 1)] = [alpha_z](const Eigen::MatrixX3d &x) -> Vec_t<Complex>
        {
            return alpha_z(x);
        };
        coef[I(0, 2)] = [alpha_y](const Eigen::MatrixX3d &x) -> Vec_t<Complex>
        {
            return alpha_y(x);
        };
        coef[I(1, 1)] = [alpha_x, alpha_y, alpha_z](const Eigen::MatrixX3d &x) -> Vec_t<Complex>
        {
            Vec_t<Complex> temp1, temp2, temp3, result;
            temp1 = alpha_x(x);
            temp2 = alpha_y(x);
            temp3 = alpha_z(x);
            result = temp1.array() * temp3.array() / temp2.array();
            return result;
        };
        coef[I(1, 2)] = [alpha_x](const Eigen::MatrixX3d &x) -> Vec_t<Complex>
        {
            return alpha_x(x);
        };
        coef[I(2, 2)] = [alpha_x, alpha_y, alpha_z](const Eigen::MatrixX3d &x) -> Vec_t<Complex>
        {
            Vec_t<Complex> temp1, temp2, temp3, result;
            temp1 = alpha_x(x);
            temp2 = alpha_y(x);
            temp3 = alpha_z(x);
            result = temp1.array() * temp2.array() / temp3.array();
            return result;
        };
        std::vector<SparseMat_t<Complex>> baseMat(6);
        if (ifOMP)
        {
#pragma omp parallel for schedule(dynamic) collapse(2)
            for (Idx Dtest = 0; Dtest < 3; ++Dtest) // 行
            {
                for (Idx Dtrail = 0; Dtrail < 3; ++Dtrail) // 列
                {
                    if (Dtrail >= Dtest)
                    {
                        Eigen::Vector3i dTest(0, 0, 0);
                        dTest(Dtest) = 1;
                        Eigen::Vector3i dTrail(0, 0, 0);
                        dTrail(Dtrail) = 1;
                        baseMat[I(Dtest, Dtrail)] = assembleMat(coef[I(Dtest, Dtrail)], myMesh, Gauss, dTest, dTrail, false);
                    }
                }
            }
        }
        else
        {
#pragma omp simd collapse(2)
            for (Idx Dtest = 0; Dtest < 3; ++Dtest) // 行
            {
                for (Idx Dtrail = 0; Dtrail < 3; ++Dtrail) // 列
                {
                    if (Dtrail >= Dtest)
                    {
                        Eigen::Vector3i dTest(0, 0, 0);
                        dTest(Dtest) = 1;
                        Eigen::Vector3i dTrail(0, 0, 0);
                        dTrail(Dtrail) = 1;
                        baseMat[I(Dtest, Dtrail)] = assembleMat(coef[I(Dtest, Dtrail)], myMesh, Gauss, dTest, dTrail, false);
                    }
                }
            }
        }
        // 组装Muu
        SparseMat_t<Complex> Muu, A00;
        if (para.w > 0)
        {
            A00 = assembleMat(coef00, myMesh, Gauss, Eigen::Vector3i{0, 0, 0}, Eigen::Vector3i{0, 0, 0}, ifOMP);
            A00 *= para.rho;
            Muu = myEigen::blkdiag<Complex>({A00, A00, A00}, ifOMP);
        }
        SparseMat_t<Complex> Kuu, Kup, Kpp;
        std::vector<std::vector<SparseMat_t<Complex>>> blkKuu(3), blkKup(3);
        for (int i = 0; i < 3; ++i)
        {
            blkKuu[i].resize(3);
            blkKup[i].resize(1);
            blkKup[i][0].resize(myMesh.nPts, myMesh.nPts);
            for (int j = 0; j < 3; ++j)
                blkKuu[i][j].resize(myMesh.nPts, myMesh.nPts);
        }
        if (ifOMP)
        {
#pragma omp parallel
            {
                // 组装Kuu
#pragma omp for collapse(2) schedule(dynamic)
                for (Idx i = 0; i < 3; ++i) // uil=\int cijkl*djvi*dkul
                {
                    for (Idx l = 0; l < 3; ++l)
                    {
                        if (i <= l)
                        {
                            for (Idx j = 0; j < 3; ++j)
                            {
                                for (Idx k = 0; k < 3; ++k)
                                {
                                    if (j <= k)
                                    {
                                        blkKuu[i][l] += para.c(I(i, j), I(k, l)) * baseMat[I(j, k)];
                                    }
                                    else
                                    {
                                        blkKuu[i][l] += para.c(I(i, j), I(k, l)) *
                                                        SparseMat_t<Complex>(baseMat[I(k, j)].transpose());
                                    }
                                }
                            }
                            if (i != l)
                                blkKuu[l][i] = blkKuu[i][l].transpose();
                        }
                    }
                }
// 组装Kup
#pragma omp for
                for (Idx i = 0; i < 3; ++i)
                {
                    for (Idx j = 0; j < 3; ++j)
                    {
                        for (Idx k = 0; k < 3; ++k)
                        {
                            if (j <= k)
                            {
                                blkKup[i][0] += para.e(k, I(i, j)) * baseMat[I(j, k)];
                            }
                            else
                            {
                                blkKup[i][0] +=
                                    para.e(k, I(i, j)) *
                                    SparseMat_t<Complex>(baseMat[I(k, j)].transpose());
                            }
                        }
                    }
                }
            }
        }
        else
        {
#pragma omp simd collapse(2)
            for (Idx i = 0; i < 3; ++i) // uil=\int cijkl*djvi*dkul
            {
                for (Idx l = 0; l < 3; ++l)
                {
                    if (i <= l)
                    {
                        for (Idx j = 0; j < 3; ++j)
                        {
                            for (Idx k = 0; k < 3; ++k)
                            {
                                if (j <= k)
                                {
                                    blkKuu[i][l] += para.c(I(i, j), I(k, l)) * baseMat[I(j, k)];
                                }
                                else
                                {
                                    blkKuu[i][l] += para.c(I(i, j), I(k, l)) *
                                                    SparseMat_t<Complex>(baseMat[I(k, j)].transpose());
                                }
                            }
                        }
                        if (i != l)
                            blkKuu[l][i] = blkKuu[i][l].transpose();
                    }
                }
            }
// 组装Kup
#pragma omp simd
            for (Idx i = 0; i < 3; ++i)
            {
                for (Idx j = 0; j < 3; ++j)
                {
                    for (Idx k = 0; k < 3; ++k)
                    {
                        if (j <= k)
                        {
                            blkKup[i][0] += para.e(k, I(i, j)) * baseMat[I(j, k)];
                        }
                        else
                        {
                            blkKup[i][0] +=
                                para.e(k, I(i, j)) *
                                SparseMat_t<Complex>(baseMat[I(k, j)].transpose());
                        }
                    }
                }
            }
        }
        // 组装Kpp
        Kpp.resize(myMesh.nPts, myMesh.nPts);
        for (Idx i = 0; i < 3; ++i)
        {
            for (Idx j = 0; j < 3; ++j)
            {
                if (i <= j)
                    Kpp -= para.epcl(i, j) * baseMat[I(i, j)];
                else
                    Kpp -= para.epcl(i, j) * SparseMat_t<Complex>(baseMat[I(j, i)].transpose());
            }
        }
        // 拼装Kuu和Kup
        Kuu = myEigen::blkMat<Complex>(blkKuu, ifOMP);
        Kup = myEigen::blkMat<Complex>(blkKup, ifOMP);
        if (para.w != 0)
            Kuu -= para.w * para.w * Muu;
        SparseMat_t<Complex> Kpu = Kup.transpose();
        K = myEigen::blkMat<Complex>({{Kuu, Kup}, {Kpu, Kpp}}, ifOMP);
        // 形成dofIdx
        dofIdx << myMesh.nodes, Vec_d::Constant(myMesh.nPts, 0),
            myMesh.nodes, Vec_d::Constant(myMesh.nPts, 1),
            myMesh.nodes, Vec_d::Constant(myMesh.nPts, 2),
            myMesh.nodes, Vec_d::Constant(myMesh.nPts, 3);
        return {dofIdx, K};
    }
    /*subStructure类成员函数实现*/
    void baseStructure::initLinearSystem(const refGaussInfo &Gauss, const mesh &myMesh, bool ifOMP)
    {
        if (type != piez && type != linearElastic && type != piezPML)
        {
            std::cout << "请输入电极区域的网格信息!\n";
            std::exit(EXIT_FAILURE);
        }
        if (type == piezPML)
        {
            auto [dofIdx_temp, K_temp] = assemblePiezPMLMat(alpha_x, alpha_y, alpha_z, para_p, myMesh, Gauss, ifOMP);
            dofIdx = std::move(dofIdx_temp);
            K = std::move(K_temp);
            Vec_i dof2nodes_temp = Vec_i::LinSpaced(myMesh.nPts, 0, myMesh.nPts - 1);
            dof2Nodes.resize(dofIdx.rows());
            dof2Nodes << dof2nodes_temp, dof2nodes_temp, dof2nodes_temp, dof2nodes_temp;
        }
        else if (type == piez)
        {
            // std::cout << "开始组装Bar刚度矩阵\n";
            auto [dofIdx_temp, K_temp] = assemblePiezMat(para_p, myMesh, Gauss, ifOMP);
            // wrMatFile::saveEigenDenseMat2Mat("lodofIdx.mat", dofIdx_temp);
            // wrMatFile::saveEigenSparseMat2Mat("loK.mat", K_temp);
            dofIdx = std::move(dofIdx_temp);
            K = K_temp.cast<Complex>();
            Vec_i dof2nodes_temp = Vec_i::LinSpaced(myMesh.nPts, 0, myMesh.nPts - 1);
            dof2Nodes.resize(dofIdx.rows());
            dof2Nodes << dof2nodes_temp, dof2nodes_temp, dof2nodes_temp, dof2nodes_temp;
        }
        else if (type == linearElastic)
        {
            auto [dofIdx_temp, K_temp] = assembleElasticMat(para_p, myMesh, Gauss, ifOMP);
            dofIdx = std::move(dofIdx_temp);
            K = K_temp.cast<Complex>();
            Vec_i dof2nodes_temp = Vec_i::LinSpaced(myMesh.nPts, 0, myMesh.nPts - 1);
            dof2Nodes.resize(dofIdx.rows());
            dof2Nodes << dof2nodes_temp, dof2nodes_temp, dof2nodes_temp;
        }
        else
        {
            std::cout << "未定义该类子结构!\n";
            std::exit(EXIT_FAILURE);
        }
        meshFEM = myMesh;
    }
    void baseStructure::initLinearSystem(const refGaussInfo &Gauss, const mesh &myMesh1, const mesh &myMesh2, bool ifOMP)
    {
        if (type != IDT && type != refGratOpen && type != refGratShort)
        {
            std::cout << "该子结构非IDT类结构，请使用单网格输入!\n";
            std::exit(EXIT_FAILURE);
        }
        double scale_x = myMesh2.Dom(0, 1) - myMesh2.Dom(0, 0);
        double scale_y = myMesh2.Dom(2, 1) - myMesh2.Dom(2, 0);
        double scale_z = myMesh2.Dom(2, 1) - myMesh2.Dom(2, 0);
        double eps_y = EPS * scale_y;
        double eps_z = EPS * scale_z;
        double eps_x = EPS * scale_x;
        double eps = std::max({eps_x, eps_y, eps_z});
        if (myMesh1.Dom(0, 0) > myMesh2.Dom(0, 0) ||
            myMesh1.Dom(0, 1) < myMesh2.Dom(0, 1) ||
            std::abs(myMesh1.Dom(1, 0) - myMesh2.Dom(1, 0)) > eps_y ||
            std::abs(myMesh1.Dom(1, 1) - myMesh2.Dom(1, 1)) > eps_y ||
            std::abs(myMesh1.Dom(2, 1) - myMesh2.Dom(2, 0)) > eps_z)
        {
            std::cout << "电极位置不正确，请检查后重新输入！\n";
            std::exit(EXIT_FAILURE);
        }
        // 基底
        auto [dofIdx_p, K_p] = assemblePiezMat(para_p, myMesh1, Gauss, ifOMP);
        // 电极
        auto [dofIdx_e, K_e] = assembleElasticMat(para_e, myMesh2, Gauss, ifOMP);
        // 拼接
        // 矩阵和自由度索引
        double interface = myMesh1.Dom(2, 1);
        auto mergeInfo = mergeFEMMat<double>(dofIdx_p, dofIdx_e, K_p, K_e, {2, interface}, eps);
        K = mergeInfo.K.cast<Complex>();
        dofIdx = std::move(mergeInfo.dofIdx);
        // 网格
        auto meshMergeInfo = mergeFEMMesh(myMesh1, myMesh2, {2, interface}, eps);
        Vec_i dof2Nodes_p = Vec_i::LinSpaced(myMesh1.nPts, 0, myMesh1.nPts - 1);
        Vec_i dof2Nodes_e(myMesh2.nPts - meshMergeInfo.nInterPts);
        Idx pos = 0;
        for (int i = 0; i < myMesh2.nPts; ++i)
        {
            if (meshMergeInfo.nodes2_old2new[i] >= myMesh1.nPts)
            {
                dof2Nodes_e[pos] = meshMergeInfo.nodes2_old2new[i];
                ++pos;
            }
        }
        dof2Nodes.resize(4 * dof2Nodes_p.size() + 3 * dof2Nodes_e.size());
        dof2Nodes << dof2Nodes_p, dof2Nodes_p, dof2Nodes_p, dof2Nodes_p,
            dof2Nodes_e, dof2Nodes_e, dof2Nodes_e;
        meshFEM = std::move(meshMergeInfo.myMesh);
    }
    void set2_5DAssumption(double yBehind, double yFront,
                           Mat_d &dofIdx, SparseMat_t<Complex> &K,
                           Vec_i &dof2Nodes)
    {
        // yBehind : 后面的y坐标，yFront：前面的y坐标
        // 提取前后面自由度
        double eps = EPS * (yFront - yBehind);
        std::unordered_map<Vec_d, Idx, PointNdHash, PointNdEqual> mapB;
        mapB.reserve(dofIdx.rows()); // 存储所有后面自由度
        for (Idx j = 0; j < dofIdx.rows(); ++j)
        {
            if (almostEqual(dofIdx(j, 1), yBehind, eps))
            {
                mapB.emplace(Eigen::Vector3d(dofIdx(j, 0), dofIdx(j, 2), dofIdx(j, 3)), j);
            }
        }
        std::vector<Idx> Idx1, Idx2;
        for (Idx i = 0; i < dofIdx.rows(); ++i)
        {
            if (!almostEqual(dofIdx(i, 1), yFront, eps))
            {
                continue;
            }
            Eigen::Vector3d p(dofIdx(i, 0), dofIdx(i, 2), dofIdx(i, 3));
            auto it = mapB.find(p);
            if (it != mapB.end())
            {
                Idx1.push_back(i);
                Idx2.push_back(it->second);
            }
        }
        // 把后面自由度对应的行列加到前面自由度行列，并删掉后面
        std::vector<Idx> mark(K.rows(), -1);
        for (Idx i = 0; i < Idx2.size(); ++i)
            mark[Idx2[i]] = Idx1[i];
        Idx count = 0;
        for (Idx i = 0; i < K.rows(); ++i)
        {
            if (mark[i] == -1)
            {
                mark[i] = count;
                ++count;
            }
        }
        for (Idx i = 0; i < Idx2.size(); ++i)
        {
            mark[Idx2[i]] = mark[Idx1[i]];
        }
        std::vector<Eigen::Triplet<Complex, Idx>> triplets;
        triplets.reserve(K.nonZeros());
        for (Idx col = 0; col < K.outerSize(); ++col)
        {
            for (SparseMat_t<Complex>::InnerIterator iter(K, col); iter; ++iter)
            {
                Idx i = mark[iter.row()];
                Idx j = mark[iter.col()];
                triplets.emplace_back(i, j, iter.value());
            }
        }
        SparseMat_t<Complex> Knew(count, count);
        // K.resize(count, count);
        Knew.setFromTriplets(triplets.begin(), triplets.end());
        K = std::move(Knew);
        dofIdx = myEigen::removeRowsDenseMat<double>(dofIdx, Idx2);
        //  dofIdx = myEigen::delColRowDenMat<double>(dofIdx, delIdx, 0);
        dof2Nodes = myEigen::removeRowsDenseMat<Idx>(dof2Nodes, Idx2);
    }
    void treatFixedGroundCond(double zBottom, Mat_d &dofIdx,
                              SparseMat_t<Complex> &K,
                              Vec_i &dof2Nodes, double scale)
    {
        double eps = scale * EPS;
        std::vector<Idx> IdxBottom;
        IdxBottom.reserve(dofIdx.rows());
        for (Idx i = 0; i < K.rows(); ++i)
        {
            if (almostEqual(dofIdx(i, 2), zBottom, eps))
                IdxBottom.emplace_back(i);
        }
        std::vector<Idx> mark(K.rows(), 0);
        for (Idx i = 0; i < IdxBottom.size(); ++i)
            mark[IdxBottom[i]] = -1;
        Idx count = 0;
        for (Idx i = 0; i < K.rows(); ++i)
        {
            if (mark[i] != -1)
                mark[i] = count++;
        }
        std::vector<Eigen::Triplet<Complex, Idx>> triplets;
        triplets.reserve(K.nonZeros());
        for (Idx col = 0; col < K.outerSize(); ++col)
        {
            for (SparseMat_t<Complex>::InnerIterator iter(K, col); iter; ++iter)
            {
                if (mark[iter.col()] != -1 && mark[iter.row()] != -1)
                    triplets.emplace_back(mark[iter.row()],
                                          mark[iter.col()],
                                          iter.value());
            }
        }
        K.resize(count, count);
        K.setFromTriplets(triplets.begin(), triplets.end());
        dofIdx = myEigen::removeRowsDenseMat<double>(dofIdx, IdxBottom);
        dof2Nodes = myEigen::removeRowsDenseMat<Idx>(dof2Nodes, IdxBottom);
    }
    void treatTerminalBoundCond(double zIntFace, double xIntFaceL, double xIntFaceR, // 交界面z坐标以及左右x坐标
                                double V0, Mat_d &dofIdx,
                                SparseMat_t<Complex> &K,
                                Vec_t<Complex> &F,
                                Vec_i &dof2Nodes, bool ifOMP)
    {
        double eps = (xIntFaceR - xIntFaceL) * EPS;
        // double eps = EPS;
        // std::cout << "eps=" << eps << " \n";
        std::vector<Idx> IdxIntFace;
        IdxIntFace.reserve(dofIdx.rows());
        for (Idx i = 0; i < K.rows(); ++i)
        {
            if (almostEqual(dofIdx(i, 2), zIntFace, eps) &&
                dofIdx(i, 0) - xIntFaceL > -eps &&
                dofIdx(i, 0) - xIntFaceR < eps &&
                almostEqual(dofIdx(i, 3), 3.0, EPS))
            {
                // std::cout << dofIdx.row(i) << "\n";
                IdxIntFace.emplace_back(i);
            }
        }
        // std::cout << "交界面自由度共" << IdxIntFace.size() << "个\n";
        std::vector<Idx> mark(K.rows(), 0);
        for (Idx i = 0; i < IdxIntFace.size(); ++i)
            mark[IdxIntFace[i]] = -1;
        Idx count = 0;
        for (Idx i = 0; i < K.rows(); ++i)
        {
            if (mark[i] != -1)
                mark[i] = count++;
        }
        // 形成右端项
        Vec_t<Complex> F_temp = Vec_t<Complex>::Zero(count);
        if (ifOMP)
        {
#pragma omp parallel for schedule(dynamic)
            for (Idx i = 0; i < mark.size(); ++i)
            {
                if (mark[i] == -1)
                    continue;
                for (Idx j = 0; j < IdxIntFace.size(); ++j)
                    F_temp(mark[i]) -= K.coeff(i, IdxIntFace[j]);
            }
        }
        else
        {
#pragma omp simd
            for (Idx i = 0; i < mark.size(); ++i)
            {
                if (mark[i] == -1)
                    continue;
                for (Idx j = 0; j < IdxIntFace.size(); ++j)
                    F_temp(mark[i]) -= K.coeff(i, IdxIntFace[j]);
            }
        }
        F_temp *= V0;
        // std::set<Idx> delIdx(IdxIntFace.begin(), IdxIntFace.end());
        if (F.size() == 0)
            F = std::move(F_temp);
        else if (F.size() > F_temp.size())
        {
            F = myEigen::removeRowsDenseMat<Complex>(F, IdxIntFace);
            F += F_temp;
        }
        else if (F.size() == F_temp.size())
            F += F_temp;
        else
        {
            std::cout << "右端项F大小错误!\n";
            std::exit(EXIT_FAILURE);
        }
        // 处理矩阵
        std::vector<Eigen::Triplet<Complex, Idx>> triplets;
        triplets.reserve(K.nonZeros());
        for (Idx col = 0; col < K.outerSize(); ++col)
        {
            for (SparseMat_t<Complex>::InnerIterator iter(K, col); iter; ++iter)
            {
                if (mark[iter.col()] != -1 && mark[iter.row()] != -1)
                    triplets.emplace_back(mark[iter.row()],
                                          mark[iter.col()],
                                          iter.value());
            }
        }
        K.resize(count, count);
        K.setFromTriplets(triplets.begin(), triplets.end());
        dofIdx = myEigen::removeRowsDenseMat<double>(dofIdx, IdxIntFace);
        dof2Nodes = myEigen::removeRowsDenseMat<Idx>(dof2Nodes, IdxIntFace);
    }
    void treatFloatPotentialCond(double zIntFace, double xIntFaceL, double xIntFaceR, // 交界面z坐标以及左右x坐标
                                 Mat_d &dofIdx, SparseMat_t<Complex> &K, Vec_i &dof2Nodes)
    {
        double eps = (xIntFaceR - xIntFaceL) * EPS;
        std::vector<Idx> IdxIntFace;
        IdxIntFace.reserve(dofIdx.rows());
        for (Idx i = 0; i < K.rows(); ++i)
        {
            if (almostEqual(dofIdx(i, 2), zIntFace, eps) &&
                dofIdx(i, 0) - xIntFaceL > -eps &&
                dofIdx(i, 0) - xIntFaceR < eps &&
                almostEqual(dofIdx(i, 3), 3.0, EPS))
            {
                // std::cout << dofIdx.row(i) << "\n";
                IdxIntFace.emplace_back(i);
            }
        }
        Idx idxEle = IdxIntFace[0]; // 保第一个电极-基底交界面自由度
        std::vector<Idx> idxDel(IdxIntFace.begin() + 1, IdxIntFace.end());
        std::vector<Idx> mark(K.rows(), -1);
        for (Idx i = 0; i < idxDel.size(); ++i)
            mark[idxDel[i]] = idxEle;
        Idx count = 0;
        for (Idx i = 0; i < dofIdx.rows(); ++i)
        {
            if (mark[i] == -1)
                mark[i] = count++;
        }
        std::vector<Eigen::Triplet<Complex, Idx>> triplets;
        triplets.reserve(K.nonZeros());
        for (Idx col = 0; col < K.outerSize(); ++col)
        {
            for (SparseMat_t<Complex>::InnerIterator iter(K, col); iter; ++iter)
            {
                Idx i = mark[iter.row()];
                Idx j = mark[iter.col()];
                triplets.emplace_back(i, j, iter.value());
            }
        }
        SparseMat_t<Complex> Knew(count, count);
        Knew.setFromTriplets(triplets.begin(), triplets.end());
        K = std::move(Knew);
        dofIdx = myEigen::removeRowsDenseMat<double>(dofIdx, idxDel);
        dof2Nodes = myEigen::removeRowsDenseMat<Idx>(dof2Nodes, idxDel);
    }
    void treatPMLDirBoundCond(Mat_d dom, Mat_d &dofIdx, SparseMat_t<Complex> &K, Vec_i &dof2Nodes, pmlPosition pos, double scale)
    {
        double eps = scale * EPS;
        std::function<bool(Vec_t<double>)> pFunc = nullptr;
        switch (pos)
        {
        case left:
            pFunc = [&dom, &eps](Vec_t<double> x) -> bool
            {
                return almostEqual(x(0), dom(0, 0), eps);
            };
            break;
        case right:
            pFunc = [&dom, &eps](Vec_t<double> x) -> bool
            {
                return almostEqual(x(0), dom(0, 1), eps);
            };
            break;
        case bottom:
            pFunc = [&dom, &eps](Vec_t<double> x) -> bool
            {
                return almostEqual(x(2), dom(2, 0), eps);
            };
            break;
        case leftBottom:
            pFunc = [&dom, &eps](Vec_t<double> x) -> bool
            {
                return almostEqual(x(0), dom(0, 0), eps) ||
                       almostEqual(x(2), dom(2, 0), eps);
            };
            break;
        case rightBottom:
            pFunc = [&dom, &eps](Vec_t<double> x) -> bool
            {
                return almostEqual(x(0), dom(0, 1), eps) ||
                       almostEqual(x(2), dom(2, 0), eps);
            };
            break;
        default:
            std::cout << "treatPMLDirBoundCond:PML位置错误！";
            std::exit(EXIT_FAILURE);
            break;
        }
        std::vector<Idx> idxDel;
        idxDel.reserve(dofIdx.rows());
        for (Idx i = 0; i < K.rows(); ++i)
        {
            if (pFunc(dofIdx.row(i)))
                idxDel.emplace_back(i);
        }
        std::vector<Idx> mark(K.rows(), 0);
        for (Idx i = 0; i < idxDel.size(); ++i)
            mark[idxDel[i]] = -1;
        Idx count = 0;
        for (Idx i = 0; i < K.rows(); ++i)
        {
            if (mark[i] != -1)
                mark[i] = count++;
        }
        std::vector<Eigen::Triplet<Complex, Idx>> triplets;
        triplets.reserve(K.nonZeros());
        for (Idx col = 0; col < K.outerSize(); ++col)
        {
            for (SparseMat_t<Complex>::InnerIterator iter(K, col); iter; ++iter)
            {
                if (mark[iter.col()] != -1 && mark[iter.row()] != -1)
                    triplets.emplace_back(mark[iter.row()],
                                          mark[iter.col()],
                                          iter.value());
            }
        }
        K.resize(count, count);
        K.setFromTriplets(triplets.begin(), triplets.end());
        dofIdx = myEigen::removeRowsDenseMat<double>(dofIdx, idxDel);
        dof2Nodes = myEigen::removeRowsDenseMat<Idx>(dof2Nodes, idxDel);
    }
    void deviceArray::setDeviceArray(Mat_i &deviceArray, Eigen::Vector3d myori)
    {
        // this->ori = myori;
        this->baseStructureArray = std::move(deviceArray);
        geoArray.resize(baseStructureArray.rows(), baseStructureArray.cols());
        double xStart = myori(0);
        for (Idx row = baseStructureArray.rows() - 1; row >= 0; --row)
        {
            double zWide;
            for (Idx col = 0; col < baseStructureArray.cols(); ++col)
            {
                geoDom geoTemp;
                Mat_d dom_p = subStructures[baseStructureArray(row, col)].dom_p;

                double xWide = dom_p(0, 1) - dom_p(0, 0);
                zWide = dom_p(2, 1) - dom_p(2, 0);
                geoTemp.dom_p = Mat_d{{myori(0), myori(0) + xWide},
                                      {myori(1), myori(1) + (dom_p(1, 1) - dom_p(1, 0))},
                                      {myori(2), myori(2) + zWide}};
                if (subStructures[baseStructureArray(row, col)].type == IDT ||
                    subStructures[baseStructureArray(row, col)].type == refGratOpen ||
                    subStructures[baseStructureArray(row, col)].type == refGratShort)
                {
                    Mat_d dom_e = subStructures[baseStructureArray(row, col)].dom_e;
                    geoTemp.dom_e = dom_e;
                    Eigen::Vector3d trans = myori - dom_p.col(0);
                    geoTemp.dom_e.colwise() += trans;
                }
                geoArray(row, col) = geoTemp;
                myori(0) += xWide;
            }
            myori(2) += zWide;
            myori(0) = xStart;
        }
        this->ori = geoArray(baseStructureArray.rows() - 1, 0).dom_p.col(0); // 全局器件原点值
        // std::cout << "器件原点为：" << this->ori << "\n";
    }
    std::map<std::vector<Idx>, Idx> deviceArray::extractBandedSubstructure()
    {
        // std::vector<double> std_vec(eig_vec.data(), eig_vec.data() + eig_vec.size());
        std::map<std::vector<Idx>, Idx> bandedSubstructure;
        Mat_i DDM(1, baseStructureArray.cols());
        Idx idx = 0;
        for (Idx i = 0; i < baseStructureArray.cols(); ++i)
        {
            // Vec_i vecTemp = baseStructureArray.col(i);
            std::vector<Idx> vecTemp(baseStructureArray.col(i).data(),
                                     baseStructureArray.col(i).data() + baseStructureArray.rows());
            auto p = bandedSubstructure.find(vecTemp);
            if (p == bandedSubstructure.end())
            {
                bandedSubstructure.insert({vecTemp, idx});
                DDM(0, i) = idx;
                idx++;
            }
            else
            {
                DDM(0, i) = bandedSubstructure[vecTemp];
            }
        }
        // baseStructureArray.resize((1, baseStructureArray.cols()));
        baseStructureArray = std::move(DDM);
        // numSubStructures = bandedSubstructure.size();
        return bandedSubstructure;
    }
    void deviceArray::genDimScales(double U0)
    {
        // 收集各子结构的特征量
        Idx N = subStructures.size();
        std::vector<double> L0, c0, e0, rho0;
        L0.reserve(2 * N);
        c0.reserve(2 * N);
        e0.reserve(N);
        rho0.reserve(N);
        for (Idx i = 0; i < N; ++i)
        {
            typeBaseStructure thisType = subStructures[i].type;
            if (thisType == piezPML)
            {
                std::cout << "第" << i << "个子结构为PML，不进行无量纲化；若要使用PML请确保自己已进行无量纲化！\n";
                continue;
            }
            else if (thisType == none)
                continue;
            else
            {
                // L0, rho0
                Vec_d width = subStructures[i].dom_p.col(1) - subStructures[i].dom_p.col(0);
                L0.emplace_back(width.minCoeff());
                rho0.emplace_back(subStructures[i].para_p.rho);
                if (subStructures[i].dom_e.rows() > 0)
                {
                    width = subStructures[i].dom_e.col(1) - subStructures[i].dom_e.col(0);
                    L0.emplace_back(width.minCoeff());
                    rho0.emplace_back(subStructures[i].para_e.rho);
                }
                // c0,e0
                if (thisType != linearElastic) // 非线弹性材料
                {
                    double maxC = subStructures[i].para_p.c.array().abs().maxCoeff();
                    c0.emplace_back(maxC);
                    double maxE = subStructures[i].para_p.e.array().abs().maxCoeff();
                    e0.emplace_back(maxE);
                    if (thisType == IDT || thisType == refGratOpen || thisType == refGratShort)
                    {
                        maxC = std::max(subStructures[i].para_e.lambda, subStructures[i].para_e.mu);
                        c0.emplace_back(maxC);
                    }
                }
                else // 线弹性材料
                {
                    double maxC = std::max(subStructures[i].para_p.lambda, subStructures[i].para_p.mu);
                    c0.emplace_back(maxC);
                }
            }
        }
        dimScales.U0 = U0;
        dimScales.c0 = *std::max_element(c0.begin(), c0.end());
        dimScales.L0 = *std::min_element(L0.begin(), L0.end());
        dimScales.e0 = *std::max_element(e0.begin(), e0.end());
        dimScales.epcl0 = dimScales.e0 * dimScales.e0 / dimScales.c0;
        dimScales.rho0 = *std::max_element(rho0.begin(), rho0.end());
        dimScales.Phi0 = dimScales.e0 * dimScales.U0 / dimScales.epcl0;
        dimScales.sigma0 = dimScales.c0 * dimScales.U0 / dimScales.L0;
        dimScales.D0 = dimScales.epcl0 * dimScales.Phi0 / dimScales.L0;
        dimScales.omega0 = std::sqrt(dimScales.c0 / (dimScales.rho0 * dimScales.L0 * dimScales.L0));
        dimScales.v0 = std::sqrt(dimScales.c0 / dimScales.rho0);
        dimScales.k0 = dimScales.omega0 / dimScales.v0;
    }
    void deviceArray::dimensionless()
    {
        Idx N = subStructures.size();
        for (Idx i = 0; i < N; ++i)
        {
            typeBaseStructure thisType = subStructures[i].type;
            if (thisType == piezPML)
            {
                std::cout << "第" << i << "个子结构为PML，不进行无量纲化；若要使用PML请确保自己已进行无量纲化！\n";
                continue;
            }
            else if (thisType == none)
                continue;
            else
            {
                auto ptr = &subStructures[i];

                ptr->dom_p /= dimScales.L0;
                if (ptr->dom_e.rows() > 0) // 有电极的
                    ptr->dom_e /= dimScales.L0;
                // 都有的参数
                ptr->para_p.w /= dimScales.omega0;
                ptr->para_p.rho /= dimScales.rho0;
                ptr->para_p.epcl /= dimScales.epcl0;

                if (thisType == IDT) // 有终端的
                    ptr->V0 /= dimScales.Phi0;

                if (thisType == linearElastic)
                {
                    ptr->para_p.lambda /= dimScales.c0;
                    ptr->para_p.mu /= dimScales.c0;
                }
                else
                {
                    ptr->para_p.c /= dimScales.c0;
                    ptr->para_p.e /= dimScales.e0;
                    if (thisType == IDT || thisType == refGratOpen || thisType == refGratShort)
                    {
                        ptr->para_e.lambda /= dimScales.c0;
                        ptr->para_e.mu /= dimScales.c0;
                        ptr->para_e.w /= dimScales.omega0;
                        ptr->para_e.rho /= dimScales.rho0;
                    }
                }
            }
        }
    }
    void deviceArray::recoverDimSolution()
    {
#pragma omp parallel
        {
            Idx rank = omp_get_thread_num();
            Idx size = omp_get_num_threads();
            auto workList = myOMP::distributeTasks(size, dofIdx.rows());
            Idx tStart = workList[rank].first;
            Idx tEnd = workList[rank].second + tStart;
#pragma omp simd
            for (Idx i = tStart; i < tEnd; ++i)
            {
                if (dofIdx(i, 3) == 3) // 电势
                    sol[i] *= dimScales.Phi0;
                else
                    sol[i] *= dimScales.U0;
                dofIdx.row(i).head(3) = dofIdx.row(i).head(3) * dimScales.L0;
            }
        }
    }
    void deviceArray::genBandedSubProbs()
    {
        // 统计所有子区域类型
        std::map<std::vector<Idx>, Idx> subProbTypes = extractBandedSubstructure();
        /***test******/
        // for (auto p = subProbTypes.begin(); p != subProbTypes.end(); ++p)
        // {
        //     std::cout << "subProbType " << p->second << ": ";
        //     for (Idx i = 0; i < p->first.size(); i++)
        //     {
        //         std::cout << p->first[i] << "\0";
        //     }
        //     std::cout << "\n";
        // }
        /*************/
        // 生成并行用的子结构向量
        std::vector<std::vector<Idx>> subProbTypesVec(subProbTypes.size());
        // subProbTypesVec.reserve(subProbTypes.size());
        for (auto p = subProbTypes.begin(); p != subProbTypes.end(); ++p)
            subProbTypesVec[p->second] = p->first;
        // subProbTypesVec.emplace_back(p->first);
        // 按子区域并行形成子问题
        Idx nSubProbs = subProbTypesVec.size();
        Idx ompSize = omp_get_max_threads();
        auto workList = myOMP::distributeTasks(ompSize, nSubProbs);
        subProbs.resize(nSubProbs);
#pragma omp parallel
        {
            Idx rank = omp_get_thread_num();
            Idx tStart{workList[rank].first},
                tEnd{workList[rank].first + workList[rank].second - 1};
            for (Idx i = tStart; i <= tEnd; ++i)
            {
                Idx idxSubStructure = subProbTypesVec[i][0]; // 当前的子结构编号
                subProbs[i].tag = subProbTypes[subProbTypesVec[i]];
                subProbs[i].dom_p = subStructures[idxSubStructure].dom_p;
                if (subStructures[idxSubStructure].dom_e.rows() > 0)
                    subProbs[i].dom_e = subStructures[idxSubStructure].dom_e;
                subProbs[i].dofIdx = subStructures[idxSubStructure].dofIdx;
                subProbs[i].K = subStructures[idxSubStructure].K;
                subProbs[i].F = subStructures[idxSubStructure].F.size() > 0
                                    ? subStructures[idxSubStructure].F
                                    : Vec_t<Complex>::Zero(subProbs[i].dofIdx.rows());
                Idx idxStart_SearchDof = 0; // 扫描与下一个区域公共点的扫描起点
                for (Idx j = 1; j < subProbTypesVec[i].size(); ++j)
                {
                    idxSubStructure = subProbTypesVec[i][j];
                    Vec_d ori_old, ori_new;
                    ori_old = subStructures[idxSubStructure].dom_p.col(0);
                    Vec_d width = subStructures[idxSubStructure].dom_p.col(1) - subStructures[idxSubStructure].dom_p.col(0);
                    ori_new = subProbs[i].dom_p.col(0);
                    ori_new(2) -= width(2);
                    Mat_d dofIdxTemp = subStructures[idxSubStructure].dofIdx;
                    translate3dPoints(dofIdxTemp, ori_old, ori_new);
                    // 开始拼接
                    double scale = width.minCoeff();
                    auto [Idx1, Idx2] =
                        myFEM::findCommonDof(subProbs[i].dofIdx, dofIdxTemp,
                                             {2, subProbs[i].dom_p(2, 0)}, scale,
                                             idxStart_SearchDof, 0);
                    // #pragma omp critical
                    //                     {
                    //                         std::cout << "线程" << rank << "拼接第" << i << "个子区域的第"
                    //                                   << j << "个子结构，共有" << Idx2.size() << "个界面自由度\n";
                    //                     }
                    subProbs[i].dom_p(2, 0) -= width(2);
                    idxStart_SearchDof = subProbs[i].dofIdx.rows();
                    subProbs[i].K =
                        myFEM::mergeSparseMat<Complex>(
                            subProbs[i].K,
                            subStructures[idxSubStructure].K,
                            Idx1, Idx2);
                    if (subStructures[idxSubStructure].F.size() > 0)
                        subProbs[i].F =
                            myFEM::mergeVec(
                                subProbs[i].F,
                                subStructures[idxSubStructure].F,
                                Idx1, Idx2);
                    else
                    {
                        subProbs[i].F.conservativeResize(subProbs[i].K.rows());
                        subProbs[i].F.tail(
                                         subStructures[idxSubStructure].dofIdx.rows() -
                                         Idx2.size())
                            .setZero();
                    }
                    subProbs[i].dofIdx = myFEM::mergeDofIdx(subProbs[i].dofIdx,
                                                            dofIdxTemp, Idx2);
                    // #pragma omp critical
                    //                     {
                    //                         std::cout << "线程" << rank << "拼接第" << subProbTypes[subProbTypesVec[i]] << "个子区域后各变量大小为：dofIdx "
                    //                                   << subProbs[i].dofIdx.rows() << "*" << subProbs[i].dofIdx.cols()
                    //                                   << ",K " << subProbs[i].K.rows() << "*" << subProbs[i].K.cols()
                    //                                   << ",F " << subProbs[i].F.size() << "\n";
                    //                     }
                }
            }
        }
        numSubStructures = nSubProbs;
        std::vector<baseStructure>().swap(subStructures);
        // std::cout << "元素个数: " << subStructures.size() << "\n";
        // 查看当前分配的内存能容纳多少个元素
        // std::cout << "容量: " << subStructures.capacity() << "\n";
        // 计算实际占用的堆内存字节数
        // std::cout << "实际占用堆内存: " << subStructures.capacity() * sizeof(baseStructure) << " 字节\n";
        // 重新生成geoArray
        geoArray.resize(1, geoArray.cols());
        Vec_d myori = this->ori;
        for (Idx col = 0; col < geoArray.cols(); ++col)
        {
            geoDom geoTemp;
            geoTemp.dom_p = subProbs[baseStructureArray(col)].dom_p;
            Vec_d dOri = myori - geoTemp.dom_p.col(0);
            geoTemp.dom_p.colwise() += dOri;
            if (subProbs[baseStructureArray(col)].dom_e.rows() > 0)
            {
                geoTemp.dom_e = subProbs[baseStructureArray(col)].dom_e;
                // geoTemp.dom_e.row(0).colwise() += dOri;
                geoTemp.dom_e.colwise() += dOri;
            }
            myori(0) += geoTemp.dom_p(0, 1) - geoTemp.dom_p(0, 0);
            geoArray(0, col) = geoTemp;
        }
        /*Test*/
        // std::cout << "形成带状子区域后，的各子区域x范围为：\n";
        // std::cout << "基底:\n";
        // for (Idx i = 0; i < geoArray.cols(); ++i)
        //     std::cout << geoArray(i).dom_p.row(0) << "\t";
        // std::cout << "电极:\n";
        // for (Idx i = 0; i < geoArray.cols(); ++i)
        // {
        //     if (geoArray(i).dom_e.rows() > 0)
        //         std::cout << geoArray(i).dom_e.row(0) << "\t";
        //     else
        //         std::cout << "NaN\t";
        // }
        // std::cout << "形成带状子区域后，的各子区域z范围为：\n";
        // std::cout << "基底:\n";
        // for (Idx i = 0; i < geoArray.cols(); ++i)
        //     std::cout << geoArray(i).dom_p.row(2) << "\t";
        // std::cout << "电极:\n";
        // for (Idx i = 0; i < geoArray.cols(); ++i)
        // {
        //     if (geoArray(i).dom_e.rows() > 0)
        //         std::cout << geoArray(i).dom_e.row(2) << "\t";
        //     else
        //         std::cout << "NaN\t";
        // }
        // for (Idx i = 0; i < subProbs.size(); ++i)
        //     std::cout << "第" << i << "个子结构的tag为" << subProbs[i].tag << "\n";
        // for (Idx i = 0; i < numSubStructures; ++i)
        // {
        //     std::string name1 = "dofIdxBand" + std::to_string(i) + ".mat";
        //     const char *c_name1 = name1.c_str();
        //     std::string name2 = "KBand" + std::to_string(i) + ".mat";
        //     const char *c_name2 = name2.c_str();
        //     wrMatFile::saveEigenDenseMat2Mat(c_name1, subProbs[i].dofIdx);
        //     wrMatFile::saveEigenSparseMat2Mat(c_name2, subProbs[i].K);
        // }
        // for (auto p = subProbTypes.begin(); p != subProbTypes.end(); ++p)
        // {
        //     std::cout << "第" << p->second << "个子区域的组成为: ";
        //     for (Idx i = 0; i < p->first.size(); ++i)
        //         std::cout << p->first[i] << "\t";
        //     std::cout << "\n";
        // }
    }
    void deviceArray::formGlobalSystem()
    {
        // 树状结构合并子问题
        // 先形成带状问题
        genBandedSubProbs();
        struct LoPr
        {
            Mat_d dofIdx;
            SparseMat_t<Complex> K;
            Vec_t<Complex> F;
            Idx searchEnd1, searchStart2; // searchEnd1在第一次赋值后就不用再改变了
            Mat_d dom;
        };
        std::vector<LoPr> loPr;
        Idx nJobs = ceil(baseStructureArray.size() / 2.0);
        // std::cout << baseStructureArray.size() << "个子区域被划分为了" << nJobs << "个局部结构\n";
        loPr.resize(nJobs);
#pragma omp parallel
        {
            /*树最底层数据生成*/
            Idx ompSize = omp_get_num_threads();
            auto jobList = myOMP::distributeTasks(ompSize, nJobs);
            Idx rank = omp_get_thread_num();
            Idx tStart = jobList[rank].first;
            Idx tEnd = tStart + jobList[rank].second;
            for (Idx i = tStart; i < tEnd; ++i)
            {
                Idx idxSt1 = 2 * i;
                Idx idxSt2 = idxSt1 + 1;
                loPr[i].K = subProbs[baseStructureArray(0, idxSt1)].K;
                loPr[i].F = subProbs[baseStructureArray(0, idxSt1)].F;
                loPr[i].dofIdx = subProbs[baseStructureArray(0, idxSt1)].dofIdx;
                loPr[i].dom = geoArray(0, idxSt1).dom_p;
                translate3dPoints(loPr[i].dofIdx,
                                  subProbs[baseStructureArray(0, idxSt1)].dom_p.col(0),
                                  geoArray(0, idxSt1).dom_p.col(0));
                loPr[i].searchEnd1 = loPr[i].dofIdx.rows();
                if (idxSt2 == baseStructureArray.size()) // 落单的子问题
                {
                    continue;
                }
                else if (idxSt2 < baseStructureArray.size())
                {
                    loPr[i].searchStart2 = loPr[i].dofIdx.rows();
                    Mat_d dofIdx_temp = subProbs[baseStructureArray(0, idxSt2)].dofIdx;
                    translate3dPoints(dofIdx_temp,
                                      subProbs[baseStructureArray(0, idxSt2)].dom_p.col(0),
                                      geoArray(0, idxSt2).dom_p.col(0));
                    double scale = (loPr[i].dom.col(1) - loPr[i].dom.col(0)).minCoeff();
                    // #pragma omp critical
                    //                     {
                    //                         std::cout << "nJobs=" << nJobs << ",线程" << rank << "拼接第"
                    //                                   << idxSt1 << "和" << idxSt2
                    //                                   << "子区域，左区域的右界面x1 = " << loPr[i].dofIdx.col(0).maxCoeff()
                    //                                   << "右区域的左界面x1 = " << dofIdx_temp.col(0).minCoeff()
                    //                                   << "\n";
                    //                     }
                    auto [Idx1, Idx2] = myFEM::findCommonDof(loPr[i].dofIdx,
                                                             dofIdx_temp,
                                                             {0, loPr[i].dom(0, 1)}, scale);
                    // #pragma omp critical
                    //                     {
                    //                         std::cout << "线程" << rank << "拼接第" << idxSt1 << "和"
                    //                                   << idxSt2 << "子区域，共有" << Idx2.size() << "个界面自由度\n";
                    //                     }
                    loPr[i].K = myFEM::mergeSparseMat<Complex>(loPr[i].K,
                                                               subProbs[baseStructureArray(0, idxSt2)].K,
                                                               Idx1, Idx2);
                    loPr[i].F = myFEM::mergeVec<Complex>(loPr[i].F,
                                                         subProbs[baseStructureArray(0, idxSt2)].F,
                                                         Idx1, Idx2);
                    loPr[i].dofIdx = myFEM::mergeDofIdx<double>(loPr[i].dofIdx,
                                                                dofIdx_temp,
                                                                Idx2);
                    loPr[i].dom(0, 1) = geoArray(0, idxSt2).dom_p(0, 1);
                }
                else
                {
                    std::cout << "formGlobalSystem:Wrong!!!！";
                    std::exit(EXIT_FAILURE);
                }
            }
#pragma omp barrier
#pragma omp single // 释放内存
            {
                std::vector<SubProb>().swap(subProbs);
                Mat_i().swap(baseStructureArray);
                Eigen::Matrix<geoDom, Eigen::Dynamic,
                              Eigen::Dynamic>()
                    .swap(geoArray);
                // std::cout << "元素个数: " << subProbs.size() << "\n";
                // // 查看当前分配的内存能容纳多少个元素
                // std::cout << "容量: " << subProbs.capacity() << "\n";
                // // 计算实际占用的堆内存字节数
                // std::cout << "实际占用堆内存: " << subProbs.capacity() * sizeof(SubProb) << " 字节\n";
                // std::cout << "共" << loPr.size() << "个局部结构\n";
                // for (Idx i = 0; i < loPr.size(); ++i)
                // {
                //     std::cout << "第一轮拼接结束后，第" << i << "个局部结构：\n"
                //               << "dom:" << loPr[i].dom << ";\n"
                //               << "dofIdx: " << loPr[i].dofIdx.rows() << "*" << loPr[i].dofIdx.cols() << ";\t"
                //               << "K:" << loPr[i].K.rows() << "*" << loPr[i].K.cols() << ";\t"
                //               << "F:" << loPr[i].F.size() << "\n";
                // }
            }
            // 开始树状合并
            while (nJobs > 1)
            {
#pragma omp barrier
#pragma omp single
                nJobs = ceil(loPr.size() / 2.0);
                jobList = myOMP::distributeTasks(ompSize, nJobs);
                tStart = jobList[rank].first;
                tEnd = tStart + jobList[rank].second;
                std::vector<LoPr> loPrTemp(jobList[rank].second); // 临时存储局部问题
                Idx loIdx = 0;
                for (Idx i = tStart; i < tEnd; ++i)
                {
                    Idx glIdxSt1 = 2 * i;
                    Idx glIdxEnd2 = glIdxSt1 + 1;
                    loPrTemp[loIdx] = std::move(loPr[glIdxSt1]);
                    // loPrTemp[loIdx] = loPr[glIdxSt1];
                    if (glIdxEnd2 < loPr.size())
                    {

                        double scale = (loPrTemp[loIdx].dom.col(1) -
                                        loPrTemp[loIdx].dom.col(0))
                                           .minCoeff();
                        // #pragma omp critical
                        //                         {
                        //                             std::cout << "nJobs=" << nJobs << ",线程" << rank << "拼接第"
                        //                                       << glIdxSt1 << "和" << glIdxEnd2
                        //                                       << "子区域，左区域的右界面x1 = " << loPrTemp[loIdx].dofIdx.col(0).maxCoeff()
                        //                                       << "右区域的左界面x1 = " << loPr[glIdxEnd2].dofIdx.col(0).minCoeff()
                        //                                       << "\n";
                        //                         }
                        auto [Idx1, Idx2] =
                            myFEM::findCommonDof(loPrTemp[loIdx].dofIdx,
                                                 loPr[glIdxEnd2].dofIdx,
                                                 {0, loPrTemp[loIdx].dom(0, 1)}, scale,
                                                 loPrTemp[loIdx].searchStart2, 0, -1,
                                                 loPr[glIdxEnd2].searchEnd1);
                        loPrTemp[loIdx].searchStart2 = loPrTemp[loIdx].dofIdx.rows();
                        // #pragma omp critical
                        //                         {
                        //                             std::cout << "nJobs=" << nJobs << ",线程" << rank << "拼接第"
                        //                                       << glIdxSt1 << "和" << glIdxEnd2
                        //                                       << "子区域，共有" << Idx2.size() << "个界面自由度\n";
                        //                         }
                        loPrTemp[loIdx].dofIdx =
                            myFEM::mergeDofIdx<double>(loPrTemp[loIdx].dofIdx,
                                                       loPr[glIdxEnd2].dofIdx, Idx2);
                        loPrTemp[loIdx].K = myFEM::mergeSparseMat<Complex>(loPrTemp[loIdx].K,
                                                                           loPr[glIdxEnd2].K,
                                                                           Idx1, Idx2);
                        loPrTemp[loIdx].F = myFEM::mergeVec<Complex>(loPrTemp[loIdx].F,
                                                                     loPr[glIdxEnd2].F,
                                                                     Idx1, Idx2);
                        loPrTemp[loIdx].dom(0, 1) = loPr[glIdxEnd2].dom(0, 1);
                    }
                    loIdx++;
                }
#pragma omp barrier
#pragma omp single
                {
                    loPr.resize(nJobs);
                    loPr.shrink_to_fit();
                }
                loIdx = 0;
                for (Idx i = tStart; i < tEnd; ++i)
                {
                    loPr[i] = std::move(loPrTemp[loIdx]);
                    ++loIdx;
                }
            }
        }
        // std::cout << "最终loPr的大小为：" << loPr.size() << "\n";
        K = std::move(loPr[0].K);
        F = std::move(loPr[0].F);
        dofIdx = std::move(loPr[0].dofIdx);
    }
    void deviceArray::basicResonator(                                     // 基本的谐振器，一对反射栅，一组叉指换能器，多项式PML
        Idx nIDT, Idx nRef, Idx nBar,                                     // 叉指换能器、反射栅和光板的个数
        typeMaterial materialSub, typeMaterial materialEle,               // 压电基底材料和电极材料
        double xIDT, double xRef, double xBar,                            // 叉指换能器、反射栅和光板的宽度（x方向的长度）
        double yDev, double zDev,                                         // 器件在y方向和z方向的长度
        double metCovRat, double zEle,                                    // 金属覆盖率、电极高度
        double omega, double V0,                                          // 工作频率、终端电压
        double xDamp, double zDamp, Idx nx, Idx nz, double dx, double dz, // PML的阻尼系数、多项式次数和深度
        Idx elemPtsPerWaveLen,
        myFEM::meshType mType, myFEM::elemType eType, // 网格类型和有限元类型
        Idx quaOrder,                                 // Gauss积分阶数
        double U0,
        bool ifNd) // 是否执行无量纲化
    {

        // 所有器件子结构的原点默认为(0,0,0)，PML区域则是按截断方向向外延伸
        if (nIDT <= 0)
        {
            std::cout << "deviceArray::basicResonator:IDT数量必须大于等于1!!!！";
            std::exit(EXIT_FAILURE);
        }
        numSubStructures = 11;
        subStructures.resize(11); // 0:IDTG; 1:IDTV; 2:Ref; 3:Bar;
                                  // 4:PMLL; 5:PMLR;
                                  // 6:PMLIDT; 7:PMLRef; 8:PMLBar
                                  // 9:PMLLB; 10:PMLRB
        baseStructureArray = genbaResArray(nIDT, nRef, nBar);
        std::cout << "器件子结构序列为：\n"
                  << baseStructureArray << "\n";
        double waveLen{2 * xIDT}; // 典型波长
        // 电极宽度
        double xEleIDT{metCovRat * xIDT}, xEleRef{metCovRat * xRef};
        /* 生成x,y,z三个方向的网格点数 */
        // Dev公共
        Idx NyDev{2}, NzDev{static_cast<Idx>(ceil(elemPtsPerWaveLen * zDev / waveLen))};
        // 电极
        Idx NxEleIDT{static_cast<Idx>(ceil(2 * xEleIDT / waveLen * elemPtsPerWaveLen))},
            NxEleRef{static_cast<Idx>(ceil(2 * xEleRef / waveLen * elemPtsPerWaveLen))},
            NzEle{6}; // 电极高度网格点数固定为6个
        // 基底部分x方向网格点数
        Idx NxIDT{static_cast<Idx>(ceil(2 * elemPtsPerWaveLen * xIDT / waveLen))},
            NxRef{static_cast<Idx>(ceil(2 * elemPtsPerWaveLen * xRef / waveLen))},
            NxBar{static_cast<Idx>(ceil(2 * elemPtsPerWaveLen * xBar / waveLen))};
        // std::cout << "NxBar=" << NxBar << ", NyBar=" << NyDev << ", NzBar=" << NzDev << "\n";
        // PML深度方向网格点数
        Idx NxPML{static_cast<Idx>(ceil(elemPtsPerWaveLen * dx / waveLen))},
            NzPML{static_cast<Idx>(ceil(elemPtsPerWaveLen * dz / waveLen))};
        /* 初始化除PML外的每个子结构（先不形成矩阵）*/
        // 形成各子结构的参考区域
        double oriEleIDTx{(xIDT - xEleIDT) / 2}, oriEleRefx{(xRef - xEleRef) / 2};
        // IDT
        Mat_d dom_p_IDT{{0, xIDT}, {0, yDev}, {0, zDev}},
            dom_e_IDT{{oriEleIDTx, oriEleIDTx + xEleIDT}, {0, yDev}, {zDev, zDev + zEle}};
        subStructures[0] = baseStructure(IDT, dom_p_IDT, dom_e_IDT, materialSub, materialEle, omega);
        subStructures[1] = baseStructure(IDT, dom_p_IDT, dom_e_IDT, materialSub, materialEle, omega);
        subStructures[1].setTerminalVoltage(V0);
        // TO DO
        if (nRef > 0) // Ref
        {
            Mat_d dom_p_Ref{{0, xRef}, {0, yDev}, {0, zDev}},
                dom_e_Ref{{oriEleRefx, oriEleRefx + xEleRef}, {0, yDev}, {zDev, zDev + zEle}};
            subStructures[2] = baseStructure(refGratOpen, dom_p_Ref, dom_e_Ref, materialSub, materialEle, omega);
        }
        if (nBar > 0) // Bar
        {
            Mat_d dom_p_Bar{{0, xBar}, {0, yDev}, {0, zDev}};
            subStructures[3] = baseStructure(piez, dom_p_Bar, materialSub, omega);
        }
        /*无量纲化*/
        if (ifNd)
        {
            // std::cout << "开始无量纲化...\n";
            this->genDimScales(U0);
            this->dimensionless();
            dx /= dimScales.L0;
            dz /= dimScales.L0;
        }
        /* 初始化PML*/
        // 拉伸函数
        auto alpha_x = [xDamp, nx, dx](const Mat_d &x) -> Vec_t<Complex>
        {
            Complex j(0.0, 1.0);
            Vec_t<Complex> temp = x.col(0).array().abs().pow(nx) / dx * xDamp * j;
            temp = 1.0 - temp.array();
            return temp;
        };
        auto alpha_z = [zDamp, nz, dz](const Mat_d &x) -> Vec_t<Complex>
        {
            Complex j(0.0, 1.0);
            Vec_t<Complex> temp = x.col(2).array().abs().pow(nz) / dz * zDamp * j;
            temp = 1.0 - temp.array();
            return temp;
        };
        // 初始化PML子结构
        initBaResPMLInfo(nRef, nBar, materialSub, dx, dz, alpha_x, alpha_z);
        /*形成矩阵并处理边界条件*/
        formBaResLoPro4SubStrs(mType, eType, quaOrder,
                               NyDev, NzDev,
                               NxIDT, NxRef, NxBar,
                               NxEleIDT, NxEleRef, NzEle,
                               NxPML, NzPML);
        // for (Idx i = 0; i < numSubStructures; ++i)
        // {
        //     switch (i)
        //     {
        //     case 0:
        //         wrMatFile::saveEigenDenseMat2Mat("dofIdxIDTG.mat", subStructures[i].dofIdx);
        //         wrMatFile::saveEigenSparseMat2Mat("KIDTG.mat", subStructures[i].K);
        //         break;
        //     case 1:
        //         wrMatFile::saveEigenDenseMat2Mat("dofIdxIDTV.mat", subStructures[i].dofIdx);
        //         wrMatFile::saveEigenSparseMat2Mat("KIDTV.mat", subStructures[i].K);
        //         wrMatFile::saveEigenVec2Mat("FIDTV.mat", subStructures[i].F);
        //         break;
        //     case 2:
        //         wrMatFile::saveEigenDenseMat2Mat("dofIdxRef.mat", subStructures[i].dofIdx);
        //         wrMatFile::saveEigenSparseMat2Mat("KRef.mat", subStructures[i].K);
        //         break;
        //     case 3:
        //         wrMatFile::saveEigenDenseMat2Mat("dofIdxBar.mat", subStructures[i].dofIdx);
        //         wrMatFile::saveEigenSparseMat2Mat("KBar.mat", subStructures[i].K);
        //         break;
        //     case 4:
        //         wrMatFile::saveEigenDenseMat2Mat("dofIdxPMLL.mat", subStructures[i].dofIdx);
        //         wrMatFile::saveEigenSparseMat2Mat("KPMLL.mat", subStructures[i].K);
        //         break;
        //     case 5:
        //         wrMatFile::saveEigenDenseMat2Mat("dofIdxPMLR.mat", subStructures[i].dofIdx);
        //         wrMatFile::saveEigenSparseMat2Mat("KPMLR.mat", subStructures[i].K);
        //         break;
        //     case 6:
        //         wrMatFile::saveEigenDenseMat2Mat("dofIdxPMLIDT.mat", subStructures[i].dofIdx);
        //         wrMatFile::saveEigenSparseMat2Mat("KPMLIDT.mat", subStructures[i].K);
        //         break;
        //     case 7:
        //         wrMatFile::saveEigenDenseMat2Mat("dofIdxPMLRef.mat", subStructures[i].dofIdx);
        //         wrMatFile::saveEigenSparseMat2Mat("KPMLRef.mat", subStructures[i].K);
        //         break;
        //     case 8:
        //         wrMatFile::saveEigenDenseMat2Mat("dofIdxPMLBar.mat", subStructures[i].dofIdx);
        //         wrMatFile::saveEigenSparseMat2Mat("KPMLBar.mat", subStructures[i].K);
        //         break;
        //     case 9:
        //         wrMatFile::saveEigenDenseMat2Mat("dofIdxPMLLB.mat", subStructures[i].dofIdx);
        //         wrMatFile::saveEigenSparseMat2Mat("KPMLLB.mat", subStructures[i].K);
        //         break;
        //     case 10:
        //         wrMatFile::saveEigenDenseMat2Mat("dofIdxPMLRB.mat", subStructures[i].dofIdx);
        //         wrMatFile::saveEigenSparseMat2Mat("KPMLRB.mat", subStructures[i].K);
        //         break;
        //     default:
        //         break;
        //     }
        // }
        formGeoArray(Eigen::Vector3d(-dx, 0, -dz));
        // for (auto p = subStructures.begin(); p != subStructures.end(); ++p)
        // {
        //     if (p->type == none)
        //     {
        //         std::cout << "第" << static_cast<size_t>(p - subStructures.begin()) << "号子结构为none\n";
        //     }
        //     else
        //     {
        //         std::cout << "第" << static_cast<size_t>(p - subStructures.begin()) << "号子结构为" << p->type
        //                   << ",规模为" << p->dofIdx.rows() << "\n";
        //     }
        // }
    }
    Mat_i deviceArray::genbaResArray(Idx nIDT, Idx nRef, Idx nBar)
    {
        Idx nCol = 2 + nIDT + 2 * nRef + 2 * nBar;
        Mat_i resArray(2, nCol);
        Idx pos = 0;
        resArray.col(0) << 4, 9;
        ++pos;
        for (Idx i = 0; i < nRef; ++i)
        {
            resArray.col(pos) << 2, 7;
            pos++;
        }
        for (Idx i = 0; i < nBar; ++i)
        {
            resArray.col(pos) << 3, 8;
            pos++;
        }
        for (Idx i = 0; i < nIDT; ++i)
        {
            if (i % 2 == 0) // V
                resArray.col(pos) << 1, 6;
            else // G
                resArray.col(pos) << 0, 6;
            ++pos;
        }
        for (Idx i = 0; i < nBar; ++i)
        {
            resArray.col(pos) << 3, 8;
            pos++;
        }
        for (Idx i = 0; i < nRef; ++i)
        {
            resArray.col(pos) << 2, 7;
            pos++;
        }
        resArray.col(pos) << 5, 10;
        return resArray;
    }
    // using Fun = std::function<Eigen::VectorXcd(const Eigen::MatrixX3d &)>;
    void deviceArray::initBaResPMLInfo(Idx nRef, Idx nBar, typeMaterial materialSub,
                                       double dx, double dz, Fun alpha_x, Fun alpha_z)
    {
        auto Id = [](const Mat_d &x) -> Vec_t<Complex>
        {
            return Vec_t<Complex>::Constant(x.rows(), {1.0, 0.0});
        };
        double omega = subStructures[0].para_p.w;
        double yDev = subStructures[0].dom_p(1, 1) - subStructures[0].dom_p(1, 0);
        double zDev = subStructures[0].dom_p(2, 1) - subStructures[0].dom_p(2, 0);
        double xIDT = subStructures[0].dom_p(0, 1) - subStructures[0].dom_p(0, 0);
        Mat_d domL{{-dx, 0}, {0, yDev}, {0, zDev}}, domR{{0, dx}, {0, yDev}, {0, zDev}},
            domBIDT{{0, xIDT}, {0, yDev}, {-dz, 0}}, domLB{{-dx, 0}, {0, yDev}, {-dz, 0}},
            domRB{{0, dx}, {0, yDev}, {-dz, 0}};
        // IDT的PML
        subStructures[4] = baseStructure(piezPML, domL, materialSub, alpha_x, Id, Id, left, omega);
        subStructures[4].para_p = subStructures[0].para_p; // 使用无量纲化后的参数。下同

        subStructures[5] = baseStructure(piezPML, domR, materialSub, alpha_x, Id, Id, right, omega);
        subStructures[5].para_p = subStructures[0].para_p;

        subStructures[6] = baseStructure(piezPML, domBIDT, materialSub, Id, Id, alpha_z, bottom, omega);
        subStructures[6].para_p = subStructures[0].para_p;

        subStructures[9] = baseStructure(piezPML, domLB, materialSub, alpha_x, Id, alpha_z, leftBottom, omega);
        subStructures[9].para_p = subStructures[0].para_p;

        subStructures[10] = baseStructure(piezPML, domRB, materialSub, alpha_x, Id, alpha_z, rightBottom, omega);
        subStructures[10].para_p = subStructures[0].para_p;
        // 反射栅
        if (nRef > 0)
        {
            double xRef = subStructures[2].dom_p(0, 1) - subStructures[2].dom_p(0, 0);
            Mat_d domBRef{{0, xRef}, {0, yDev}, {-dz, 0}};
            subStructures[7] = baseStructure(piezPML, domBRef, materialSub, Id, Id, alpha_z, bottom, omega);
            subStructures[7].para_p = subStructures[2].para_p;
        }
        if (nBar > 0)
        {
            double xBar = subStructures[3].dom_p(0, 1) - subStructures[3].dom_p(0, 0);
            Mat_d domBBar{{0, xBar}, {0, yDev}, {-dz, 0}};
            subStructures[8] = baseStructure(piezPML, domBBar, materialSub, Id, Id, alpha_z, bottom, omega);
            subStructures[8].para_p = subStructures[3].para_p;
        }
    }
    void deviceArray::formBaResLoPro4SubStrs(myFEM::meshType mType, myFEM::elemType eType, Idx quaOrder,
                                             Idx NyDev, Idx NzDev, Idx NxIDT, Idx NxRef, Idx NxBar,
                                             Idx NxEleIDT, Idx NxEleRef, Idx NzEle,
                                             Idx NxPML, Idx NzPML)
    {
        // 参考单元信息
        // refGaussInfo Gauss = genRefGauss(quaOrder, mType, eType);
        // 网格信息
        // 器件
        Vec_d yGridDev = Vec_d::LinSpaced(NyDev, subStructures[0].dom_p(1, 0), subStructures[0].dom_p(1, 1));
        Vec_d zGridDev = Vec_d::LinSpaced(NzDev, subStructures[0].dom_p(2, 0), subStructures[0].dom_p(2, 1));
        // IDT
        auto meshStepIDT = genMeshStepOfIDT(subStructures[0].dom_p, subStructures[0].dom_e,
                                            NxIDT, NyDev, NzDev, NxEleIDT, NzEle);
        // Ref
        meshStep meshStepRef;
        if (subStructures[2].type == refGratOpen)
            meshStepRef = genMeshStepOfIDT(subStructures[2].dom_p, subStructures[2].dom_e,
                                           NxRef, NyDev, NzDev, NxEleRef, NzEle);
        // Bar
        Vec_d xGridBar;
        if (subStructures[3].type == piez)
            xGridBar = Vec_d::LinSpaced(NxBar, subStructures[3].dom_p(0, 0), subStructures[3].dom_p(0, 1));
        // std::cout << "BarDom=" << subStructures[3].dom_p << "\n";

#pragma omp parallel for schedule(dynamic)
        for (Idx i = 0; i < numSubStructures; ++i)
        {
            refGaussInfo Gauss = genRefGauss(quaOrder, mType, eType);
            auto thisType = subStructures[i].type;
            if (thisType == none)
                continue;
            else if (thisType == piezPML)
            {
                Vec_d xGrid;
                Vec_d zGrid;
                if (i == 4 || i == 5) //  左右PML
                {
                    xGrid = Vec_d::LinSpaced(NxPML, subStructures[i].dom_p(0, 0), subStructures[i].dom_p(0, 1));
                    zGrid = meshStepIDT.zGrid_p;
                }
                else if (i == 9 || i == 10) // 左下、右下PML
                {
                    xGrid = Vec_d::LinSpaced(NxPML, subStructures[i].dom_p(0, 0), subStructures[i].dom_p(0, 1));
                    zGrid = Vec_d::LinSpaced(NzPML, subStructures[i].dom_p(2, 0), subStructures[i].dom_p(2, 1));
                }
                else if (i == 6) // IDT下部PML
                {
                    xGrid = meshStepIDT.xGrid_p;
                    zGrid = Vec_d::LinSpaced(NzPML, subStructures[i].dom_p(2, 0), subStructures[i].dom_p(2, 1));
                }
                else if (i == 7) // Ref下部PML
                {
                    xGrid = meshStepRef.xGrid_p;
                    zGrid = Vec_d::LinSpaced(NzPML, subStructures[i].dom_p(2, 0), subStructures[i].dom_p(2, 1));
                }
                else if (i == 8) // Bar下部PML
                {
                    xGrid = xGridBar;
                    zGrid = Vec_d::LinSpaced(NzPML, subStructures[i].dom_p(2, 0), subStructures[i].dom_p(2, 1));
                }
                mesh meshPML = genFEMMesh3D(subStructures[i].dom_p, xGrid, yGridDev, zGrid, eType, false);
                genAffineInfo(meshPML, false);
                affineGauss2AllElems(Gauss, meshPML, false);
                subStructures[i].initLinearSystem(Gauss, meshPML, false);
                subStructures[i].myTreatPeriodBoundCond();
                subStructures[i].myTreatPMLDirBoundCond();
            }
            else if (thisType == IDT)
            {
                mesh meshSub = genFEMMesh3D(subStructures[i].dom_p, meshStepIDT.xGrid_p, meshStepIDT.yGrid, meshStepIDT.zGrid_p, eType, false);
                mesh meshEle = genFEMMesh3D(subStructures[i].dom_e, meshStepIDT.xGrid_e, meshStepIDT.yGrid, meshStepIDT.zGrid_e, eType, false);
                genAffineInfo(meshSub, false);
                genAffineInfo(meshEle, false);
                subStructures[i].initLinearSystem(Gauss, meshSub, meshEle, false);
                subStructures[i].myTreatPeriodBoundCond();
                subStructures[i].myTreatTerminalBoundCond(false);
            }
            else if (thisType == refGratOpen)
            {
                mesh meshSub = genFEMMesh3D(subStructures[i].dom_p, meshStepRef.xGrid_p, meshStepRef.yGrid, meshStepRef.zGrid_p, eType, false);
                mesh meshEle = genFEMMesh3D(subStructures[i].dom_e, meshStepRef.xGrid_e, meshStepRef.yGrid, meshStepRef.zGrid_e, eType, false);
                genAffineInfo(meshSub, false);
                genAffineInfo(meshEle, false);
                subStructures[i].initLinearSystem(Gauss, meshSub, meshEle, false);
                subStructures[i].myTreatPeriodBoundCond();
                subStructures[i].myTreatFloatPotentialCond();
            }
            else if (thisType == piez)
            {
                mesh meshBar = genFEMMesh3D(subStructures[i].dom_p, xGridBar, yGridDev, zGridDev, eType, false);
                genAffineInfo(meshBar, false);
                subStructures[i].initLinearSystem(Gauss, meshBar, false);
                subStructures[i].myTreatPeriodBoundCond();
            }
        }
    }
    void deviceArray::formGeoArray(Eigen::Vector3d myori)
    {
        geoArray.resize(baseStructureArray.rows(), baseStructureArray.cols());
        double xStart = myori(0);
        for (Idx row = baseStructureArray.rows() - 1; row >= 0; --row)
        {
            double zWide;
            for (Idx col = 0; col < baseStructureArray.cols(); ++col)
            {
                geoDom geoTemp;
                Mat_d dom_p = subStructures[baseStructureArray(row, col)].dom_p;

                double xWide = dom_p(0, 1) - dom_p(0, 0);
                zWide = dom_p(2, 1) - dom_p(2, 0);
                geoTemp.dom_p = Mat_d{{myori(0), myori(0) + xWide},
                                      {myori(1), myori(1) + (dom_p(1, 1) - dom_p(1, 0))},
                                      {myori(2), myori(2) + zWide}};
                if (subStructures[baseStructureArray(row, col)].type == IDT ||
                    subStructures[baseStructureArray(row, col)].type == refGratOpen ||
                    subStructures[baseStructureArray(row, col)].type == refGratShort)
                {
                    Mat_d dom_e = subStructures[baseStructureArray(row, col)].dom_e;
                    geoTemp.dom_e = dom_e;
                    Eigen::Vector3d trans = myori - dom_p.col(0);
                    geoTemp.dom_e.colwise() += trans;
                }
                geoArray(row, col) = geoTemp;
                myori(0) += xWide;
            }
            myori(2) += zWide;
            myori(0) = xStart;
        }
        this->ori = geoArray(baseStructureArray.rows() - 1, 0).dom_p.col(0); // 全局器件原点值
        // std::cout << "器件原点为：" << this->ori << "\n";
    }
}
// std::cout << "无量纲化前:\n";
//         for (Idx i = 0; i < subStructures.size(); ++i)
//         {
//             typeBaseStructure thisType = subStructures[i].type;
//             if (thisType == none)
//                 continue;
//             else
//             {
//                 std::cout << "子结构" << i << ": dom_p:" << subStructures[i].dom_p << "\t"
//                           << "c_max:" << subStructures[i].para_p.c.maxCoeff() << "\t"
//                           << "e_max:" << subStructures[i].para_p.e.maxCoeff() << "\t"
//                           << "epcl_max" << subStructures[i].para_p.epcl.maxCoeff() << "\t"
//                           << "omega" << subStructures[i].para_p.w << "\t";
//                 if (thisType == IDT)
//                 {
//                     std::cout << "V0: " << subStructures[i].V0 << "\t"
//                               << "dom_e:" << subStructures[i].dom_e;
//                 }
//                 std::cout << "\n";
//             }
//         }

// std::cout << "无量纲化后电极:\n";
//         for (Idx i = 0; i < subStructures.size(); ++i)
//         {
//             typeBaseStructure thisType = subStructures[i].type;
//             if (thisType == none)
//                 continue;
//             else if (thisType == IDT || thisType == refGratOpen)
//             {
//                 std::cout << "子结构" << i << ": dom_e:" << subStructures[i].dom_e << "\t"
//                           << "lambda:" << subStructures[i].para_e.lambda << "\t"
//                           << "mu:" << subStructures[i].para_e.mu << "\t"
//                           << "rho_p" << subStructures[i].para_p.rho << "\t"
//                           << "rho_e" << subStructures[i].para_e.rho << "\t";
//                 std::cout << "\n";
//             }
//         }