#include "myfem.hpp"
#include <omp.h>
#include <Eigen/Dense>
#include <map>
namespace myFEM
{

    mesh genMesh3D(Mat_d dom, const Vec_d &xGrid,
                   const Vec_d &yGrid, const Vec_d &zGrid,
                   meshType type)
    {
        mesh myMesh;
        myMesh.Dom = dom;
        myMesh.meshtype = type;
        size_t nx = xGrid.size();
        size_t ny = yGrid.size();
        size_t nz = zGrid.size();
        myMesh.nXnYnZ << nx, ny, nz;
        switch (type)
        {
        case tet:
            std::cout << "undefined!\n";
            break;
        case hex:
        {
            Idx nElems = (nx - 1) * (ny - 1) * (nz - 1);
            Idx nNodes = nx * ny * nz;
            Idx nElems_xy = (nx - 1) * (ny - 1); // 每层的单元数
            Idx nNode_xy = nx * ny;
            myMesh.elems.resize(nElems, 8);
            myMesh.nElems = nElems;
            myMesh.nPts = nNodes;
            myMesh.nodes.resize(nNodes, 3);
#pragma omp parallel
            {
#pragma omp for nowait
                for (int n = 0; n < nElems; ++n)
                {
                    // 获取单元的三维编号
                    Idx k = n / nElems_xy;                    // 单元的z编号
                    Idx j = (n % nElems_xy) / (nx - 1);       // 单元的y编号
                    Idx i = n - k * nElems_xy - j * (nx - 1); // 单元的x编号
                    // 计算所有顶点的三维编号
                    Mat_i idx3d_P(3, 8);
                    idx3d_P << i, i + 1, i + 1, i, i, i + 1, i + 1, i,
                        j, j, j + 1, j + 1, j, j, j + 1, j + 1,
                        k, k, k, k, k + 1, k + 1, k + 1, k + 1;
                    // 计算出顶点
                    myMesh.elems.row(n) = idx3d_P.row(0) + idx3d_P.row(1) * nx + idx3d_P.row(2) * nNode_xy;
                }
#pragma omp for
                for (int n = 0; n < nNodes; ++n)
                {
                    Idx k = n / nNode_xy;              // 节点的z编号
                    Idx j = (n % nNode_xy) / nx;       // 节点的y编号
                    Idx i = n - k * nNode_xy - j * nx; // 节点的x编号
                    myMesh.nodes.row(n) << xGrid(i), yGrid(j), zGrid(k);
                }
            }
            break;
        }
        default:
            std::cout << "undefined!\n";
            break;
        }

        return myMesh;
    }
    mesh genFEMMesh3D(Mat_d dom, const Vec_d &xGrid,
                      const Vec_d &yGrid, const Vec_d &zGrid,
                      elemType type)
    {
        mesh myMesh;
        myMesh.elemtype = type;
        switch (type)
        {
        case Q1:
        {
            break;
        }
        case Q2:
        {
            myMesh = genMesh3D(dom, xGrid, yGrid, zGrid, hex);
            addEdgeMidPt(myMesh, xGrid, yGrid, zGrid);
            addFaceMidPt(myMesh, xGrid, yGrid, zGrid);
            addBodyMidPt(myMesh, xGrid, yGrid, zGrid);
            break;
        }
        }
        return myMesh;
    }
    void addEdgeMidPt(mesh &myMesh, const Vec_d &xGrid,
                      const Vec_d &yGrid, const Vec_d &zGrid)
    {
        const Vec_i *nPt = &myMesh.nXnYnZ;
        Idx nVertex = myMesh.nodes.rows(); // 网格顶点数

        Idx nMidPtx_x = (*nPt)(0) - 1;
        Idx nMidPtx_y = (*nPt)(0);
        Idx nMidPtx_z = (*nPt)(0);

        Idx nMidPtxy_x = nMidPtx_x * (*nPt)(1);
        Idx nMidPtxy_y = ((*nPt)(1) - 1) * nMidPtx_y;
        Idx nMidPtxy_z = (*nPt)(1) * nMidPtx_z;

        Idx nMidPtx = nMidPtxy_x * (*nPt)(2);
        Idx nMidPty = nMidPtxy_y * (*nPt)(2);
        Idx nMidPtz = ((*nPt)(2) - 1) * nMidPtxy_z;

        Idx nMidPt = nMidPtx + nMidPty + nMidPtz; // 边中点总数
        myMesh.nPts += nMidPt;
        Idx startX = nVertex;
        Idx startY = startX + nMidPtx;
        Idx startZ = startY + nMidPty;

        const Mat_d *myNode = &myMesh.nodes;
        const Mat_i *myElem = &myMesh.elems;

        Mat_d nodes_x(nMidPtx, 3), nodes_y(nMidPty, 3), nodes_z(nMidPtz, 3); // 边中点坐标
        Mat_i elems(myElem->rows(), 12);                                     // 每个单元边中点的编号

        Idx nElems_xy = ((*nPt)(0) - 1) * ((*nPt)(1) - 1);
        Idx nElems_x = (*nPt)(0) - 1;

#pragma omp parallel
        {

#pragma omp for collapse(2) nowait
            for (int n = 0; n < myElem->rows(); ++n)
            {
                for (int midType = 0; midType < 3; ++midType)
                {
                    Idx k = n / nElems_xy;                    // 单元的z编号
                    Idx j = (n % nElems_xy) / nElems_x;       // 单元的y编号
                    Idx i = n - k * nElems_xy - j * nElems_x; // 单元的x编号
                    switch (midType)
                    {
                    case 0:
                    {
                        Mat_i idx3d_Px(3, 4);
                        idx3d_Px
                            << i,
                            i, i, i, // 与x轴平行的边上的中点
                            j, j + 1, j, j + 1,
                            k, k, k + 1, k + 1;
                        elems.row(n).segment(0, 4) = idx3d_Px.row(0) +
                                                     idx3d_Px.row(1) * nMidPtx_x +
                                                     idx3d_Px.row(2) * nMidPtxy_x;
                        elems.row(n).segment(0, 4).array() += startX;
                        break;
                    }
                    case 1:
                    {
                        Mat_i idx3d_Py(3, 4);
                        idx3d_Py
                            << i,
                            i + 1, i, i + 1, // 与y轴平行的边上的中点
                            j, j, j, j,
                            k, k, k + 1, k + 1;
                        elems.row(n).segment(4, 4) = idx3d_Py.row(0) +
                                                     idx3d_Py.row(1) * nMidPtx_y +
                                                     idx3d_Py.row(2) * nMidPtxy_y;
                        elems.row(n).segment(4, 4).array() += startY;
                        break;
                    }
                    case 2:
                    {
                        Mat_i idx3d_Pz(3, 4);
                        idx3d_Pz
                            << i,
                            i + 1, i, i + 1, // 与z轴平行的边上的中点
                            j, j, j + 1, j + 1,
                            k, k, k, k;
                        elems.row(n).segment(8, 4) = idx3d_Pz.row(0) +
                                                     idx3d_Pz.row(1) * nMidPtx_z +
                                                     idx3d_Pz.row(2) * nMidPtxy_z;
                        elems.row(n).segment(8, 4).array() += startZ;
                        break;
                    }
                    }
                }
            }
#pragma omp for nowait // 填充与x轴平行边上的中点坐标
            for (int n = 0; n < nMidPtx; ++n)
            {
                Idx k = n / nMidPtxy_x;                     // 中点的z编号
                Idx j = (n % nMidPtxy_x) / nMidPtx_x;       // 中点的y编号
                Idx i = n - k * nMidPtxy_x - j * nMidPtx_x; // 中点的x编号
                nodes_x.row(n) << (xGrid(i) + xGrid(i + 1)) * 0.5, yGrid(j), zGrid(k);
            }
#pragma omp for nowait // 填充与y轴平行边上的中点坐标
            for (int n = 0; n < nMidPty; ++n)
            {
                Idx k = n / nMidPtxy_y;                     // 中点的z编号
                Idx j = (n % nMidPtxy_y) / nMidPtx_y;       // 中点的y编号
                Idx i = n - k * nMidPtxy_y - j * nMidPtx_y; // 中点的x编号
                nodes_y.row(n) << xGrid(i), (yGrid(j) + yGrid(j + 1)) * 0.5, zGrid(k);
            }
#pragma omp for // 填充与z轴平行边上的中点坐标
            for (int n = 0; n < nMidPtz; ++n)
            {
                Idx k = n / nMidPtxy_z;                     // 中点的z编号
                Idx j = (n % nMidPtxy_z) / nMidPtx_z;       // 中点的y编号
                Idx i = n - k * nMidPtxy_z - j * nMidPtx_z; // 中点的x编号
                nodes_z.row(n) << xGrid(i), yGrid(j), (zGrid(k) + zGrid(k + 1)) * 0.5;
            }
        }
        Mat_i elemTemp(myElem->rows(), myElem->cols() + 12);
        elemTemp << *myElem, elems;
        myMesh.elems = std::move(elemTemp);
        Mat_d nodeTemp(nVertex + nMidPt, 3);
        nodeTemp << *myNode, nodes_x, nodes_y, nodes_z;
        myMesh.nodes = std::move(nodeTemp);
    }

    void addFaceMidPt(mesh &myMesh, const Vec_d &xGrid,
                      const Vec_d &yGrid, const Vec_d &zGrid)
    {
        const Vec_i *nPt = &myMesh.nXnYnZ;
        Idx nVertex = myMesh.nodes.rows(); // 网格顶点数

        Idx nMidPtx_x = (*nPt)(0) - 1; // xz
        Idx nMidPtx_y = (*nPt)(0);     // yz
        Idx nMidPtx_z = (*nPt)(0) - 1; // xy

        Idx nMidPtxy_x = nMidPtx_x * (*nPt)(1);
        Idx nMidPtxy_y = ((*nPt)(1) - 1) * nMidPtx_y;
        Idx nMidPtxy_z = ((*nPt)(1) - 1) * nMidPtx_z;

        Idx nMidPtx = nMidPtxy_x * ((*nPt)(2) - 1);
        Idx nMidPty = nMidPtxy_y * ((*nPt)(2) - 1);
        Idx nMidPtz = (*nPt)(2) * nMidPtxy_z;

        Idx nMidPt = nMidPtx + nMidPty + nMidPtz; // 面中点总数
        myMesh.nPts += nMidPt;
        Idx startX = nVertex;
        Idx startY = startX + nMidPtx;
        Idx startZ = startY + nMidPty;

        const Mat_d *myNode = &myMesh.nodes;
        const Mat_i *myElem = &myMesh.elems;

        Mat_d nodes_x(nMidPtx, 3), nodes_y(nMidPty, 3), nodes_z(nMidPtz, 3); // 边中点坐标
        Mat_i elems(myElem->rows(), 6);                                      // 每个单元边中点的编号

        Idx nElems_xy = ((*nPt)(0) - 1) * ((*nPt)(1) - 1);
        Idx nElems_x = (*nPt)(0) - 1;

#pragma omp parallel
        {

#pragma omp for collapse(2) nowait
            for (int n = 0; n < myElem->rows(); ++n)
            {
                for (int midType = 0; midType < 3; ++midType)
                {
                    Idx k = n / nElems_xy;                    // 单元的z编号
                    Idx j = (n % nElems_xy) / nElems_x;       // 单元的y编号
                    Idx i = n - k * nElems_xy - j * nElems_x; // 单元的x编号
                    switch (midType)
                    {
                    case 0: // 与xz面平行的面的中点
                    {
                        Mat_i idx3d_Px(3, 2);
                        idx3d_Px
                            << i,
                            i,
                            j, j + 1,
                            k, k;
                        elems.row(n).segment(0, 2) = idx3d_Px.row(0) +
                                                     idx3d_Px.row(1) * nMidPtx_x +
                                                     idx3d_Px.row(2) * nMidPtxy_x;
                        elems.row(n).segment(0, 2).array() += startX;
                        break;
                    }
                    case 1: // 与yz面平行的面的中点
                    {
                        Mat_i idx3d_Py(3, 2);
                        idx3d_Py
                            << i,
                            i + 1,
                            j, j,
                            k, k;
                        elems.row(n).segment(2, 2) = idx3d_Py.row(0) +
                                                     idx3d_Py.row(1) * nMidPtx_y +
                                                     idx3d_Py.row(2) * nMidPtxy_y;
                        elems.row(n).segment(2, 2).array() += startY;
                        break;
                    }
                    case 2: // 与xy面平行的面的中点
                    {
                        Mat_i idx3d_Pz(3, 2);
                        idx3d_Pz
                            << i,
                            i,
                            j, j,
                            k, k + 1;
                        elems.row(n).segment(4, 2) = idx3d_Pz.row(0) +
                                                     idx3d_Pz.row(1) * nMidPtx_z +
                                                     idx3d_Pz.row(2) * nMidPtxy_z;
                        elems.row(n).segment(4, 2).array() += startZ;
                        break;
                    }
                    }
                }
            }
#pragma omp for nowait // 填充与x轴平行边上的中点坐标
            for (int n = 0; n < nMidPtx; ++n)
            {
                Idx k = n / nMidPtxy_x;                     // 中点的z编号
                Idx j = (n % nMidPtxy_x) / nMidPtx_x;       // 中点的y编号
                Idx i = n - k * nMidPtxy_x - j * nMidPtx_x; // 中点的x编号
                nodes_x.row(n) << (xGrid(i) + xGrid(i + 1)) * 0.5, yGrid(j), (zGrid(k) + zGrid(k + 1)) * 0.5;
            }
#pragma omp for nowait // 填充与y轴平行边上的中点坐标
            for (int n = 0; n < nMidPty; ++n)
            {
                Idx k = n / nMidPtxy_y;                     // 中点的z编号
                Idx j = (n % nMidPtxy_y) / nMidPtx_y;       // 中点的y编号
                Idx i = n - k * nMidPtxy_y - j * nMidPtx_y; // 中点的x编号
                nodes_y.row(n) << xGrid(i), (yGrid(j) + yGrid(j + 1)) * 0.5, (zGrid(k) + zGrid(k + 1)) * 0.5;
            }
#pragma omp for // 填充与z轴平行边上的中点坐标
            for (int n = 0; n < nMidPtz; ++n)
            {
                Idx k = n / nMidPtxy_z;                     // 中点的z编号
                Idx j = (n % nMidPtxy_z) / nMidPtx_z;       // 中点的y编号
                Idx i = n - k * nMidPtxy_z - j * nMidPtx_z; // 中点的x编号
                nodes_z.row(n) << (xGrid(i) + xGrid(i + 1)) * 0.5, (yGrid(j) + yGrid(j + 1)) * 0.5, zGrid(k);
            }
        }
        Mat_i elemTemp(myElem->rows(), myElem->cols() + 6);
        elemTemp << *myElem, elems;
        myMesh.elems = std::move(elemTemp);
        Mat_d nodeTemp(nVertex + nMidPt, 3);
        nodeTemp << *myNode, nodes_x, nodes_y, nodes_z;
        myMesh.nodes = std::move(nodeTemp);
    }
    void addBodyMidPt(mesh &myMesh, const Vec_d &xGrid,
                      const Vec_d &yGrid, const Vec_d &zGrid)
    {
        const Vec_i *nPt = &myMesh.nXnYnZ;
        Idx nVertex = myMesh.nodes.rows(); // 网格顶点数
        Idx nMidPt = myMesh.elems.rows();
        Idx start = nVertex;

        Mat_i elems(myMesh.elems.rows(), 1);
        Mat_d nodes(nMidPt, 3);
        myMesh.nPts += nMidPt;
        Idx nElems_xy = ((*nPt)(0) - 1) * ((*nPt)(1) - 1);
        Idx nElems_x = (*nPt)(0) - 1;
#pragma omp parallel for
        for (int n = 0; n < myMesh.elems.rows(); ++n)
        {
            Idx k = n / nElems_xy;                    // 单元的z编号
            Idx j = (n % nElems_xy) / nElems_x;       // 单元的y编号
            Idx i = n - k * nElems_xy - j * nElems_x; // 单元的x编号
            elems(n) = n + start;
            nodes.row(n) << (xGrid(i) + xGrid(i + 1)) * 0.5,
                (yGrid(j) + yGrid(j + 1)) * 0.5,
                (zGrid(k) + zGrid(k + 1)) * 0.5;
        }
        Mat_i elemTemp(myMesh.elems.rows(), myMesh.elems.cols() + 1);
        elemTemp << myMesh.elems, elems;
        myMesh.elems = std::move(elemTemp);
        Mat_d nodeTemp(nVertex + nMidPt, 3);
        nodeTemp << myMesh.nodes, nodes;
        myMesh.nodes = std::move(nodeTemp);
    }
    double baseFunRef3D(Eigen::Vector3d pt, Idx idxFun, Vec_i diff, elemType type)
    { // pt:要计算的点，idxFun:基函数编号，diff：导数阶，type：有限元类型
        Mat_i cube;
        // Idx nrows = pt.at(0).size();
        // Idx ncols = pt.at(0).cols();
        double result = 1;
        switch (type)
        {
        case Q1:
        {
            cube.resize(8, 3);
            cube << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1;
            break;
        }

        case Q2: // 0->-1(左端点),1->0(中点), 2->1(右端点)
        {
            cube.resize(27, 3);
            cube << 0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0, 2,
                2, 0, 2, 2, 2, 2, 0, 2, 2, 1, 0, 0, 1, 2, 0,
                1, 0, 2, 1, 2, 2, 0, 1, 0, 2, 1, 0, 0, 1, 2,
                2, 1, 2, 0, 0, 1, 2, 0, 1, 0, 2, 1, 2, 2, 1,
                1, 0, 1, 1, 2, 1, 0, 1, 1, 2, 1, 1, 1, 1, 0,
                1, 1, 2, 1, 1, 1;

            for (Idx k = 0; k < 3; ++k)
            {
                double z = pt(k);
                double result_temp;
                if (diff(k) == 0)
                {
                    switch (cube(idxFun, k))
                    {
                    case 0:
                    {
                        result_temp = z * (z - 1) * 0.5;
                        break;
                    }
                    case 1:
                    {
                        result_temp = 1 - z * z;
                        break;
                    }
                    case 2:
                    {
                        result_temp = z * (z + 1) * 0.5;
                        break;
                    }
                    }
                }
                else if (diff(k) == 1)
                {
                    switch (cube(idxFun, k))
                    {
                    case 0:
                    {
                        result_temp = z - 0.5;
                        break;
                    }
                    case 1:
                    {
                        result_temp = -2 * z;
                        break;
                    }
                    case 2:
                    {
                        result_temp = z + 0.5;
                        break;
                    }
                    }
                }
                else
                {
                    std::cerr << "错误：高阶偏导数未定义" << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                result *= result_temp;
            }
        }
        }
        return result;
    }
    double baseFun3D(Eigen::Vector3d pt, Idx idxFun, Vec_i diff, elemType type, Mat_d elem)
    {
        Eigen::Vector3d v_minus = elem.col(1) - elem.col(0);
        Eigen::Vector3d v_plus = elem.col(1) + elem.col(0);

        for (int i = 0; i < 3; ++i)
        {
            double a = 2 / v_minus(i);
            double b = -v_plus(i) / v_minus(i);
            pt[i] *= a;
            pt[i] = pt[i] + b;
        }
        double result = baseFunRef3D(pt, idxFun, diff, type);
        for (int i = 0; i < 3; ++i)
        {
            if (diff(i) == 1)
            {
                double temp = 2 / v_minus(i);
                result *= temp;
            }
        }
        return result;
    }
    void genAffineInfo(mesh &myMesh)
    {
        Idx nElems = myMesh.elems.rows();
        switch (myMesh.meshtype)
        {
        case hex:
        {
            Mat_d a(nElems, 3);
#pragma omp parallel for
            for (int n = 0; n < nElems; ++n)
            {
                Vec_d P1 = myMesh.nodes.row(myMesh.elems(n, 0));
                Vec_d P2 = myMesh.nodes.row(myMesh.elems(n, 6));
                a.row(n) = (P2 - P1) * 0.5;
            }
            myMesh.Jacobi = a.array().rowwise().prod();
            myMesh.Jacobi_inv = 1.0 / a.array();
            break;
        }
        case tet:
        {
            break;
        }
        }
    }
    refGaussInfo genRefGauss(Idx order, meshType meshtype, elemType elemtype)
    {
        refGaussInfo Gauss;
        Gauss.elemtype = elemtype;
        Gauss.meshtype = meshtype;
        Vec_d x, h;
        switch (order)
        {
        case 2:
        {
            x.resize(2);
            h.resize(2);
            h << 1, 1;
            x << 1 / sqrt(3), -1 / sqrt(3);
            break;
        }
        case 4:
        {
            x.resize(4);
            h.resize(4);
            h << 0.347854845137454, 0.347854845137454, 0.652145154862546, 0.652145154862546;
            x << 0.861136311594053, -0.861136311594053, 0.339981043584856, -0.339981043584856;
            break;
        }
        case 8:
        {
            x.resize(8);
            h.resize(8);
            h << 0.1012285363, 0.1012285363, 0.2223810345, 0.2223810345, 0.3137066459, 0.3137066459, 0.3626837834, 0.3626837834;
            x << 0.9602898565, -0.9602898565, 0.7966664774, -0.7966664774, 0.5255324099, -0.5255324099, 0.1834346425, -0.1834346425;
            break;
        }
        default:
        {
            std::cout << "未定义的Gauss点类型\n";
            std::exit(EXIT_FAILURE);
            break;
        }
        }
        // 生成三维Gauss点和权重
        Idx nPt_xy = order * order;
        Idx nPt = nPt_xy * order;

        Gauss.pt.resize(nPt, 3);
        Gauss.weight.resize(nPt);
        Idx nBaseFun; // 参考单元中基函数个数
        switch (elemtype)
        {
        case Q1:
            nBaseFun = 8;
            break;
        case Q2:
            nBaseFun = 27;
            break;
        default:
            std::cout << "未定义基函数类型\n";
            std::exit(EXIT_FAILURE);
            break;
        }

        Gauss.phi.resize(4);
        for (int i = 0; i < 4; ++i)
            Gauss.phi[i].resize(nBaseFun, nPt);
#pragma omp parallel
        {
#pragma omp for collapse(3)
            for (Idx i = 0; i < order; ++i)
            {
                for (Idx j = 0; j < order; ++j)
                {
                    for (Idx k = 0; k < order; ++k)
                    {
                        Idx idxPt = i + j * order + k * nPt_xy;
                        Gauss.pt.row(idxPt) << x(i), x(j), x(k);
                        Gauss.weight(idxPt) = h(i) * h(j) * h(k);
                    }
                }
            }
#pragma omp for collapse(3)
            for (Idx i = 0; i < 4; ++i) // 导数
            {
                for (Idx j = 0; j < nBaseFun; ++j) // 遍历基函数
                {
                    for (Idx k = 0; k < nPt; ++k) // 遍历Gauss积分点
                    {
                        Vec_i diff(3);
                        switch (i)
                        {
                        case 0:
                            diff << 0, 0, 0;
                            break;
                        case 1:
                            diff << 1, 0, 0;
                            break;
                        case 2:
                            diff << 0, 1, 0;
                            break;
                        case 3:
                            diff << 0, 0, 1;
                            break;
                        }
                        Gauss.phi[i](j, k) = baseFunRef3D(Gauss.pt.row(k), j, diff, elemtype);
                    }
                }
            }
        }
        return Gauss;
    }
    void affineGauss2AllElems(const refGaussInfo &GaussInfo, mesh &myMesh)
    {
        myMesh.GaussPt.resize(myMesh.nElems);
        // Vec_d a(myMesh.nElems), b(myMesh.nElems), c(myMesh.nElems);
        // Mat_d mid(myMesh.nElems, 3);
        for (int i = 0; i < myMesh.nElems; ++i)
            myMesh.GaussPt[i].resize(GaussInfo.pt.rows(), 3);

#pragma omp parallel for
        for (int i = 0; i < myMesh.nElems; ++i)
        {
            Idx P1 = myMesh.elems(i, 0);
            Idx P2 = myMesh.elems(i, 6);
            double a = (myMesh.nodes(P2, 0) - myMesh.nodes(P1, 0)) * 0.5;
            double b = (myMesh.nodes(P2, 1) - myMesh.nodes(P1, 1)) * 0.5;
            double c = (myMesh.nodes(P2, 2) - myMesh.nodes(P1, 2)) * 0.5;
            Eigen::Vector3d mid;
            mid << (myMesh.nodes(P2, 0) + myMesh.nodes(P1, 0)) * 0.5,
                (myMesh.nodes(P2, 1) + myMesh.nodes(P1, 1)) * 0.5,
                (myMesh.nodes(P2, 2) + myMesh.nodes(P1, 2)) * 0.5;
            for (int j = 0; j < GaussInfo.pt.rows(); ++j)
            {
                myMesh.GaussPt[i].row(j) << a * GaussInfo.pt(j, 0) + mid(0),
                    b * GaussInfo.pt(j, 1) + mid(1),
                    c * GaussInfo.pt(j, 2) + mid(2);
            }
        }
    }
    SparseMat_t<double> assembleMat(const mesh &myMesh, const refGaussInfo &Gauss,
                                    Eigen::Vector3i DfTest, Eigen::Vector3i DfTrail, bool enable_omp)
    {
        // Dftest：测试函数的导数阶，Dftrail：试探函数的导数阶
        // 组装刚度矩阵K = {\int_{\Omega}\frac{\partial\phi_i}{\partial x_m}\frac{\partial\phi_j}{\partial x_n}}_{i,j=0}^{N}
        const Mat_d *testPtr, *trailPtr;
        testPtr = &Gauss.phi[0];
        trailPtr = &Gauss.phi[0];
        int varDiff_test = -1;
        int varDiff_trail = -1; // -1即默认不求导，0，1，2分别对应x,y,z的一阶偏导
        for (int i = 0; i < 3; ++i)
        {
            if (DfTest(i) == 1)
            {
                testPtr = &Gauss.phi[i + 1];
                varDiff_test = i;
            }
            if (DfTrail(i) == 1)
            {
                trailPtr = &Gauss.phi[i + 1];
                varDiff_trail = i;
            }
        }
        Idx nBaseFun = myMesh.elems.cols();
        Mat_d refMat(nBaseFun, nBaseFun);
        Idx nnzLocal = nBaseFun * nBaseFun;
        Idx nnz = nnzLocal * myMesh.nElems;
        std::vector<Eigen::Triplet<double, Idx>> triplets(nnz);
        // triplets.reserve(nnz);
        // double t0 = omp_get_wtime();
        if (enable_omp)
        {
#pragma omp parallel
            {
#pragma omp for collapse(2)
                for (Idx idxRow = 0; idxRow < nBaseFun; ++idxRow)
                {
                    for (Idx idxCol = 0; idxCol < nBaseFun; ++idxCol)
                    {
                        Vec_d temp = testPtr->row(idxRow).array() * trailPtr->row(idxCol).array();
                        refMat(idxRow, idxCol) = temp.dot(Gauss.weight);
                    }
                }
#pragma omp for
                for (Idx n = 0; n < myMesh.nElems; ++n)
                {
                    Idx startGlobal = n * nnzLocal;
                    for (Idx idxRow = 0; idxRow < nBaseFun; ++idxRow)
                    {
                        Idx startLocal = startGlobal + idxRow * nBaseFun;
                        for (Idx idxCol = 0; idxCol < nBaseFun; ++idxCol)
                        {
                            Idx i = startLocal + idxCol;
                            double val = refMat(idxRow, idxCol) * myMesh.Jacobi(n);
                            if (varDiff_test != -1)
                                val *= myMesh.Jacobi_inv(n, varDiff_test);
                            if (varDiff_trail != -1)
                                val *= myMesh.Jacobi_inv(n, varDiff_trail);
                            triplets[i] = {myMesh.elems(n, idxRow),
                                           myMesh.elems(n, idxCol), val};
                        }
                    }
                }
            }
        }
        else
        {
            for (Idx idxRow = 0; idxRow < nBaseFun; ++idxRow)
            {
                for (Idx idxCol = 0; idxCol < nBaseFun; ++idxCol)
                {
                    Vec_d temp = testPtr->row(idxRow).array() * trailPtr->row(idxCol).array();
                    refMat(idxRow, idxCol) = temp.dot(Gauss.weight);
                }
            }
            for (Idx n = 0; n < myMesh.nElems; ++n)
            {
                Idx startGlobal = n * nnzLocal;
                for (Idx idxRow = 0; idxRow < nBaseFun; ++idxRow)
                {
                    Idx startLocal = startGlobal + idxRow * nBaseFun;
                    for (Idx idxCol = 0; idxCol < nBaseFun; ++idxCol)
                    {
                        Idx i = startLocal + idxCol;
                        double val = refMat(idxRow, idxCol) * myMesh.Jacobi(n);
                        if (varDiff_test != -1)
                            val *= myMesh.Jacobi_inv(n, varDiff_test);
                        if (varDiff_trail != -1)
                            val *= myMesh.Jacobi_inv(n, varDiff_trail);
                        triplets[i] = {myMesh.elems(n, idxRow),
                                       myMesh.elems(n, idxCol), val};
                    }
                }
            }
        }
        // double t1 = omp_get_wtime();
        // std::cout << "共启用" << omp_get_max_threads() << "个线程，生成刚triplets:" << t1 - t0 << " s\n";
        // t0 = omp_get_wtime();
        SparseMat_t<double> K(myMesh.nPts, myMesh.nPts);
        K.setFromTriplets(triplets.begin(), triplets.end());
        // t1 = omp_get_wtime();
        // std::cout << "共启用" << omp_get_max_threads() << "个线程，组装刚度矩阵耗时:" << t1 - t0 << " s\n";
        return K;
    }
    //     SparseMat_t<double> assembleMat(const mesh &myMesh, const refGaussInfo &Gauss,
    //                                     Eigen::Vector3i DfTest, Eigen::Vector3i DfTrail)
    //     {
    //         // Dftest：测试函数的导数阶，Dftrail：试探函数的导数阶
    //         // 组装刚度矩阵K = {\int_{\Omega}\frac{\partial\phi_i}{\partial x_m}\frac{\partial\phi_j}{\partial x_n}}_{i,j=0}^{N}
    //         const Mat_d *testPtr, *trailPtr;
    //         testPtr = &Gauss.phi[0];
    //         trailPtr = &Gauss.phi[0];
    //         int varDiff_test = -1;
    //         int varDiff_trail = -1; // -1即默认不求导，0，1，2分别对应x,y,z的一阶偏导
    //         for (int i = 0; i < 3; ++i)
    //         {
    //             if (DfTest(i) == 1)
    //             {
    //                 testPtr = &Gauss.phi[i + 1];
    //                 varDiff_test = i;
    //             }
    //             if (DfTrail(i) == 1)
    //             {
    //                 trailPtr = &Gauss.phi[i + 1];
    //                 varDiff_trail = i;
    //             }
    //         }
    //         Idx nBaseFun = myMesh.elems.cols();
    //         Mat_d refMat(nBaseFun, nBaseFun);
    //         Idx nnzLocal = nBaseFun * nBaseFun;
    //         Idx nnz = nnzLocal * myMesh.nElems;
    //         // std::vector<Eigen::Triplet<double, Idx>> triplets(nnz);
    //         // 分配任务
    //         int size_omp = omp_get_max_threads();
    //         int quo = myMesh.nElems / size_omp;
    //         int rem = myMesh.nElems % size_omp;
    //         std::vector<std::pair<int, int>> task_omp(size_omp);
    //         std::vector<SparseMat_t<double>> K_local(size_omp);
    //         for (int i = 0; i < size_omp; ++i)
    //         {
    //             K_local[i].resize(myMesh.nPts, myMesh.nPts);
    //             int count = rem < 1 ? quo : quo + 1;
    //             int start = i == 0 ? 0 : task_omp[i - 1].first + task_omp[i - 1].second;
    //             rem--;
    //             task_omp[i] = {start, count};
    //         }
    //         // double t0 = omp_get_wtime();
    // #pragma omp parallel
    //         {
    // #pragma omp for collapse(2)
    //             for (Idx idxRow = 0; idxRow < nBaseFun; ++idxRow)
    //             {
    //                 for (Idx idxCol = 0; idxCol < nBaseFun; ++idxCol)
    //                 {
    //                     Vec_d temp = testPtr->row(idxRow).array() * trailPtr->row(idxCol).array();
    //                     refMat(idxRow, idxCol) = temp.dot(Gauss.weight);
    //                 }
    //             }
    //             std::vector<Eigen::Triplet<double, Idx>> triplets;
    //             int rank = omp_get_thread_num();
    //             for (Idx n = task_omp[rank].first; n < task_omp[rank].first + task_omp[rank].second; ++n)
    //             {
    //                 for (Idx idxRow = 0; idxRow < nBaseFun; ++idxRow)
    //                 {
    //                     for (Idx idxCol = 0; idxCol < nBaseFun; ++idxCol)
    //                     {
    //                         double val = refMat(idxRow, idxCol) * myMesh.Jacobi(n);
    //                         if (varDiff_test != -1)
    //                             val *= myMesh.Jacobi_inv(n, varDiff_test);
    //                         if (varDiff_trail != -1)
    //                             val *= myMesh.Jacobi_inv(n, varDiff_trail);
    //                         triplets.emplace_back(Eigen::Triplet<double, Idx>{myMesh.elems(n, idxRow),
    //                                                                           myMesh.elems(n, idxCol), val});
    //                     }
    //                 }
    //             }
    //             K_local[rank].setFromTriplets(triplets.begin(), triplets.end());
    //         }
    //         SparseMat_t<double> K(myMesh.nPts, myMesh.nPts);
    //         for (int i = 0; i < size_omp; ++i)
    //             K += K_local[i];
    //         return K;
    //     }
    template <typename Scalar>
    SparseMat_t<Scalar>
    assembleMat(const Fun_t<Scalar> &coef, const mesh &myMesh,
                const refGaussInfo &Gauss, Eigen::Vector3i DfTest,
                Eigen::Vector3i DfTrail, bool enable_omp)
    {
        if (myMesh.GaussPt.size() != myMesh.nElems)
        {
            std::cout << "请先构造所有单元的Gauss积分节点\n";
            std::exit(EXIT_FAILURE);
        }
        const Mat_d *testPtr, *trailPtr;
        testPtr = &Gauss.phi[0];
        trailPtr = &Gauss.phi[0];
        int varDiff_test = -1;
        int varDiff_trail = -1; // -1即默认不求导，0，1，2分别对应x,y,z的一阶偏导
        for (int i = 0; i < 3; ++i)
        {
            if (DfTest(i) == 1)
            {
                testPtr = &Gauss.phi[i + 1];
                varDiff_test = i;
            }
            if (DfTrail(i) == 1)
            {
                trailPtr = &Gauss.phi[i + 1];
                varDiff_trail = i;
            }
        }
        Idx nBaseFun = myMesh.elems.cols();
        Idx nnzLocal = nBaseFun * nBaseFun;
        Idx nnz = nnzLocal * myMesh.nElems;
        Mat_d refMat(nnzLocal, Gauss.pt.rows()); // 每行对应一个矩阵值
        std::vector<Eigen::Triplet<Scalar, Idx>> triplets(nnz);
        // triplets.reserve(nnz);
        if (enable_omp)
        {
#pragma omp parallel
            {
#pragma omp for collapse(2)
                for (Idx idxRow = 0; idxRow < nBaseFun; ++idxRow)
                {
                    for (Idx idxCol = 0; idxCol < nBaseFun; ++idxCol)
                    {
                        Idx i = idxRow * nBaseFun + idxCol;
                        refMat.row(i) = testPtr->row(idxRow).array() * trailPtr->row(idxCol).array();
                    }
                }
#pragma omp for
                for (Idx n = 0; n < myMesh.nElems; ++n)
                {
                    Idx startGlobal = n * nnzLocal;
                    Eigen::Vector<Scalar, Eigen::Dynamic> value_coef = coef(myMesh.GaussPt[n]);
                    for (Idx idxRow = 0; idxRow < nBaseFun; ++idxRow)
                    {
                        Idx startLocal = startGlobal + idxRow * nBaseFun;
                        for (Idx idxCol = 0; idxCol < nBaseFun; ++idxCol)
                        {
                            Idx i = startLocal + idxCol;
                            Idx iLocal = idxRow * nBaseFun + idxCol;
                            Eigen::Vector<Scalar, Eigen::Dynamic> temp = value_coef.array() * refMat.row(iLocal).reshaped().array();
                            Scalar val = temp.transpose() * Gauss.weight;
                            // temp.dot(Gauss.weight);
                            val *= myMesh.Jacobi(n);
                            if (varDiff_test != -1)
                                val *= myMesh.Jacobi_inv(n, varDiff_test);
                            if (varDiff_trail != -1)
                                val *= myMesh.Jacobi_inv(n, varDiff_trail);
                            triplets[i] = {myMesh.elems(n, idxRow),
                                           myMesh.elems(n, idxCol), val};
                        }
                    }
                }
            }
        }
        else
        {
            for (Idx idxRow = 0; idxRow < nBaseFun; ++idxRow)
            {
                for (Idx idxCol = 0; idxCol < nBaseFun; ++idxCol)
                {
                    Idx i = idxRow * nBaseFun + idxCol;
                    refMat.row(i) = testPtr->row(idxRow).array() * trailPtr->row(idxCol).array();
                }
            }
            for (Idx n = 0; n < myMesh.nElems; ++n)
            {
                Idx startGlobal = n * nnzLocal;
                Eigen::Vector<Scalar, Eigen::Dynamic> value_coef = coef(myMesh.GaussPt[n]);
                for (Idx idxRow = 0; idxRow < nBaseFun; ++idxRow)
                {
                    Idx startLocal = startGlobal + idxRow * nBaseFun;
                    for (Idx idxCol = 0; idxCol < nBaseFun; ++idxCol)
                    {
                        Idx i = startLocal + idxCol;
                        Idx iLocal = idxRow * nBaseFun + idxCol;
                        Eigen::Vector<Scalar, Eigen::Dynamic> temp = value_coef.array() * refMat.row(iLocal).reshaped().array();
                        // Scalar val = temp.dot(Gauss.weight);
                        Scalar val = temp.transpose() * Gauss.weight;
                        val *= myMesh.Jacobi(n);
                        if (varDiff_test != -1)
                            val *= myMesh.Jacobi_inv(n, varDiff_test);
                        if (varDiff_trail != -1)
                            val *= myMesh.Jacobi_inv(n, varDiff_trail);
                        triplets[i] = {myMesh.elems(n, idxRow),
                                       myMesh.elems(n, idxCol), val};
                    }
                }
            }
        }
        SparseMat_t<Scalar> K(myMesh.nPts, myMesh.nPts);
        K.setFromTriplets(triplets.begin(), triplets.end());
        return K;
    }
    Vec_t<double> assembleVec(const mesh &myMesh, const refGaussInfo &Gauss)
    {
        int nBaseFun = myMesh.elems.cols();
        // 分配任务
        int size_omp = omp_get_max_threads();
        int quo = myMesh.nElems / size_omp;
        int rem = myMesh.nElems % size_omp;
        std::vector<std::pair<int, int>> task_omp(size_omp);
        // std::vector<Vec_d> V_local(size_omp);
        for (int i = 0; i < size_omp; ++i)
        {
            // V_local[i] = Vec_d::Zero(myMesh.nPts);
            int count = rem < 1 ? quo : quo + 1;
            int start = i == 0 ? 0 : task_omp[i - 1].first + task_omp[i - 1].second;
            rem--;
            task_omp[i] = {start, count};
        }
        Vec_d V = Vec_d::Zero(myMesh.nPts);
#pragma omp parallel
        {
            Vec_d Vlocal = Vec_d::Zero(myMesh.nPts);
            int rank = omp_get_thread_num();
            Vec_d refVec(nBaseFun);
            for (int i = 0; i < nBaseFun; ++i)
                refVec(i) = Gauss.phi[0].row(i).dot(Gauss.weight);
            for (Idx n = task_omp[rank].first; n < task_omp[rank].first + task_omp[rank].second; ++n)
            {
                for (int i = 0; i < nBaseFun; ++i)
                    Vlocal(myMesh.elems(n, i)) += refVec(i) * myMesh.Jacobi(n);
            }
#pragma omp critical
            {
                V += Vlocal;
            }
        }
        return V;
    }
    template <typename Scalar>
    Vec_t<Scalar> assembleVec(const Fun_t<Scalar> &RHS, const mesh &myMesh, const refGaussInfo &Gauss)
    {
        if (myMesh.GaussPt.size() != myMesh.nElems)
        {
            std::cout << "请先构造所有单元的Gauss积分节点\n";
            std::exit(EXIT_FAILURE);
        }
        int nBaseFun = myMesh.elems.cols();
        // 分配任务
        int size_omp = omp_get_max_threads();
        int quo = myMesh.nElems / size_omp;
        int rem = myMesh.nElems % size_omp;
        std::vector<std::pair<int, int>> task_omp(size_omp);
        // std::vector<Vec_d> V_local(size_omp);
        for (int i = 0; i < size_omp; ++i)
        {
            // V_local[i] = Vec_d::Zero(myMesh.nPts);
            int count = rem < 1 ? quo : quo + 1;
            int start = i == 0 ? 0 : task_omp[i - 1].first + task_omp[i - 1].second;
            rem--;
            task_omp[i] = {start, count};
        }
        Vec_t<Scalar> V = Vec_t<Scalar>::Zero(myMesh.nPts);
#pragma omp parallel
        {
            Vec_t<Scalar> Vlocal = Vec_t<Scalar>::Zero(myMesh.nPts);
            int rank = omp_get_thread_num();
            // Vec_d refVec(nBaseFun);
            // for (int i = 0; i < nBaseFun; ++i)
            //     refVec(i) = Gauss.phi[0].row(i).dot(Gauss.weight);
            Vec_t<Scalar> valFun(Gauss.pt.rows());
            for (Idx n = task_omp[rank].first; n < task_omp[rank].first + task_omp[rank].second; ++n)
            {
                valFun = RHS(myMesh.GaussPt[n]);
                for (int i = 0; i < nBaseFun; ++i)
                {
                    Vec_t<Scalar> temp = valFun.array() * Gauss.phi[0].row(i).reshaped().array();
                    Vlocal(myMesh.elems(n, i)) += temp.dot(Gauss.weight) * myMesh.Jacobi(n);
                }
            }
#pragma omp critical
            {
                V += Vlocal;
            }
        }
        return V;
    }
    bool almostEqual(double x, double y, double eps)
    {
        return std::abs(x - y) <= eps;
    }
    Vec_d getPointNd(const Mat_d &X, Idx row)
    {
        Vec_d v = X.row(row);
        return v;
    }
    std::size_t PointNdHash::operator()(const Vec_d &p) const
    {
        std::size_t h = 0;

        for (Idx i = 0; i < p.size(); ++i)
        {
            std::size_t hx = std::hash<double>{}(p(i));

            // hash combine
            h ^= hx + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return h;
    }
    bool PointNdEqual::operator()(const Vec_d &a, const Vec_d &b) const
    {
        if (a.size() != b.size())
        {
            std::cout << "点的维度不同!\n";
            std::exit(EXIT_FAILURE);
        }
        for (int k = 0; k < a.size(); ++k)
        {
            if (!almostEqual(a[k], b[k]))
            {
                return false;
            }
        }
        return true;
    }
    std::pair<std::vector<Idx>, std::vector<Idx>>
    findCommonDof(const Mat_d &dofIdx1, const Mat_d &dofIdx2,
                  std::pair<int, double> interFace,
                  double scale, Idx searchStart1, Idx searchStart2)
    {
        // dofIdx1, dofIdx2:自由度索引，一行一个自由度. interFace:两个区域的交界面信息，first为维度，second为交界面坐标
        // searchStart1：dofIdx1的扫描起点，用于确认前searchStart1个点不会与dofIdx2相交的情况，默认为0
        // searchStart2：dofIdx2的扫描起点，用于确认前searchStart2个点不会与dofIdx1相交的情况，默认为0
        std::unordered_map<Vec_d, Idx, PointNdHash, PointNdEqual> mapB;
        mapB.reserve(static_cast<std::size_t>(dofIdx2.rows()));
        double eps = EPS * scale;
        for (Idx j = searchStart2; j < dofIdx2.rows(); ++j)
        {
            if (almostEqual(dofIdx2(j, interFace.first), interFace.second, eps))
            {
                mapB.emplace(getPointNd(dofIdx2, j), j);
            }
        }
        std::vector<Idx> Idx1, Idx2;
        for (Idx i = searchStart1; i < dofIdx1.rows(); ++i)
        {
            if (!almostEqual(dofIdx1(i, interFace.first), interFace.second, eps))
            {
                continue;
            }
            Vec_d p = getPointNd(dofIdx1, i);
            auto it = mapB.find(p);
            if (it != mapB.end())
            {
                Idx1.push_back(i);
                Idx2.push_back(it->second);
            }
        }
        return {Idx1, Idx2};
    }
    mergeFEMMesh_info mergeFEMMesh(const mesh &myMesh1, const mesh &myMesh2,
                                   std::pair<int, double> interFace,
                                   double scale)
    {
        auto [Idx1, Idx2] = findCommonDof(myMesh1.nodes, myMesh2.nodes, interFace, scale);
        std::vector<Idx> mark(myMesh2.nPts, -1);
        for (Idx i = 0; i < Idx2.size(); ++i)
        {
            mark[Idx2[i]] = Idx1[i];
        }
        Idx count = 0;
        for (Idx i = 0; i < mark.size(); ++i)
        {
            if (mark[i] != -1)
                ++count;
            else
            {
                mark[i] = myMesh1.nPts + i - count;
            }
        }
        mesh myMesh;
        myMesh.nPts = myMesh1.nPts + myMesh2.nPts - Idx1.size();
        myMesh.nElems = myMesh1.nElems + myMesh2.nElems;

        myMesh.nodes.resize(myMesh.nPts, 3);
        myMesh.elems.resize(myMesh.nElems, myMesh1.elems.cols());
        myMesh.meshtype = myMesh1.meshtype;
        myMesh.elemtype = myMesh1.elemtype;
        myMesh.nodes.topRows(myMesh1.nPts) = myMesh1.nodes;
        myMesh.elems << myMesh1.elems, myMesh2.elems;

        // 填入网格点
        for (Idx i = 0; i < mark.size(); ++i)
        {
            if (mark[i] >= myMesh1.nPts)
                myMesh.nodes.row(mark[i]) = myMesh2.nodes.row(i);
        }
        // 填入单元
        for (Idx i = myMesh1.nElems; i < myMesh.nElems; ++i)
        {
            for (Idx j = 0; j < myMesh.elems.cols(); ++j)
            {
                myMesh.elems(i, j) = mark[myMesh.elems(i, j)];
            }
        }
        return mergeFEMMesh_info{myMesh, Idx2.size(), mark};
    }
    template <typename Scalar>
    SparseMat_t<Scalar> mergeSparseMat(const SparseMat_t<Scalar> &K1, const SparseMat_t<Scalar> &K2,
                                       const std::vector<Idx> &idx1, const std::vector<Idx> &idx2)
    {
        // idx1 ：K1中的全局自由度编号
        // idx2 ：K2中的局部自由度编号
        constexpr Idx INVALID =
            std::numeric_limits<Idx>::max();
        std::vector<Idx> mark(K2.rows(), INVALID);
        for (Idx i = 0; i < idx2.size(); ++i)
        {
            mark[idx2[i]] = idx1[i];
        }
        // std::vector<Idx> idxTemp(K2.rows());
        Idx count = 0;
        for (Idx i = 0; i < K2.rows(); ++i)
        {
            if (mark[i] != INVALID) // 被标记的点
            {
                // idxTemp[i] = mark[i];
                ++count;
            }
            else
            {
                mark[i] = i + K1.rows() - count;
            }
        }
        std::vector<Eigen::Triplet<Scalar, Idx>> triplets;
        triplets.reserve((K1.nonZeros() + K2.nonZeros()));
        for (Idx col = 0; col < K1.outerSize(); ++col)
        {
            // 内层迭代器：遍历当前列所有非零元
            for (typename SparseMat_t<Scalar>::InnerIterator iter(K1, col); iter; ++iter)
            {
                triplets.emplace_back(iter.row(), iter.col(), iter.value());
            }
        }
        for (Idx col = 0; col < K2.outerSize(); ++col)
        {
            // 内层迭代器：遍历当前列所有非零元
            for (typename SparseMat_t<Scalar>::InnerIterator iter(K2, col); iter; ++iter)
            {
                Idx i = mark[iter.row()];
                Idx j = mark[iter.col()];
                triplets.emplace_back(i, j, iter.value());
            }
        }
        Idx N = K1.rows() + K2.rows() - idx2.size();
        SparseMat_t<Scalar> K(N, N);
        K.setFromTriplets(triplets.begin(), triplets.end());
        return K;
    }
    template <typename Scalar>
    Vec_t<Scalar> mergeVec(const Vec_t<Scalar> &F1, const Vec_t<Scalar> &F2,
                           const std::vector<Idx> &idx1, const std::vector<Idx> &idx2)
    { // F = [F1;F2]; F(idx2+N1,:)=[];
        constexpr Idx INVALID =
            std::numeric_limits<Idx>::max();
        std::vector<Idx> mark(F2.rows(), INVALID);
        for (Idx i = 0; i < idx2.size(); ++i)
        {
            mark[idx2[i]] = idx1[i];
        }
        Idx nDof1 = F1.size();
        Idx nDof2 = F2.size();
        Idx ncomDof = idx2.size();
        Vec_t<Scalar> F(nDof1 + nDof2 - ncomDof);
        F.head(nDof1) = F1;
        Idx count = nDof1;
        for (Idx i = 0; i < nDof2; ++i)
        {
            if (mark[i] == INVALID)
            {
                F(count) = F2(i);
                count++;
            }
            else
                F(mark[i]) += F2(i);
        }
        return F;
    }
    template <typename Scalar>
    Mat_t<Scalar> mergeDofIdx(const Mat_t<Scalar> &dofIdx1,
                              const Mat_t<Scalar> &dofIdx2,
                              const std::vector<Idx> &idx2)
    {
        Idx nDof1 = dofIdx1.rows();
        Idx nDof2 = dofIdx2.rows();
        Idx ncomDof = idx2.size();
        Mat_t<Scalar> dofIdx(nDof1 + nDof2 - ncomDof, dofIdx1.cols());
        dofIdx.topRows(nDof1) = dofIdx1;
        constexpr Idx INVALID =
            std::numeric_limits<Idx>::max();
        std::vector<Idx> mark(dofIdx2.rows(), INVALID);
        for (Idx i = 0; i < idx2.size(); ++i)
        {
            mark[idx2[i]] = -1;
        }
        Idx pos = nDof1;
        for (Idx i = 0; i < nDof2; ++i)
        {
            if (mark[i] == INVALID)
            {
                dofIdx.row(pos) = dofIdx2.row(i);
                ++pos;
            }
        }
        return dofIdx;
    }
    template <typename Scalar>
    mergeFEMMat_info<Scalar>
    mergeFEMMat(const Mat_d &dofIdx1, const Mat_d &dofIdx2,
                const SparseMat_t<Scalar> &K1, const SparseMat_t<Scalar> &K2,
                std::pair<int, double> interFace, double scale)
    {
        SparseMat_t<Scalar> K;
        Mat_t<double> dofIdx;
        auto [idx1, idx2] = findCommonDof(dofIdx1, dofIdx2, interFace, scale);
        K = mergeSparseMat<Scalar>(K1, K2, idx1, idx2);
        dofIdx = mergeDofIdx<double>(dofIdx1, dofIdx2, idx2);
        return mergeFEMMat_info<Scalar>{K, dofIdx, idx1, idx2};
    }

    /*模板定义*/
    template SparseMat_t<double> assembleMat<double>(
        const Fun_t<double> &,
        const mesh &,
        const refGaussInfo &,
        Eigen::Vector3i,
        Eigen::Vector3i, bool enable_omp);
    template SparseMat_t<Complex> assembleMat<Complex>(
        const Fun_t<Complex> &,
        const mesh &,
        const refGaussInfo &,
        Eigen::Vector3i,
        Eigen::Vector3i, bool enable_omp);
    template Vec_t<double>
    assembleVec<double>(const Fun_t<double> &,
                        const mesh &,
                        const refGaussInfo &);
    template Vec_t<Complex>
    assembleVec<Complex>(const Fun_t<Complex> &,
                         const mesh &,
                         const refGaussInfo &);
    template mergeFEMMat_info<double>
    mergeFEMMat<double>(const Mat_d &dofIdx1, const Mat_d &dofIdx2,
                        const SparseMat_t<double> &K1, const SparseMat_t<double> &K2,
                        std::pair<int, double> interFace, double scale);
    template mergeFEMMat_info<Complex>
    mergeFEMMat<Complex>(const Mat_d &dofIdx1, const Mat_d &dofIdx2,
                         const SparseMat_t<Complex> &K1, const SparseMat_t<Complex> &K2,
                         std::pair<int, double> interFace, double scale);
    template SparseMat_t<double>
    mergeSparseMat<double>(const SparseMat_t<double> &K1,
                           const SparseMat_t<double> &K2,
                           const std::vector<Idx> &idx1,
                           const std::vector<Idx> &idx2);
    template SparseMat_t<Complex>
    mergeSparseMat<Complex>(const SparseMat_t<Complex> &K1,
                            const SparseMat_t<Complex> &K2,
                            const std::vector<Idx> &idx1,
                            const std::vector<Idx> &idx2);
    template Mat_t<double>
    mergeDofIdx(const Mat_t<double> &dofIdx1,
                const Mat_t<double> &dofIdx2,
                const std::vector<Idx> &idx2);
    template Mat_t<Idx>
    mergeDofIdx(const Mat_t<Idx> &dofIdx1,
                const Mat_t<Idx> &dofIdx2,
                const std::vector<Idx> &idx2);
    template Vec_t<double>
    mergeVec(const Vec_t<double> &F1,
             const Vec_t<double> &F2,
             const std::vector<Idx> &idx1,
             const std::vector<Idx> &idx2);
    template Vec_t<Complex>
    mergeVec(const Vec_t<Complex> &F1,
             const Vec_t<Complex> &F2,
             const std::vector<Idx> &idx1,
             const std::vector<Idx> &idx2);
}