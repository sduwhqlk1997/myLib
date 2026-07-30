#include "mat2eigen.hpp"
#include <matio.h>
#include <iostream>
namespace wrMatFile
{
    void readSparseMatFromMatlab(const char *filename, const char *varname, SparseMat &K)
    {
        mat_t *mat = Mat_Open(filename, MAT_ACC_RDONLY);
        if (!mat)
        {
            std::cerr << "错误：无法打开 .mat 文件！" << std::endl;
        }
        matvar_t *p = Mat_VarRead(mat, varname);
        int rows = p->dims[0];
        int cols = p->dims[1];
        mat_sparse_t *KData = static_cast<mat_sparse_t *>(p->data);
        mat_uint32_t *ir = KData->ir;
        mat_uint32_t *jc = KData->jc;
        int nnz = jc[cols];
        mat_complex_split_t *complexData = static_cast<mat_complex_split_t *>(KData->data);

        double *realPart = static_cast<double *>(complexData->Re);
        double *imagPart = static_cast<double *>(complexData->Im);
        K.resize(rows, cols);
        K.setZero();
        std::vector<Eigen::Triplet<Complex>> triplets;
        triplets.reserve(nnz);

        for (int j = 0; j < cols; ++j)
        {
            for (int idx = jc[j]; idx < jc[j + 1]; ++idx)
            {
                int i = ir[idx];
                Complex val(realPart[idx], imagPart[idx]);
                triplets.emplace_back(i, j, val);
            }
        }
        K.setFromSortedTriplets(triplets.begin(), triplets.end());
        // 释放资源
        Mat_VarFree(p);
        Mat_Close(mat);
    }
    void readDenseMatFromMatlab(const char *filename, const char *varname, DenseMat &K)
    {
        mat_t *mat = Mat_Open(filename, MAT_ACC_RDONLY);
        if (!mat)
        {
            std::cerr << "错误：无法打开 .mat 文件！" << std::endl;
        }
        matvar_t *p = Mat_VarRead(mat, varname);
        int rows = p->dims[0]; // 行数
        int cols = p->dims[1]; // 列数
        double *data = static_cast<double *>(p->data);
        Eigen::Map<DenseMat> temp(data, rows, cols);
        K = temp;
        // 释放资源
        Mat_VarFree(p);
        Mat_Close(mat);
    }
    void readVecFromMatlab(const char *filename, const char *varname, Vec &K)
    {
        mat_t *mat = Mat_Open(filename, MAT_ACC_RDONLY);
        if (!mat)
        {
            std::cerr << "错误：无法打开 .mat 文件！" << std::endl;
        }
        matvar_t *p = Mat_VarRead(mat, varname);
        int rows = p->dims[0];
        int cols = p->dims[1];
        int n = std::max(rows, cols);
        double *data = static_cast<double *>(p->data);
        Eigen::Map<Eigen::VectorXd> v(data, n);
        K = v.cast<Complex>();
        // 释放资源
        Mat_VarFree(p);
        Mat_Close(mat);
    }
    void saveEigenDenseMat2Mat(const char *filename, DenseMat &K)
    {
        if (filename == nullptr)
        {
            throw std::invalid_argument("filename is nullptr");
        }

        // 创建 MAT 文件
        mat_t *matfp = Mat_CreateVer(filename, nullptr, MAT_FT_MAT5);
        if (matfp == nullptr)
        {
            throw std::runtime_error("Failed to create mat file.");
        }

        size_t dims[2];
        dims[0] = static_cast<size_t>(K.rows());
        dims[1] = static_cast<size_t>(K.cols());

        std::string varname(filename);
        size_t dot = varname.find('.');
        if (dot != std::string::npos)
            varname = varname.substr(0, dot);
        // MATLAB 与 Eigen 默认均为列主序
        matvar_t *matvar = Mat_VarCreate(
            varname.c_str(), // MATLAB变量名
            MAT_C_DOUBLE,    // C类型
            MAT_T_DOUBLE,    // MATLAB类型
            2,               // 维数
            dims,
            (void *)K.data(), // 数据指针
            MAT_F_DONT_COPY_DATA);

        if (matvar == nullptr)
        {
            Mat_Close(matfp);
            throw std::runtime_error("Failed to create MATLAB variable.");
        }

        if (Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE) != 0)
        {
            Mat_VarFree(matvar);
            Mat_Close(matfp);
            throw std::runtime_error("Failed to write MATLAB variable.");
        }

        Mat_VarFree(matvar);
        Mat_Close(matfp);
    }

    void saveEigenDenseMat2Mat(const char *filename, DenseMatcd &K)
    {
        if (filename == nullptr)
        {
            throw std::invalid_argument("filename is nullptr");
        }

        // 创建 MAT 文件
        mat_t *matfp = Mat_CreateVer(filename, nullptr, MAT_FT_MAT5);

        if (matfp == nullptr)
        {
            throw std::runtime_error("Failed to create mat file.");
        }

        size_t rows = static_cast<size_t>(K.rows());
        size_t cols = static_cast<size_t>(K.cols());

        size_t dims[2] = {rows, cols};

        /*
            MATLAB complex storage:
            Re: double array
            Im: double array
        */

        std::vector<double> realPart(rows * cols);
        std::vector<double> imagPart(rows * cols);

        /*
            Eigen默认列主序，与MATLAB一致

            index:
            i + j*rows
        */
        for (size_t j = 0; j < cols; ++j)
        {
            for (size_t i = 0; i < rows; ++i)
            {
                size_t id = i + j * rows;

                realPart[id] = K(i, j).real();
                imagPart[id] = K(i, j).imag();
            }
        }

        mat_complex_split_t complexData;

        complexData.Re = realPart.data();
        complexData.Im = imagPart.data();

        std::string varname(filename);
        size_t dot = varname.find('.');
        if (dot != std::string::npos)
            varname = varname.substr(0, dot);

        matvar_t *matvar = Mat_VarCreate(
            varname.c_str(), // MATLAB变量名
            MAT_C_DOUBLE,
            MAT_T_DOUBLE,
            2,
            dims,
            &complexData,
            MAT_F_COMPLEX | MAT_F_DONT_COPY_DATA);

        if (matvar == nullptr)
        {
            Mat_Close(matfp);
            throw std::runtime_error(
                "Failed to create MATLAB complex variable.");
        }

        if (Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE) != 0)
        {
            Mat_VarFree(matvar);
            Mat_Close(matfp);

            throw std::runtime_error(
                "Failed to write MATLAB variable.");
        }

        Mat_VarFree(matvar);
        Mat_Close(matfp);
    }
    void saveEigenDenseMat2Mat(const char *filename, Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic> &K)
    {
        if (filename == nullptr)
        {
            throw std::invalid_argument("filename is nullptr");
        }

        // 创建 MAT 文件 (MATLAB v5 格式)
        mat_t *matfp = Mat_CreateVer(filename, nullptr, MAT_FT_MAT5);
        if (matfp == nullptr)
        {
            throw std::runtime_error("Failed to create mat file.");
        }

        // MATLAB 矩阵维度
        size_t dims[2];
        dims[0] = static_cast<size_t>(K.rows());
        dims[1] = static_cast<size_t>(K.cols());

        /*
         * Eigen::Index 通常是 long 或 long long，
         * MATLAB int64 对应 int64_t。
         *
         * 为保证平台一致性，复制到 int64_t 缓冲区。
         */
        Eigen::Matrix<int64_t,
                      Eigen::Dynamic,
                      Eigen::Dynamic>
            K64 =
                K.template cast<int64_t>();

        /*
         * MATLAB:
         *
         * class(K)
         * ans =
         *     'int64'
         *
         */
        std::string varname(filename);
        size_t dot = varname.find('.');
        if (dot != std::string::npos)
            varname = varname.substr(0, dot);
        matvar_t *matvar = Mat_VarCreate(
            varname.c_str(), // MATLAB变量名
            MAT_C_INT64,     // C数据类型
            MAT_T_INT64,     // MATLAB数据类型
            2,               // 二维矩阵
            dims,
            static_cast<void *>(K64.data()),
            0 // 让 matio 自己复制数据
        );

        if (matvar == nullptr)
        {
            Mat_Close(matfp);
            throw std::runtime_error(
                "Failed to create MATLAB variable.");
        }

        // 写入MAT文件
        if (Mat_VarWrite(matfp,
                         matvar,
                         MAT_COMPRESSION_NONE) != 0)
        {
            Mat_VarFree(matvar);
            Mat_Close(matfp);

            throw std::runtime_error(
                "Failed to write MATLAB variable.");
        }

        Mat_VarFree(matvar);
        Mat_Close(matfp);
    }
    void saveEigenVec2Mat(const char *filename, Vec &V)
    {
        if (!filename)
            throw std::invalid_argument("filename is nullptr");

        mat_t *matfp = Mat_CreateVer(filename, nullptr, MAT_FT_MAT5);

        if (!matfp)
            throw std::runtime_error("Failed to create mat file.");

        const size_t n = static_cast<size_t>(V.size());

        std::vector<double> re(n);
        std::vector<double> im(n);

        for (size_t i = 0; i < n; ++i)
        {
            re[i] = V(i).real();
            im[i] = V(i).imag();
        }

        mat_complex_split_t z;
        z.Re = re.data();
        z.Im = im.data();

        size_t dims[2] = {n, 1};
        std::string varname(filename);
        size_t dot = varname.find('.');
        if (dot != std::string::npos)
            varname = varname.substr(0, dot);
        matvar_t *matvar =
            Mat_VarCreate(
                varname.c_str(),
                MAT_C_DOUBLE,
                MAT_T_DOUBLE,
                2,
                dims,
                &z,
                MAT_F_COMPLEX);

        if (!matvar)
        {
            Mat_Close(matfp);
            throw std::runtime_error("Mat_VarCreate failed.");
        }

        int err =
            Mat_VarWrite(
                matfp,
                matvar,
                MAT_COMPRESSION_NONE);

        if (err != 0)
        {
            Mat_VarFree(matvar);
            Mat_Close(matfp);
            throw std::runtime_error("Mat_VarWrite failed.");
        }

        Mat_VarFree(matvar);
        Mat_Close(matfp);
    }
    void saveEigenSparseMat2Mat(const char *filename, SparseMat &K)
    {
        if (filename == nullptr)
        {
            throw std::invalid_argument("filename is nullptr.");
        }

        K.makeCompressed();

        mat_t *matfp =
            Mat_CreateVer(filename, nullptr, MAT_FT_MAT5);

        if (matfp == nullptr)
        {
            throw std::runtime_error(
                "Failed to create MAT file.");
        }

        const size_t rows = static_cast<size_t>(K.rows());
        const size_t cols = static_cast<size_t>(K.cols());
        const size_t nnz = static_cast<size_t>(K.nonZeros());

        /*
         * MATLAB sparse matrix structure
         */
        mat_sparse_t *sparse =
            static_cast<mat_sparse_t *>(
                malloc(sizeof(mat_sparse_t)));

        if (sparse == nullptr)
        {
            Mat_Close(matfp);
            throw std::bad_alloc();
        }

        sparse->nzmax = nnz;
        sparse->nir = nnz;
        sparse->njc = cols + 1;
        sparse->ndata = nnz;

        /*
         * CSC indices
         */
        sparse->ir =
            static_cast<mat_uint32_t *>(
                malloc(nnz * sizeof(mat_uint32_t)));

        sparse->jc =
            static_cast<mat_uint32_t *>(
                malloc((cols + 1) * sizeof(mat_uint32_t)));

        if (sparse->ir == nullptr ||
            sparse->jc == nullptr)
        {
            free(sparse->ir);
            free(sparse->jc);
            free(sparse);

            Mat_Close(matfp);
            throw std::bad_alloc();
        }

        for (size_t k = 0; k < nnz; ++k)
        {
            sparse->ir[k] =
                static_cast<mat_uint32_t>(
                    K.innerIndexPtr()[k]);
        }

        for (size_t k = 0; k < cols + 1; ++k)
        {
            sparse->jc[k] =
                static_cast<mat_uint32_t>(
                    K.outerIndexPtr()[k]);
        }

        /*
         * complex values
         */
        double *realPart =
            static_cast<double *>(
                malloc(nnz * sizeof(double)));

        double *imagPart =
            static_cast<double *>(
                malloc(nnz * sizeof(double)));

        if (realPart == nullptr ||
            imagPart == nullptr)
        {
            free(realPart);
            free(imagPart);
            free(sparse->ir);
            free(sparse->jc);
            free(sparse);

            Mat_Close(matfp);
            throw std::bad_alloc();
        }

        const Complex *valPtr = K.valuePtr();

        for (size_t k = 0; k < nnz; ++k)
        {
            realPart[k] = valPtr[k].real();
            imagPart[k] = valPtr[k].imag();
        }

        mat_complex_split_t *complexData =
            static_cast<mat_complex_split_t *>(
                malloc(sizeof(mat_complex_split_t)));

        if (complexData == nullptr)
        {
            free(realPart);
            free(imagPart);
            free(sparse->ir);
            free(sparse->jc);
            free(sparse);

            Mat_Close(matfp);
            throw std::bad_alloc();
        }

        complexData->Re = realPart;
        complexData->Im = imagPart;

        sparse->data = complexData;

        size_t dims[2] = {rows, cols};
        std::string varname(filename);
        size_t dot = varname.find('.');
        if (dot != std::string::npos)
            varname = varname.substr(0, dot);
        matvar_t *matvar =
            Mat_VarCreate(
                varname.c_str(),
                MAT_C_SPARSE,
                MAT_T_DOUBLE,
                2,
                dims,
                sparse,
                MAT_F_COMPLEX);

        if (matvar == nullptr)
        {
            free(complexData);
            free(realPart);
            free(imagPart);
            free(sparse->ir);
            free(sparse->jc);
            free(sparse);

            Mat_Close(matfp);

            throw std::runtime_error(
                "Failed to create sparse variable.");
        }

        if (Mat_VarWrite(
                matfp,
                matvar,
                MAT_COMPRESSION_NONE) != 0)
        {
            Mat_VarFree(matvar);

            free(complexData);
            free(realPart);
            free(imagPart);
            free(sparse->ir);
            free(sparse->jc);
            free(sparse);

            Mat_Close(matfp);

            throw std::runtime_error(
                "Failed to write sparse matrix.");
        }

        Mat_VarFree(matvar);

        free(complexData);
        free(realPart);
        free(imagPart);
        free(sparse->ir);
        free(sparse->jc);
        free(sparse);

        Mat_Close(matfp);
    }
    void saveEigenSparseMat2Mat(const char *filename, Eigen::SparseMatrix<double, Eigen::ColMajor> &K)
    {
        if (filename == nullptr)
        {
            throw std::invalid_argument("filename is nullptr.");
        }

        K.makeCompressed();

        mat_t *matfp = Mat_CreateVer(filename, nullptr, MAT_FT_MAT5);

        if (matfp == nullptr)
        {
            throw std::runtime_error("Failed to create MAT file.");
        }

        const size_t rows = static_cast<size_t>(K.rows());
        const size_t cols = static_cast<size_t>(K.cols());
        const size_t nnz = static_cast<size_t>(K.nonZeros());

        // MATLAB sparse matrix structure
        mat_sparse_t *sparse =
            static_cast<mat_sparse_t *>(malloc(sizeof(mat_sparse_t)));

        if (sparse == nullptr)
        {
            Mat_Close(matfp);
            throw std::bad_alloc();
        }

        sparse->nzmax = nnz;
        sparse->nir = nnz;
        sparse->njc = cols + 1;
        sparse->ndata = nnz;

        // CSC row indices
        sparse->ir = static_cast<mat_uint32_t *>(
            malloc(nnz * sizeof(mat_uint32_t)));

        // CSC column pointers
        sparse->jc = static_cast<mat_uint32_t *>(
            malloc((cols + 1) * sizeof(mat_uint32_t)));

        if (sparse->ir == nullptr || sparse->jc == nullptr)
        {
            free(sparse->ir);
            free(sparse->jc);
            free(sparse);

            Mat_Close(matfp);
            throw std::bad_alloc();
        }

        for (size_t k = 0; k < nnz; ++k)
        {
            sparse->ir[k] =
                static_cast<mat_uint32_t>(K.innerIndexPtr()[k]);
        }

        for (size_t k = 0; k < cols + 1; ++k)
        {
            sparse->jc[k] =
                static_cast<mat_uint32_t>(K.outerIndexPtr()[k]);
        }

        // Real values
        double *values =
            static_cast<double *>(malloc(nnz * sizeof(double)));

        if (values == nullptr)
        {
            free(sparse->ir);
            free(sparse->jc);
            free(sparse);

            Mat_Close(matfp);
            throw std::bad_alloc();
        }

        const double *valPtr = K.valuePtr();

        for (size_t k = 0; k < nnz; ++k)
        {
            values[k] = valPtr[k];
        }

        sparse->data = values;

        size_t dims[2] = {rows, cols};

        std::string varname(filename);
        size_t dot = varname.find('.');
        if (dot != std::string::npos)
            varname = varname.substr(0, dot);

        matvar_t *matvar =
            Mat_VarCreate(
                varname.c_str(),
                MAT_C_SPARSE,
                MAT_T_DOUBLE,
                2,
                dims,
                sparse,
                0); // 实数矩阵，没有 MAT_F_COMPLEX

        if (matvar == nullptr)
        {
            free(values);
            free(sparse->ir);
            free(sparse->jc);
            free(sparse);

            Mat_Close(matfp);

            throw std::runtime_error("Failed to create sparse variable.");
        }

        if (Mat_VarWrite(
                matfp,
                matvar,
                MAT_COMPRESSION_NONE) != 0)
        {
            Mat_VarFree(matvar);

            free(values);
            free(sparse->ir);
            free(sparse->jc);
            free(sparse);

            Mat_Close(matfp);

            throw std::runtime_error("Failed to write sparse matrix.");
        }

        Mat_VarFree(matvar);

        free(values);
        free(sparse->ir);
        free(sparse->jc);
        free(sparse);

        Mat_Close(matfp);
    }
    void saveEigenVec2Mat(const char *filename, Eigen::VectorXd &V)
    {
        if (!filename)
            throw std::invalid_argument("filename is nullptr");

        mat_t *matfp = Mat_CreateVer(filename, nullptr, MAT_FT_MAT5);
        if (!matfp)
            throw std::runtime_error("Failed to create mat file.");

        size_t dims[2] = {static_cast<size_t>(V.size()), 1};

        // 变量名使用文件名（去掉扩展名）
        std::string varname(filename);
        size_t dot = varname.find('.');
        if (dot != std::string::npos)
            varname = varname.substr(0, dot);

        matvar_t *matvar = Mat_VarCreate(
            varname.c_str(),
            MAT_C_DOUBLE,
            MAT_T_DOUBLE,
            2,
            dims,
            V.data(),
            0);

        if (!matvar)
        {
            Mat_Close(matfp);
            throw std::runtime_error("Mat_VarCreate failed.");
        }

        int err = Mat_VarWrite(
            matfp,
            matvar,
            MAT_COMPRESSION_NONE);

        if (err != 0)
        {
            Mat_VarFree(matvar);
            Mat_Close(matfp);
            throw std::runtime_error("Mat_VarWrite failed.");
        }

        Mat_VarFree(matvar);
        Mat_Close(matfp);
    }
    // template void saveEigenDenseMatReal2Mat(const char *filename, Eigen::MatrixBase<double> &K);
    // template void saveEigenDenseMatReal2Mat(const char *filename, Eigen::MatrixBase<Eigen::Index> &K);
}