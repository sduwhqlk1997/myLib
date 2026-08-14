#include <myEigen.hpp>
#include <myOMP.hpp>
namespace myEigen
{
    template <typename Scalar>
    SparseMat_t<Scalar> blkdiag(const std::vector<SparseMat_t<Scalar>> &matBloks)
    {
        if (matBloks.empty())
            return SparseMat_t<Scalar>();
        int sizeOmp = omp_get_max_threads();
        std::vector<std::pair<int, int>> omp_task = myOMP::distributeTasks(sizeOmp, matBloks.size());
        std::vector<Idx> posStartRow(matBloks.size(), 0), nnzStart(matBloks.size(), 0),
            posStartCol(matBloks.size(), 0);
        Idx nnz = matBloks[0].nonZeros();
        Idx totalRows = matBloks[0].rows();
        Idx totalCols = matBloks[0].cols();
        for (Idx i = 1; i < matBloks.size(); ++i)
        {
            totalRows += matBloks[i].rows();
            totalCols += matBloks[i].cols();
            nnzStart[i] = nnz;
            nnz += matBloks[i].nonZeros();
            posStartRow[i] = posStartRow[i - 1] + matBloks[i - 1].rows();
            posStartCol[i] = posStartCol[i - 1] + matBloks[i - 1].cols();
        }
        std::vector<Eigen::Triplet<Scalar, Idx>> triplets(nnz);
        // triplets.reserve(nnz);
#pragma omp parallel
        {
            Idx rankOmp = omp_get_thread_num();
            for (Idx i = omp_task[rankOmp].first; i < omp_task[rankOmp].first + omp_task[rankOmp].second; ++i)
            {
                Idx idxTemp = 0;
                for (Idx col = 0; col < matBloks[i].outerSize(); ++col)
                {
                    // 内层迭代器：遍历当前列所有非零元
                    for (typename SparseMat_t<Scalar>::InnerIterator iter(matBloks[i], col); iter; ++iter)
                    {
                        triplets[nnzStart[i] + idxTemp] =
                            Eigen::Triplet<Scalar, Idx>{iter.row() + posStartRow[i],
                                                        iter.col() + posStartCol[i], iter.value()};
                        idxTemp++;
                    }
                }
            }
        }
        SparseMat_t<Scalar> result(totalRows, totalCols);

        result.setFromTriplets(triplets.begin(), triplets.end());

        return result;
    }

    template <typename Scalar>
    SparseMat_t<Scalar> blkMat(const std::vector<std::vector<SparseMat_t<Scalar>>> &matBloks)
    {
        if (matBloks.empty())
            return SparseMat_t<Scalar>();
        int sizeOmp = omp_get_max_threads();
        int nrowBlk = matBloks.size();
        int ncolBlk = matBloks[0].size();
        int ntotalBlk = nrowBlk * ncolBlk;
        std::vector<std::pair<int, int>> omp_task = myOMP::distributeTasks(sizeOmp, ntotalBlk);
        auto I = [nrowBlk](int i, int j) -> int
        {
            return i + j * nrowBlk;
        };
        std::vector<Idx> posStartRow(ntotalBlk, 0), nnzStart(ntotalBlk, 0),
            posStartCol(ntotalBlk, 0);
        Idx nnz = matBloks[0][0].nonZeros();
        Idx totalRows = matBloks[0][0].rows();
        Idx totalCols = matBloks[0][0].cols();
        for (Idx i = 0; i < nrowBlk; ++i)
        {
            for (Idx j = 0; j < ncolBlk; ++j)
            {
                if (0 == i)
                {
                    posStartRow[I(i, j)] = 0;
                    if (j > 0)
                        totalCols += matBloks[i][j].cols();
                }
                else
                    posStartRow[I(i, j)] = posStartRow[I(i - 1, j)] + matBloks[i - 1][j].rows();
                if (0 == j)
                {
                    posStartCol[I(i, j)] = 0;
                    if (i > 0)
                        totalRows += matBloks[i][j].rows();
                }
                else
                    posStartCol[I(i, j)] = posStartCol[I(i, j - 1)] + matBloks[i][j - 1].cols();
                if (I(i, j) > 0)
                {
                    nnzStart[I(i, j)] = nnz;
                    nnz += matBloks[i][j].nonZeros();
                }
            }
        }
        std::vector<Eigen::Triplet<Scalar, Idx>> triplets(nnz);
#pragma omp parallel num_threads(sizeOmp)
        {
            Idx rankOmp = omp_get_thread_num();
            for (Idx i = omp_task[rankOmp].first; i < omp_task[rankOmp].first + omp_task[rankOmp].second; ++i)
            {
                Idx idxTemp = 0;
                int blk_i = i % nrowBlk;
                int blk_j = i / nrowBlk;
                for (Idx col = 0; col < matBloks[blk_i][blk_j].outerSize(); ++col)
                {
                    // 内层迭代器：遍历当前列所有非零元
                    for (typename SparseMat_t<Scalar>::InnerIterator iter(matBloks[blk_i][blk_j], col); iter; ++iter)
                    {
                        triplets[nnzStart[i] + idxTemp] =
                            Eigen::Triplet<Scalar, Idx>{iter.row() + posStartRow[i],
                                                        iter.col() + posStartCol[i], iter.value()};
                        idxTemp++;
                    }
                }
            }
        }
        SparseMat_t<Scalar> result(totalRows, totalCols);
        result.setFromTriplets(triplets.begin(), triplets.end());
        return result;
    }
    template <typename Scalar>
    SparseMat_t<Scalar>
    delColRowSpMat(
        const SparseMat_t<Scalar> &K,
        const std::set<Eigen::Index> &idxDel,
        Eigen::Index flag)
    {
        using SpMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor>;
        using Index = Eigen::Index;

        const Index nRows = K.rows();
        const Index nCols = K.cols();

        if (flag < 0 || flag > 2)
        {
            throw std::invalid_argument(
                "Flag只能为0、1、2，分别代表行、列、行和列");
        }

        // 标记需要删除的行/列
        std::vector<bool> delRow(nRows, false);
        std::vector<bool> delCol(nCols, false);

        for (const auto idx : idxDel)
        {
            if (flag == 0 || flag == 2)
            {
                if (idx < 0 || idx >= nRows)
                {
                    throw std::out_of_range(
                        "delColRowSpMat: row index out of range.");
                }
                delRow[idx] = true;
            }

            if (flag == 1 || flag == 2)
            {
                if (idx < 0 || idx >= nCols)
                {
                    throw std::out_of_range(
                        "delColRowSpMat: column index out of range.");
                }
                delCol[idx] = true;
            }
        }

        // 原始行/列 -> 新矩阵行/列的映射
        std::vector<Index> rowMap(nRows, -1);
        std::vector<Index> colMap(nCols, -1);

        Index newRows = 0;
        Index newCols = 0;

        for (Index i = 0; i < nRows; ++i)
        {
            if (!delRow[i])
            {
                rowMap[i] = newRows++;
            }
        }

        for (Index j = 0; j < nCols; ++j)
        {
            if (!delCol[j])
            {
                colMap[j] = newCols++;
            }
        }

        SpMat result(newRows, newCols);

        // 预估非零元数量
        result.reserve(K.nonZeros());

        for (Index k = 0; k < K.outerSize(); ++k)
        {
            for (typename SpMat::InnerIterator it(K, k); it; ++it)
            {
                const Index i = it.row();
                const Index j = it.col();

                // 如果该元素所在的行或列被删除，则跳过
                if (delRow[i] || delCol[j])
                {
                    continue;
                }

                result.insertBack(rowMap[i], colMap[j]) = it.value();
            }
        }

        result.makeCompressed();

        return result;
    }
    template <typename Scalar>
    Mat_t<Scalar>
    delColRowDenMat(const Mat_t<Scalar> &K, const std::set<Eigen::Index> &idxDel, Eigen::Index flag)
    {
        using Index = Eigen::Index;
        const Index nRows = K.rows();
        const Index nCols = K.cols();

        // 检查 flag
        if (flag < 0 || flag > 2)
        {
            throw std::invalid_argument(
                "Flag只能为0、1、2，分别代表行、列、行和列.");
        }

        // ============================================================
        // flag == 0：删除行
        // ============================================================
        if (flag == 0)
        {
            const Index newRows = nRows - static_cast<Index>(idxDel.size());

            Mat_t<Scalar> result(newRows, nCols);

            Index newRow = 0;

            for (Index i = 0; i < nRows; ++i)
            {
                if (idxDel.find(i) != idxDel.end())
                {
                    continue;
                }

                result.row(newRow) = K.row(i);
                ++newRow;
            }

            return result;
        }

        // ============================================================
        // flag == 1：删除列
        // ============================================================
        if (flag == 1)
        {
            const Index newCols = nCols - static_cast<Index>(idxDel.size());

            Mat_t<Scalar> result(nRows, newCols);

            Index newCol = 0;

            for (Index j = 0; j < nCols; ++j)
            {
                if (idxDel.find(j) != idxDel.end())
                {
                    continue;
                }

                result.col(newCol) = K.col(j);
                ++newCol;
            }

            return result;
        }

        // ============================================================
        // flag == 2：同时删除行和列
        // ============================================================
        const Index newRows = nRows - static_cast<Index>(idxDel.size());
        const Index newCols = nCols - static_cast<Index>(idxDel.size());

        Mat_t<Scalar> result(newRows, newCols);

        Index newRow = 0;

        for (Index i = 0; i < nRows; ++i)
        {

            if (idxDel.find(i) != idxDel.end())
            {
                continue;
            }

            Index newCol = 0;

            for (Index j = 0; j < nCols; ++j)
            {

                if (idxDel.find(j) != idxDel.end())
                {
                    continue;
                }

                result(newRow, newCol) = K(i, j);

                ++newCol;
            }

            ++newRow;
        }

        return result;
    }
    template SparseMat_t<Complex> blkdiag(const std::vector<SparseMat_t<Complex>> &matBloks);
    template SparseMat_t<double> blkdiag(const std::vector<SparseMat_t<double>> &matBloks);
    template SparseMat_t<Complex> blkMat(const std::vector<std::vector<SparseMat_t<Complex>>> &matBloks);
    template SparseMat_t<double> blkMat(const std::vector<std::vector<SparseMat_t<double>>> &matBloks);
    template SparseMat_t<double> delColRowSpMat(const SparseMat_t<double> &K, const std::set<Eigen::Index> &idxDel, Eigen::Index flag);
    template SparseMat_t<Complex> delColRowSpMat(const SparseMat_t<Complex> &K, const std::set<Eigen::Index> &idxDel, Eigen::Index flag);
    template Mat_t<double> delColRowDenMat(const Mat_t<double> &K, const std::set<Eigen::Index> &idxDel, Eigen::Index flag);
    template Mat_t<Idx> delColRowDenMat(const Mat_t<Idx> &K, const std::set<Eigen::Index> &idxDel, Eigen::Index flag);
    template Mat_t<Complex> delColRowDenMat(const Mat_t<Complex> &K, const std::set<Eigen::Index> &idxDel, Eigen::Index flag);
};