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
    template <typename Scalar>
    void addSpMatColOrRow(
        SparseMat_t<Scalar> &K,
        std::vector<Idx> Idx1,
        std::vector<Idx> Idx2,
        Idx flag)
    {
        using Triplet = Eigen::Triplet<Scalar>;

        if (Idx1.size() != Idx2.size())
        {
            throw std::invalid_argument(
                "Idx1 and Idx2 must have the same size");
        }

        if (flag != 0 && flag != 1 && flag != 2)
        {
            throw std::invalid_argument(
                "flag must be 0, 1 or 2");
        }

        Idx nPair = static_cast<Idx>(Idx1.size());

        // 保存原矩阵所有非零项
        std::vector<Triplet> triplets;
        triplets.reserve(K.nonZeros() * 2);

        //=========================================================
        // flag=0 或 flag=2
        //
        // K(Idx1,:) = K(Idx1,:) + K(Idx2,:)
        //
        //=========================================================
        if (flag == 0 || flag == 2)
        {
            // 保存所有需要增加的行元素
            std::vector<std::tuple<Idx, Idx, Scalar>> rowAdd;

            for (Idx k = 0; k < nPair; ++k)
            {
                Idx srcRow = Idx2[k];
                Idx dstRow = Idx1[k];

                // ColMajor，需要扫描所有列找源行
                for (Idx col = 0; col < K.cols(); ++col)
                {
                    Scalar val = K.coeff(srcRow, col);

                    if (val != Scalar(0))
                    {
                        rowAdd.emplace_back(
                            dstRow,
                            col,
                            val);
                    }
                }
            }

            // 添加原矩阵
            for (Idx col = 0; col < K.outerSize(); ++col)
            {
                for (typename SparseMat_t<Scalar>::InnerIterator it(K, col);
                     it;
                     ++it)
                {
                    triplets.emplace_back(
                        it.row(),
                        it.col(),
                        it.value());
                }
            }

            // 添加行增加项
            for (auto &[row, col, val] : rowAdd)
            {
                triplets.emplace_back(row, col, val);
            }
        }

        //=========================================================
        // flag=1 或 flag=2
        //
        // K(:,Idx1)=K(:,Idx1)+K(:,Idx2)
        //
        //=========================================================
        if (flag == 1 || flag == 2)
        {
            std::vector<std::tuple<Idx, Idx, Scalar>> colAdd;

            for (Idx k = 0; k < nPair; ++k)
            {
                Idx srcCol = Idx2[k];
                Idx dstCol = Idx1[k];

                // ColMajor，直接读取源列
                for (typename SparseMat_t<Scalar>::InnerIterator it(K, srcCol);
                     it;
                     ++it)
                {
                    colAdd.emplace_back(
                        it.row(),
                        dstCol,
                        it.value());
                }
            }

            // 如果前面没有加入原矩阵，则这里加入
            if (flag == 1)
            {
                for (Idx col = 0; col < K.outerSize(); ++col)
                {
                    for (typename SparseMat_t<Scalar>::InnerIterator it(K, col);
                         it;
                         ++it)
                    {
                        triplets.emplace_back(
                            it.row(),
                            it.col(),
                            it.value());
                    }
                }
            }

            // 添加列增加项
            for (auto &[row, col, val] : colAdd)
            {
                triplets.emplace_back(row, col, val);
            }
        }

        //=========================================================
        // 重构SparseMatrix
        //=========================================================
        SparseMat_t<Scalar> Knew(
            K.rows(),
            K.cols());

        Knew.setFromTriplets(
            triplets.begin(),
            triplets.end(),
            [](const Scalar &a, const Scalar &b)
            {
                return a + b;
            });

        Knew.prune(Scalar(0));
        K.swap(Knew);
    }
    template <typename Scalar>
    Vec_t<Scalar> delValDenVec(const Vec_t<Scalar> &V, const std::set<Eigen::Index> &idxDel)
    {
        using Idx = Eigen::Index;
        Idx n = V.size();

        // 新向量长度
        Idx newSize = n - static_cast<Idx>(idxDel.size());

        Vec_t<Scalar> Vnew(newSize);

        Idx count = 0;

        for (Idx i = 0; i < n; ++i)
        {
            // 如果 i 不在删除集合中，则保留
            if (idxDel.find(i) == idxDel.end())
            {
                Vnew(count++) = V(i);
            }
        }

        return Vnew;
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
    template void addSpMatColOrRow(SparseMat_t<double> &K, std::vector<Idx> Idx1, std::vector<Idx> Idx2, Idx flag);
    template void addSpMatColOrRow(SparseMat_t<Complex> &K, std::vector<Idx> Idx1, std::vector<Idx> Idx2, Idx flag);
    template Vec_t<double> delValDenVec(const Vec_t<double> &V, const std::set<Eigen::Index> &idxDel);
    template Vec_t<Idx> delValDenVec(const Vec_t<Idx> &V, const std::set<Eigen::Index> &idxDel);
    template Vec_t<int> delValDenVec(const Vec_t<int> &V, const std::set<Eigen::Index> &idxDel);
    template Vec_t<Complex> delValDenVec(const Vec_t<Complex> &V, const std::set<Eigen::Index> &idxDel);

};