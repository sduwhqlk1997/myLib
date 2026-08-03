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

    template SparseMat_t<Complex> blkdiag(const std::vector<SparseMat_t<Complex>> &matBloks);
    template SparseMat_t<double> blkdiag(const std::vector<SparseMat_t<double>> &matBloks);
    template SparseMat_t<Complex> blkMat(const std::vector<std::vector<SparseMat_t<Complex>>> &matBloks);
    template SparseMat_t<double> blkMat(const std::vector<std::vector<SparseMat_t<double>>> &matBloks);
};