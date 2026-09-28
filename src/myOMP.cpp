#include <myOMP.hpp>
#include <mkl.h>
#include <Eigen/Dense>
namespace myOMP
{
    std::vector<std::pair<int, int>> distributeTasks(int numThreads, int totalTasks)
    {
        // 返回值：first为对应线程的起始任务编号，second为任务数量
        int quo = totalTasks / numThreads;
        int rem = totalTasks % numThreads;
        std::vector<std::pair<int, int>> task_omp(numThreads);
        for (int i = 0; i < numThreads; ++i)
        {
            int count = rem < 1 ? quo : quo + 1;
            int start = i == 0 ? 0 : task_omp[i - 1].first + task_omp[i - 1].second;
            rem--;
            task_omp[i] = {start, count};
        }
        return task_omp;
    }
    void initOmpMklNestedParallel()
    {
        omp_set_dynamic(0);

        // 允许 nested OpenMP
        omp_set_max_active_levels(2);

        // 禁止 MKL 动态缩小线程数
        mkl_set_dynamic(0);
    }
    template <typename T>
    std::vector<int> assignThreadCount(int numThreads, std::vector<T> nJobs)
    {
        // 将numThreads仅可能成比例地分配给nJobs个任务，返回每个任务分配的线程数
        std::vector<int> threadCounts(nJobs.size(), 0);
        if (numThreads <= 0 || nJobs.empty())
            return threadCounts;

        long long totalJobs = 0;
        for (int jobs : nJobs)
            totalJobs += jobs > 0 ? jobs : 0;
        if (totalJobs == 0)
            return threadCounts;

        int activeJobs = 0;
        for (const auto jobs : nJobs)
            if (jobs > 0)
                ++activeJobs;
        // There are not enough threads to give every nonzero task one.
        if (numThreads < activeJobs)
            return threadCounts;
        const int remainingThreads = numThreads - activeJobs;

        std::vector<long long> remainders(nJobs.size(), 0);
        int assigned = activeJobs;
        for (size_t i = 0; i < nJobs.size(); ++i)
        {
            const long long jobs = nJobs[i] > 0 ? nJobs[i] : 0;
            if (jobs == 0)
                continue;
            threadCounts[i] = 1;
            const long long weightedThreads = static_cast<long long>(remainingThreads) * jobs;
            const int extraThreads = static_cast<int>(weightedThreads / totalJobs);
            threadCounts[i] += extraThreads;
            remainders[i] = weightedThreads % totalJobs;
            assigned += extraThreads;
        }

        // 按余数从大到小分配因整数取整而剩余的线程。
        while (assigned < numThreads)
        {
            size_t best = 0;
            for (size_t i = 1; i < remainders.size(); ++i)
            {
                if (remainders[i] > remainders[best])
                    best = i;
            }
            if (remainders[best] < 0)
                break;
            ++threadCounts[best];
            remainders[best] = -1;
            ++assigned;
        }
        return threadCounts;
    }
    template std::vector<int> assignThreadCount(int numThreads, std::vector<int> nJobs);
    template std::vector<int> assignThreadCount(int numThreads, std::vector<Eigen::Index> nJobs);
}